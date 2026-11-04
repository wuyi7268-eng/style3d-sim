"""StandUp recovery contracts, late reward stages and evaluation without reset-induced successes."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from examples.microduck.common import create_env, setup
from examples.microduck.env import MicroDuckEnv
from examples.microduck.evaluate_standup import START_POSES, evaluate, select_start_pose, standing_mask
from examples.microduck.events import prepare_reset, scheduled_params, termination
from examples.microduck.ppo import apply_official_ppo, ppo_config
from examples.microduck.tests.test_observations_events import recipe_stub

ROOT = Path(__file__).parents[1]


def test_standup_training_preset_preserves_task_ppo():
    args, template, _ = setup("train", ["--training-config", str(ROOT / "configs/rtx4090_synreal_standup.json")])
    spec = json.loads((ROOT / "configs/official_snapshot.json").read_text())["tasks"]["standup"]
    assert (args.solver, args.task, args.terrain, args.iterations) == ("synreal", "standup", "default", 15000)
    assert args.resume is args.init_from is None
    assert not template.get("ppo")
    cfg = apply_official_ppo(ppo_config(args.num_envs), spec, args.num_envs)
    assert cfg["num_steps_per_env"] == 24 and cfg["save_interval"] == 250
    assert cfg["algorithm"]["schedule"] == "adaptive"
    assert cfg["empirical_normalization"]


@pytest.mark.parametrize("iteration,expected", [
    (0, [0.40, 0.40, 0.20, 0.00]),
    (600, [0.25, 0.30, 0.35, 0.10]),
    (1500, [0.20, 0.25, 0.30, 0.25]),
    (2500, [0.15, 0.20, 0.30, 0.35]),
])
def test_ground_pose_curriculum_and_no_fall_termination(iteration, expected):
    recipe = recipe_stub("standup")
    recipe.common_step_counter = iteration * 24
    params = scheduled_params(recipe, "set_ground_state", "events")
    assert [params[f"{name}_prob"] for name in START_POSES] == expected
    # Both a horizontal and an inverted robot need time to recover.
    assert not termination(recipe, torch.tensor([[1., 0., 0.], [0., 0., 1.]])).any()


@pytest.mark.parametrize("pose", START_POSES)
def test_forced_upstream_reset_pose_and_partial_reset(pose):
    torch.manual_seed(42)
    recipe = recipe_stub("standup", 32)
    recipe.common_step_counter = 96001
    select_start_pose(recipe, pose)
    before = recipe.env.task_logic.spawn.clone()
    ids = torch.arange(16)
    prepare_reset(recipe, ids)
    q = recipe.env.task_logic.spawn
    torch.testing.assert_close(q[16:], before[16:])
    torch.testing.assert_close(q[:16, 3:7].norm(dim=-1), torch.ones(16))
    x, y, z, w = q[:16, 3:7].unbind(-1)
    upright = 1 - 2 * (x.square() + y.square())
    if pose in ("standing", "sitting"):
        assert (upright > 0.96).all()
        if pose == "standing":
            assert ((q[:16, 2] >= 0.11) & (q[:16, 2] <= 0.12)).all()
            torch.testing.assert_close(q[:16, 7:21], recipe.env.home.expand(16, -1))
        else:
            assert (q[:16, 2] <= 0.09).all()
            assert (q[:16, 10] > 0.9).all() and (q[:16, 19] < -0.9).all()
    else:
        assert (upright.abs() < 1e-6).all()
        assert ((q[:16, 2] >= 0.05) & (q[:16, 2] <= 0.09)).all()
        gravity_x = 2 * (w * y - x * z)
        assert (gravity_x > 0.99).all() if pose == "face_down" else (gravity_x <= 0).all()
    assert scheduled_params(recipe, "set_ground_state", "events")[f"{pose}_prob"] == 1


def test_all_standup_rewards_execute_at_final_curriculum():
    env = MicroDuckEnv(task="standup", device="cpu", num_envs=1)
    try:
        assert (env.num_obs, env.num_privileged_obs, env.max_episode_length) == (61, 68, 300)
        assert len(env.official.supported) == 20 and not env.official.omitted
        env.official.common_step_counter = 96001
        for pose in START_POSES:
            select_start_pose(env.official, pose)
            env.reset()
            obs, reward, done, extras = env.step(torch.zeros(1, 14), apply_interval_events=False)
            assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
            assert not done.any()
            # These terms are initially disabled; a bootstrap-only smoke test misses them.
            for term in ("arrival_damping", "head_pose_bias", "body_pose_tracking", "joint_torque_rate_l2"):
                assert f"Task/{term}" in extras["log"]
    finally:
        env.close()


def test_recovery_requires_height_orientation_and_both_feet():
    height = torch.tensor([0.115, 0.06, 0.115, 0.115])
    gravity = torch.tensor([[0., 0., -1.], [0., 0., -1.], [1., 0., 0.], [0., 0., -1.]])
    feet = torch.tensor([[1., 1.], [1., 1.], [1., 1.], [1., 0.]])
    assert standing_mask(height, gravity, feet).tolist() == [True, False, False, False]


@pytest.mark.skipif(os.environ.get("MICRODUCK_GPU_TESTS") != "1", reason="Requires SynReal login and CUDA")
def test_synreal_standup_late_curriculum_all_start_poses():
    args, _, _ = setup("train", ["--solver", "synreal", "--task", "standup", "--num-envs", "8"])
    env = create_env(args)
    try:
        assert type(env.solver).__module__ == "style3d_sim.newton_solver"
        env.official.common_step_counter = 96001
        for pose in START_POSES:
            select_start_pose(env.official, pose)
            env.reset()
            for _ in range(20):
                obs, reward, done, extras = env.step(torch.zeros(8, 14), apply_interval_events=False)
                assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
                assert not done.any(), "A fallen start must retain its recovery horizon"
                assert "Task/body_pose_tracking" in extras["log"]
    finally:
        env.close()


def test_evaluation_does_not_count_flicker_or_auto_reset_as_recovery():
    recipe = recipe_stub("standup", 3)
    recipe.observations = SimpleNamespace(spec={"actor": {"enable_corruption": True}})
    recipe.sensors = SimpleNamespace(feet=SimpleNamespace(data=SimpleNamespace(found=torch.ones(3, 2))))
    env = SimpleNamespace(
        family="standup", mixed_terrain=False, max_episode_length=4, step_dt=0.02, num_envs=3,
        device="cpu", viewer=None, official=recipe, commands=torch.ones(3, 13),
        terrain=SimpleNamespace(origins=torch.zeros(3, 3)),
    )
    frame = 0
    root = torch.zeros(3, 7)

    def reset():
        nonlocal frame
        frame = 0
        recipe.update_commands()

    def step(actions, *, apply_interval_events):
        nonlocal frame
        assert not apply_interval_events and not env.commands.any()
        frame += 1
        # First world stands continuously, second flickers, third auto-resets to a standing pose.
        root[:, 2] = torch.tensor([0.115, 0.115 if frame % 2 else 0.06, 0.115])
        return torch.zeros(3, 61), None, torch.tensor([False, False, frame == 1]), {}

    env.reset = reset
    env.get_observations = lambda: (torch.zeros(3, 61), {})
    env.step = step
    env._kinematics = lambda: (None, None, root, None, None, torch.tensor([[0., 0., -1.]] * 3))
    report = evaluate(env, lambda obs: torch.zeros(3, 14), poses=("face_down",), episodes=1, hold_seconds=0.04)
    assert report["face_down"]["recovered"] == report["face_down"]["standing_at_end"] == 1
    assert report["face_down"]["mean_time_to_held_stand_s"] == pytest.approx(0.04)
    assert env.max_episode_length == 4
