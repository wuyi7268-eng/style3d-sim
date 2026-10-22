"""Regression coverage for asymmetric observations and scheduled recovery resets."""

import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from examples.microduck.official_events import prepare_reset, scheduled_params, termination
from examples.microduck.official_observations import ObservationDelay, OfficialObservations
from examples.microduck.policy_spec import HOME

ROOT = Path(__file__).parents[1]


def recipe_stub(task, count=256):
    spec = json.loads((ROOT / "configs/official_snapshot.json").read_text())["tasks"][task]
    q = torch.zeros(count, 28)
    q[:, 6] = 1
    q[:, 7:21] = torch.tensor(HOME)
    env = SimpleNamespace(
        initial_q=q,
        num_envs=count,
        device="cpu",
        task=task,
        qi=torch.arange(7, 21),
        home=torch.tensor(HOME),
        actions=torch.zeros(count, 14),
        commands=torch.zeros(count, 13),
        task_logic=SimpleNamespace(spawn=q.clone(), ball_q=21),
    )
    scene = SimpleNamespace(env_origins=torch.zeros(count, 3))
    return SimpleNamespace(
        spec=spec,
        env=env,
        scene=scene,
        common_step_counter=0,
        limit=torch.tensor([[-1.57, 1.57]] * 14),
        num_envs=count,
        device="cpu",
        step_dt=0.02,
        episode_length_buf=torch.full((count,), 2),
    )


def test_reset_formulas_keep_upstream_ast():
    source = ast.parse((ROOT / "microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py").read_text())
    original = {n.name: n for n in source.body if isinstance(n, ast.FunctionDef)}
    copied = ast.parse((ROOT / "official_reset_functions.py").read_text())
    for node in copied.body:
        if isinstance(node, ast.FunctionDef):
            assert ast.dump(node) == ast.dump(original[node.name])


def test_delay_has_one_step_history_and_partial_reset():
    term = dict(  # noqa: C408
        delay_min_lag=1, delay_max_lag=1, delay_update_period=0, delay_per_env=True, delay_hold_prob=0, delay_per_env_phase=True
    )
    delay = ObservationDelay(term, 2, "cpu")
    torch.testing.assert_close(delay(torch.tensor([[1.0], [10.0]])), torch.tensor([[1.0], [10.0]]))
    torch.testing.assert_close(delay(torch.tensor([[2.0], [20.0]])), torch.tensor([[1.0], [10.0]]))
    delay.reset(torch.tensor([0]))
    torch.testing.assert_close(delay(torch.tensor([[3.0], [30.0]])), torch.tensor([[3.0], [20.0]]))


def test_recovery_stage_boundary_and_partial_reset():
    torch.manual_seed(42)
    recipe = recipe_stub("velstand")
    recipe.common_step_counter = 33600
    params = scheduled_params(recipe, "random_prone_init", "events")
    assert params["prone_prob"] == 0.45
    untouched = recipe.env.task_logic.spawn[128:].clone()
    prepare_reset(recipe, torch.arange(128))
    q = recipe.env.task_logic.spawn
    torch.testing.assert_close(q[128:], untouched)
    torch.testing.assert_close(q[:128, 3:7].norm(dim=-1), torch.ones(128))
    # Late curriculum includes SO(3), side, crouched and sitting starts, not only standing yaw.
    assert (q[:128, 3:5].norm(dim=-1) > 0.1).sum() > 35
    assert q[:128, :2].abs().max() <= 0.5
    assert (q[:128, 7:21] - recipe.env.home).abs().max() > 0.5


def test_ball_reset_follows_root_yaw_and_position():
    recipe = recipe_stub("ballkick")
    prepare_reset(recipe, torch.arange(256))
    q = recipe.env.task_logic.spawn
    quat = q[:, 3:7]
    yaw = torch.atan2(
        2 * (quat[:, 3] * quat[:, 2] + quat[:, 0] * quat[:, 1]), 1 - 2 * (quat[:, 1].square() + quat[:, 2].square())
    )
    d = q[:, 21:23] - q[:, :2]
    local = torch.stack((yaw.cos() * d[:, 0] + yaw.sin() * d[:, 1], -yaw.sin() * d[:, 0] + yaw.cos() * d[:, 1]), -1)
    assert ((local - torch.tensor([0.09, -0.042])).abs() <= 0.015001).all()
    torch.testing.assert_close(recipe._ball_kick_dir_w, torch.stack((yaw.cos(), yaw.sin()), -1))


def test_fell_over_disabled_at_official_boundary_and_timer():
    recipe = recipe_stub("velstand", 2)
    pose = torch.tensor([[0.0, 0.0, 0.06], [0.0, 0.0, 0.12]])
    quat = torch.tensor([[math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    recipe.robot = SimpleNamespace(data=SimpleNamespace(root_link_pos_w=pose, root_link_quat_w=quat))
    recipe.scene = {"robot": recipe.robot}
    recipe.scene = type("Scene", (dict,), {})(recipe.scene)
    recipe.scene.terrain = SimpleNamespace(env_origins=torch.zeros(2, 3))
    gravity = torch.tensor([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    assert termination(recipe, gravity).tolist() == [True, False]
    recipe.common_step_counter = 2400
    assert termination(recipe, gravity).tolist() == [False, False]
    recipe._fallen_timer_s[:] = 7.99
    assert termination(recipe, gravity).tolist() == [True, False]


@pytest.mark.parametrize("task,expected", [("velocity", 70), ("velstand", 70), ("groundpick", 68)])
def test_actor_corruption_clean_privileged_critic_and_cache(task, expected):
    recipe = recipe_stub(task, 2)
    env = recipe.env
    q = env.initial_q
    qd = torch.zeros(2, 20)
    qd[:, 6:20] = 2
    pose = q[:, :7]
    env.qdi = torch.arange(6, 20)
    env._kinematics = lambda: (q, qd, pose, torch.ones(2, 3), torch.zeros(2, 3), torch.tensor([[0.0, 0.0, -1.0]] * 2))
    recipe.refresh = lambda *args: None
    recipe.sensors = SimpleNamespace(
        feet=SimpleNamespace(data=SimpleNamespace(current_air_time=torch.ones(2, 2), found=torch.ones(2, 2))),
        height=SimpleNamespace(data=SimpleNamespace(heights=torch.ones(2, 2) * 0.1)),
    )
    observations = OfficialObservations(recipe)
    actor, critic = observations.compute()
    assert critic.shape == (2, expected)
    torch.testing.assert_close(critic[:, :3], torch.ones(2, 3))
    torch.testing.assert_close(critic[:, 9:23], torch.zeros(2, 14))
    assert actor[:, 6:20].abs().max() > 0
    steps = observations.delays["actor", "joint_vel"].steps.clone()
    again, _ = observations.compute()
    torch.testing.assert_close(actor, again)
    torch.testing.assert_close(steps, observations.delays["actor", "joint_vel"].steps)
    observations.cache = None
    qd[:, 6:20] = 7
    delayed, clean = observations.compute()
    assert delayed[:, 20:34].max() < 2.251
    torch.testing.assert_close(clean[:, 23:37], torch.full((2, 14), 7.0))


@pytest.mark.parametrize("task", ["velocity", "velstand", "groundpick"])
def test_fast_actor_matches_full_observations_with_delays_and_resets(task):
    import copy

    from examples.microduck.playback import PolicyPlayback

    recipe = recipe_stub(task, 2)
    env = recipe.env
    env.official = recipe
    q = env.initial_q
    qd = torch.zeros(2, 20)
    env.qdi = torch.arange(6, 20)
    env._kinematics = lambda: (q, qd, q[:, :7], torch.zeros(2, 3), torch.ones(2, 3), torch.tensor([[0.0, 0.0, -1.0]] * 2))
    recipe.refresh = lambda *args: None
    recipe.sensors = SimpleNamespace(
        feet=SimpleNamespace(data=SimpleNamespace(current_air_time=torch.ones(2, 2), found=torch.ones(2, 2))),
        height=SimpleNamespace(data=SimpleNamespace(heights=torch.ones(2, 2))),
    )
    recipe.observations = OfficialObservations(recipe)
    fast = PolicyPlayback(env)
    for step in range(6):
        if step == 3:
            recipe.observations.reset(torch.tensor([0]))
        qd[:, 6:] = step
        processor = recipe.observations
        processor.cache = None
        delays = copy.deepcopy(processor.delays)
        rng = torch.get_rng_state()
        expected = processor.compute()[0].clone()
        processor.delays = delays
        processor.cache = None
        torch.set_rng_state(rng)
        torch.testing.assert_close(fast.observe(), expected, atol=0, rtol=0)
