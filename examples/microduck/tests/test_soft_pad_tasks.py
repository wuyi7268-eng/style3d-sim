"""Terrain composition must preserve task resets, command meanings, and recovery horizons."""

import os
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from examples.microduck.common import setup
from examples.microduck.env import PAD_CONFIG, MicroDuckEnv
from examples.microduck.task_registry import TASKS


@pytest.mark.parametrize("task", TASKS)
def test_soft_pad_cli_preserves_selected_task(task):
    args, _, _ = setup("train", ["--solver", "synreal", "--task", task, "--terrain", "mixed_soft_pad"])
    assert args.task == task and args.terrain == "mixed_soft_pad"


def test_workbench_keeps_standup_when_selecting_soft_terrain(tmp_path):
    from examples.microduck.ui_backend import training_plan

    argv, _ = training_plan({"task": "standup", "environment_profile": "mixed_soft_pad"}, tmp_path)
    assert argv[argv.index("--task") + 1] == "standup"
    assert argv[argv.index("--environment-profile") + 1] == "mixed_soft_pad"


@pytest.mark.parametrize("task", ["standup", "sitstand", "roulade"])
def test_viewer_does_not_freeze_task_commands(task):
    env = MicroDuckEnv(task=task, device="cpu", num_envs=1)
    try:
        env.viewer = SimpleNamespace()
        env.render = lambda: None
        env.official.timers["twist"].fill_(2.0)
        env.step(torch.zeros(1, 14), apply_interval_events=False)
        assert env.official.timers["twist"].item() == pytest.approx(2.0 - env.step_dt)
    finally:
        env.viewer = None
        env.close()


@pytest.mark.parametrize("sampling", ["fixed_groups_center", "independent_per_episode"])
def test_pad_spawn_translation_keeps_ball_offset_and_surface_height(sampling):
    from examples.microduck.tests.test_mixed_terrain import terrain_fixture

    terrain, env = terrain_fixture(count=2)
    terrain.sampling = sampling
    terrain.pad_probability = 1.0
    env.task_logic = SimpleNamespace(spawn=torch.zeros(2, 28), ball_body=1, ball_q=21)
    spawn = env.task_logic.spawn
    spawn[:, :2] = torch.tensor([0.3, -0.2])
    spawn[:, 21:23] = spawn[:, :2] + torch.tensor([0.09, -0.042])
    spawn[:, 23] = 0.035
    terrain.reset(torch.arange(2))
    terrain.adjust_spawn(torch.arange(2))
    torch.testing.assert_close(spawn[:, 21:23] - spawn[:, :2], torch.tensor([[0.09, -0.042]]).expand(2, -1))
    torch.testing.assert_close(spawn[:, 23], torch.full((2,), 0.035 + terrain.config.top_z))


@pytest.mark.parametrize("task,expected_done", [("standup", False), ("velstand", False), ("velocity", True)])
def test_soft_surface_does_not_end_recovery_at_low_trunk_height(task, expected_done):
    env = MicroDuckEnv(task=task, device="cpu", num_envs=1)
    try:
        # Isolate termination composition from dynamics: a low trunk is a valid recovery state.
        env.official.common_step_counter = 96001
        env._views()[2][:, 2] = 0.05
        guard = SimpleNamespace(
            failed=torch.zeros(1, dtype=torch.bool), terminations=lambda failure, timeout: (failure, timeout)
        )
        env.terrain = SimpleNamespace(
            origins=torch.zeros(1, 3), guard=guard, update_support=lambda pose: None,
            heights=lambda positions, rotations=None: positions[..., 2].clamp_min(0),
            out_of_bounds=lambda pose: torch.zeros(1, dtype=torch.bool), log=lambda *args: {},
        )
        env.mixed_terrain = True
        env.advance_physics = lambda *args, **kwargs: torch.zeros(1, 14)
        env._reset = lambda ids: None
        _, reward, done, _ = env.step(torch.zeros(1, 14), apply_interval_events=False)
        assert torch.isfinite(reward).all()
        assert done.item() is expected_done
    finally:
        env.close()


@pytest.mark.skipif(os.environ.get("MICRODUCK_GPU_TESTS") != "1", reason="Requires SynReal login and CUDA")
def test_standup_soft_pad_keeps_fallen_starts_and_partial_reset_isolation():
    import warp as wp

    from examples.microduck.evaluate_standup import START_POSES, select_start_pose

    env = MicroDuckEnv(solver="synreal", task="standup", num_envs=2, terrain="mixed_soft_pad",
                       pad_probability=0.5, pad_config=replace(PAD_CONFIG, size=(1.2, 0.8, 0.06), cells=(8, 8, 3)))
    try:
        env.official.common_step_counter = 96001
        assert env.num_privileged_obs == 68
        for pose in START_POSES:
            select_start_pose(env.official, pose)
            env.reset()
            height = env._views()[2][:, 2] - env.terrain.origins[:, 2]
            assert (height >= 0.049).all() and (height <= 0.121).all()
            for _ in range(5):
                obs, reward, done, _ = env.step(torch.zeros(2, 14, device=env.device), apply_interval_events=False)
                assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
                assert not done.any(), "Fallen starts must retain their recovery horizon on either surface"
        before = [wp.to_torch(state.body_q).reshape(2, -1, 7)[0].clone() for state in env.states]
        env._reset(torch.tensor([1], device=env.device))
        for state, flat in zip(env.states, before, strict=True):
            torch.testing.assert_close(wp.to_torch(state.body_q).reshape(2, -1, 7)[0], flat, rtol=0, atol=0)
    finally:
        env.close()
