"""Contracts for the complete bundled task registry and added task adapters."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from examples.microduck.config import ppo_config
from examples.microduck.env import MicroDuckEnv
from examples.microduck.official_ppo import apply_official_ppo, microduck_symmetry
from examples.microduck.policy_spec import JOINT_NAMES, build_robot, joint_coordinates
from examples.microduck.task_registry import TASKS, task_key

ROOT = Path(__file__).parents[1]


def test_registry_exactly_matches_exported_official_ids():
    tasks = json.loads((ROOT / "configs/official_snapshot.json").read_text())["tasks"]
    assert len(TASKS) == 33
    assert set(TASKS) == set(tasks)
    assert {task_key(t["official_task_id"]) for t in tasks.values()} == set(TASKS)
    assert "spin_backlash" not in TASKS
    assert "rollerstandup_backlash" not in TASKS


@pytest.mark.parametrize("task,passive", [("velocity_backlash", 14), ("rollers", 4), ("rollers_backlash", 18)])
def test_passive_coordinates_are_distinct_from_servo_coordinates(task, passive):
    robot, qi, qdi = build_robot(task=task)
    coordinates = joint_coordinates(robot)
    assert len([n for n in coordinates if n.startswith("passive_")]) == passive
    assert len(set(qi)) == len(set(qdi)) == 14
    for i, name in enumerate(JOINT_NAMES):
        assert coordinates[name] == (qi[i], qdi[i])
        if task.endswith("_backlash"):
            assert coordinates[f"passive_{name}_backlash"] == (qi[i] + 1, qdi[i] + 1)


def test_encoder_includes_backlash_without_wheel_coordinates():
    env = SimpleNamespace(
        qi=torch.arange(14), qdi=torch.arange(14), backlash_qi=torch.arange(14, 28), backlash_qdi=torch.arange(14, 28)
    )
    q = torch.ones(2, 32)
    q[:, 14:28] = 0.02
    q[:, 28:] = 999
    position, velocity = MicroDuckEnv.encoder_state(env, q, q * 2)
    torch.testing.assert_close(position, torch.full((2, 14), 1.02))
    torch.testing.assert_close(velocity, torch.full((2, 14), 2.04))


def test_symmetry_is_involution_and_preserves_pitch_commands():
    obs = torch.randn(3, 61)
    actions = torch.randn(3, 14)
    mirrored, ma = microduck_symmetry(None, obs, actions)
    twice, ta = microduck_symmetry(None, mirrored[3:], ma[3:])
    torch.testing.assert_close(twice[3:], obs)
    torch.testing.assert_close(ta[3:], actions)
    torch.testing.assert_close(mirrored[3:, 51:53], obs[:, 51:53])


def test_task_ppo_keeps_roller_entropy_and_roulade_mirror_without_teachers():
    tasks = json.loads((ROOT / "configs/official_snapshot.json").read_text())["tasks"]
    assert apply_official_ppo(ppo_config(4), tasks["rollers"], 4)["algorithm"]["entropy_coef"] == 0.03
    cfg = apply_official_ppo(ppo_config(4), tasks["roulade"], 4)
    assert cfg["algorithm"]["symmetry_cfg"]["mirror_loss_coeff"] == 0.5
    cfg = apply_official_ppo(ppo_config(4), tasks["velstand"], 4)
    assert cfg["algorithm"]["class_name"] == "PPO"
    assert "bc_cfg" not in cfg["algorithm"]


def test_sitstand_binary_command_and_rate_limited_target():
    env = MicroDuckEnv(task="sitstand", device="cpu", num_envs=1)
    try:
        recipe = env.official
        assert env.commands[0, 0].item() in (0.0, 1.0)
        torch.testing.assert_close(env.commands[:, 1:3], torch.zeros(1, 2))
        env.episode_length_buf[:] = 10
        env.commands[:, 0] = 1
        recipe.posture_alpha[:] = 0
        for timer in recipe.timers.values():
            timer[:] = 100
        recipe.update_commands()
        torch.testing.assert_close(recipe.posture_alpha, torch.tensor([0.01]))
    finally:
        env.close()


def test_roulade_airborne_rotation_does_not_advance_frontier():
    from examples.microduck import official_reward_functions as reward

    env = MicroDuckEnv(task="roulade", device="cpu", num_envs=1)
    try:
        env.get_observations()
        recipe = env.official
        recipe._roulade_accum.zero_()
        recipe._roulade_max.zero_()
        recipe.robot.data.root_link_ang_vel_b[:, 1] = 3
        recipe.sensors.robot_ground.data.found.zero_()
        reward._update_roulade_accum(recipe, recipe.robot)
        assert recipe._roulade_max.item() == 0
        recipe.common_step_counter += 1
        recipe.robot.data.root_link_quat_w[:] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        recipe.sensors.robot_ground.data.found.fill_(1)
        reward._update_roulade_accum(recipe, recipe.robot)
        assert recipe._roulade_max.item() == pytest.approx(0.06)
        reward._update_roulade_accum(recipe, recipe.robot)
        assert recipe._roulade_max.item() == pytest.approx(0.06)
    finally:
        env.close()
