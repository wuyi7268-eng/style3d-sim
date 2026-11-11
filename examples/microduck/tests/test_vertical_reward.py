"""Disabled walking shaping stays off across curriculum boundaries; formulas remain available."""

import json
from types import SimpleNamespace

import pytest
import torch

from examples.microduck.recipe import SNAPSHOT, OfficialRecipe


@pytest.mark.parametrize("task", ["velocity", "velstand"])
@pytest.mark.parametrize("terrain", ["default", "mixed_soft_pad", "cloth"])
def test_relaxed_weights_remain_effective_across_surfaces_and_curricula(task, terrain):
    from examples.microduck.common import setup

    args, _, _ = setup("train", ["--solver", "synreal", "--task", task, "--terrain", terrain])
    assert args.task == task and args.terrain == terrain
    spec = json.loads(SNAPSHOT.read_text())["tasks"][args.task]
    disabled = {"commanded_foot_placement", "foot_swing_progress", "standing_foot_motion",
                "standing_foot_contact", "vertical_body_velocity"}
    fixed = {"track_yaw_fine": 1.0, "body_ang_vel": -0.05,
             "neck_action_rate": -0.025, "head_angular_velocity": -0.025}
    ramped = {"hip_roll_neutral", "straight_foot_heading"}
    selected = disabled | fixed.keys() | ramped
    recipe = OfficialRecipe.__new__(OfficialRecipe)
    recipe.spec = spec
    recipe.supported = {name: name for name in selected}
    recipe.params = {name: {} for name in selected}
    recipe.num_envs, recipe.device, recipe.step_dt = 1, "cpu", .02
    recipe.env = SimpleNamespace(
        mixed_terrain=terrain == "mixed_soft_pad",
        terrain=SimpleNamespace(kind=terrain, guard=SimpleNamespace(failed=torch.zeros(1, dtype=torch.bool))),
    )
    recipe.extras = {"log": {}}
    recipe.refresh = lambda *args: None

    def unit_cost(name, params):
        # Exercise the production weight dispatcher without running deformable physics.
        assert name not in disabled, "Disabled terms must not be evaluated, even after a curriculum transition"
        return torch.ones(1)

    recipe.base_reward = unit_cost
    steps = {0, 1000000}
    for cfg in spec["curricula"].values():
        for stage in cfg["params"].get("weight_stages", []):
            steps.update((stage["step"], stage["step"] + 1))
    for step in sorted(steps):
        recipe.common_step_counter = step
        total, terms = recipe.reward(*([None] * 8))
        assert not disabled.intersection(terms)
        for name, weight in fixed.items():
            assert terms[name].item() == pytest.approx(weight)
        for name in ramped:
            assert -.150001 <= terms[name].item() < 0
            if step == 0 or step == 1000000:
                assert terms[name].item() == pytest.approx(-.015 if step == 0 else -.15)
        torch.testing.assert_close(total, sum(terms.values()) * recipe.step_dt)


@pytest.mark.parametrize("step", [0, 24000, 24001, 36001, 48001, 60001, 72000, 72001, 1000000])
@pytest.mark.parametrize("task", ["velocity", "velstand"])
def test_vertical_cost_and_weighted_total(step, task):
    recipe = OfficialRecipe.__new__(OfficialRecipe)
    recipe.spec = {"rewards": {}, "curricula": {}}
    velocity = json.loads(SNAPSHOT.read_text())["tasks"][task]
    recipe.spec["rewards"]["vertical_body_velocity"] = velocity["rewards"]["vertical_body_velocity"]
    recipe.spec["curricula"]["vertical_body_velocity_weight"] = velocity["curricula"]["vertical_body_velocity_weight"]
    recipe.supported = {"vertical_body_velocity": "vertical_body_velocity_penalty"}
    recipe.params = {"vertical_body_velocity": {}}
    recipe.robot = SimpleNamespace(data=SimpleNamespace(
        root_link_lin_vel_w=torch.tensor([[0., 0., 0.], [1., 2., 0.], [0., 0., .5], [0., 0., -.5]])
    ))
    recipe.env = SimpleNamespace()
    recipe.refresh = lambda *args: None
    recipe.extras = {"log": {}}
    recipe.num_envs, recipe.device, recipe.step_dt = 4, "cpu", .02
    recipe.common_step_counter = step
    total, terms = recipe.reward(*([None] * 8))
    torch.testing.assert_close(recipe.base_reward("vertical_body_velocity_penalty", {}), torch.tensor([0., 0., .25, .25]))
    assert "vertical_body_velocity" not in terms
    torch.testing.assert_close(total, torch.zeros(4))


def test_local_reward_does_not_change_recovery_or_other_tasks():
    for task in ("standup", "ballkick", "velocity_rough", "velocity_backlash"):
        spec = json.loads(SNAPSHOT.read_text())["tasks"][task]
        assert "vertical_body_velocity" not in spec["rewards"]


@pytest.mark.parametrize("task", ["velocity", "velstand"])
def test_vertical_reward_registered_and_logged_in_cpu_environment(task):
    from examples.microduck.env import MicroDuckEnv

    env = MicroDuckEnv(task=task, solver="mujoco", device="cpu", num_envs=1)
    try:
        recipe = env.official
        assert "vertical_body_velocity" in recipe.supported
        assert "vertical_body_velocity" not in recipe.omitted
        assert recipe.alignment["local_rewards"]["vertical_body_velocity"]["weight"] == 0
        assert recipe.alignment["local_curricula"]["vertical_body_velocity_weight"]["local"] is True
        for _ in range(3):
            _, reward, _, extras = env.step(torch.zeros_like(env.actions))
            assert torch.isfinite(reward).all()
            assert "Task/vertical_body_velocity" not in extras["log"]
    finally:
        env.close()
