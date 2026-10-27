"""Local walking shaping must penalize vertical motion without taxing recovery tasks."""

import json
from types import SimpleNamespace

import torch

from examples.microduck.official_recipe import OfficialRecipe, SNAPSHOT


def test_vertical_cost_and_weighted_total():
    recipe = OfficialRecipe.__new__(OfficialRecipe)
    recipe.spec = {"rewards": {}, "curricula": {}}
    rewards = json.loads(SNAPSHOT.read_text())["tasks"]["velocity"]["rewards"]
    recipe.spec["rewards"]["vertical_body_velocity"] = rewards["vertical_body_velocity"]
    recipe.supported = {"vertical_body_velocity": "vertical_body_velocity_penalty"}
    recipe.params = {"vertical_body_velocity": {}}
    recipe.robot = SimpleNamespace(data=SimpleNamespace(
        root_link_lin_vel_w=torch.tensor([[0., 0., 0.], [1., 2., 0.], [0., 0., .5], [0., 0., -.5]])
    ))
    recipe.env = SimpleNamespace()
    recipe.refresh = lambda *args: None
    recipe.extras = {"log": {}}
    recipe.num_envs, recipe.device, recipe.step_dt = 4, "cpu", .02
    recipe.common_step_counter = 0
    total, terms = recipe.reward(*([None] * 8))
    torch.testing.assert_close(terms["vertical_body_velocity"], torch.tensor([0., 0., -.25, -.25]))
    torch.testing.assert_close(total, torch.tensor([0., 0., -.005, -.005]))


def test_local_reward_does_not_change_recovery_or_other_tasks():
    for task in ("velstand", "standup", "ballkick", "velocity_rough", "velocity_backlash"):
        spec = json.loads(SNAPSHOT.read_text())["tasks"][task]
        assert "vertical_body_velocity" not in spec["rewards"]


def test_vertical_reward_registered_and_logged_in_cpu_environment():
    from examples.microduck.env import MicroDuckEnv

    env = MicroDuckEnv(task="velocity", solver="mujoco", device="cpu", num_envs=1)
    try:
        recipe = env.official
        assert "vertical_body_velocity" in recipe.supported
        assert "vertical_body_velocity" not in recipe.omitted
        assert recipe.alignment["local_rewards"]["vertical_body_velocity"]["weight"] == -1.0
        for _ in range(3):
            _, reward, _, extras = env.step(torch.zeros_like(env.actions))
            assert torch.isfinite(reward).all()
            value = extras["log"]["Task/vertical_body_velocity"]
            assert torch.isfinite(value) and value <= 0
    finally:
        env.close()
