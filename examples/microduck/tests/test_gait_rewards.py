"""Standing and heading shaping must preserve commanded turns and recovery motion."""

import json
import math
import os
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from examples.microduck.official_recipe import SNAPSHOT, OfficialRecipe, RobotView
from examples.microduck.policy_spec import HOME


def recipe_for_test(count=5):
    recipe = OfficialRecipe.__new__(OfficialRecipe)
    recipe.env = SimpleNamespace(commands=torch.zeros(count, 13))
    recipe.robot = SimpleNamespace(
        data=SimpleNamespace(
            site_lin_vel_w=torch.zeros(count, 3, 3),
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(count, 1),
            site_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]).repeat(count, 3, 1),
            projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]).repeat(count, 1),
        )
    )
    return recipe


def test_standing_motion_uses_all_foot_axes_but_not_head_and_all_twist_commands():
    recipe = recipe_for_test(6)
    data = recipe.robot.data
    data.site_lin_vel_w[0, 0] = 5.0  # Head motion must remain free.
    data.site_lin_vel_w[1, 1:, 2] = 0.2  # Vertical stepping with a stationary trunk.
    data.site_lin_vel_w[2, 1:, 0] = 0.2  # Horizontal shuffle.
    data.site_lin_vel_w[3:, 1:] = 0.2
    recipe.env.commands[3, 2] = 0.5  # Turning in place is not standing.
    recipe.env.commands[4, 0] = 0.2
    data.projected_gravity_b[5] = torch.tensor([1.0, 0.0, 0.0])
    actual = recipe.base_reward(
        "standing_foot_motion_penalty",
        {
            "linear_threshold": 0.01,
            "angular_threshold": 0.01,
            "max_tilt_deg": 35.0,
        },
    )
    torch.testing.assert_close(actual, torch.tensor([0.0, 0.08, 0.08, 0.0, 0.0, 0.0]))


def test_standing_contact_prefers_both_feet_even_when_a_raised_foot_is_motionless():
    term = json.loads(SNAPSHOT.read_text())["tasks"]["velocity"]["rewards"]["standing_foot_contact"]
    recipe = recipe_for_test(10)
    # One foot may produce several matches; they cannot substitute for the other foot.
    found = torch.tensor([[4., 3.], [8., 0.], [0., 9.], [0., 0.]] + [[8., 0.]] * 6)
    recipe.scene = {"feet_ground_contact": SimpleNamespace(data=SimpleNamespace(found=found))}
    recipe.env.commands[4, 0] = 0.2
    recipe.env.commands[5, 1] = 0.2
    recipe.env.commands[6, 2] = 0.5
    recipe.robot.data.projected_gravity_b[7] = torch.tensor([1., 0., 0.])
    recipe.env.commands[8, 3:7] = 0.5  # Head commands do not release the feet from standing.
    recipe.env.commands[9, :2] = 0.008  # Combined translation exceeds the standing threshold.
    cost = recipe.base_reward(term["func"], term["params"])
    torch.testing.assert_close(cost, torch.tensor([0., 1., 1., 2., 0., 0., 0., 0., 1., 0.]))
    assert (cost[0] * term["weight"]) > (cost[1] * term["weight"]) > (cost[3] * term["weight"])
    # Foot-speed shaping alone cannot distinguish these stationary support patterns.
    motion_params = json.loads(SNAPSHOT.read_text())["tasks"]["velocity"]["rewards"]["standing_foot_motion"]["params"]
    assert not recipe.base_reward("standing_foot_motion_penalty", motion_params).any()


def yaw_quat(angle):
    return torch.tensor([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)])


def test_heading_relative_to_trunk_with_tolerance_and_turn_lateral_gates():
    recipe = recipe_for_test()
    data = recipe.robot.data
    heading = math.radians(179)
    data.root_link_quat_w[:] = yaw_quat(heading)
    for i in range(5):
        data.site_quat_w[i, 1] = yaw_quat(heading - math.radians(15))
        data.site_quat_w[i, 2] = yaw_quat(heading + math.radians(15))
    data.site_quat_w[0, 1] = yaw_quat(heading - math.radians(4))
    data.site_quat_w[0, 2] = yaw_quat(heading + math.radians(4))
    recipe.env.commands[1, 0] = 0.2
    recipe.env.commands[2, 2] = 0.5
    recipe.env.commands[3, 1] = 0.2
    data.projected_gravity_b[4] = torch.tensor([1.0, 0.0, 0.0])
    actual = recipe.base_reward(
        "straight_foot_heading_penalty",
        {
            "tolerance_deg": 5.0,
            "angular_threshold": 0.1,
            "lateral_threshold": 0.05,
            "max_tilt_deg": 35.0,
        },
    )
    torch.testing.assert_close(
        actual, torch.tensor([0.0, 2 * math.radians(10) ** 2, 0.0, 0.0, 0.0])
    )


def test_local_gait_rewards_scope_and_negative_weights():
    tasks = json.loads(SNAPSHOT.read_text())["tasks"]
    for name in ("standing_foot_motion", "straight_foot_heading", "standing_foot_contact"):
        term = tasks["velocity"]["rewards"][name]
        assert term["weight"] < 0 and term["local"] is True
        assert all(
            name not in spec["rewards"] for task, spec in tasks.items() if task != "velocity"
        )


def test_hip_roll_cost_targets_only_roll_and_still_distinguishes_large_deviations():
    term = json.loads(SNAPSHOT.read_text())["tasks"]["velocity"]["rewards"]["hip_roll_neutral"]
    recipe = OfficialRecipe.__new__(OfficialRecipe)
    recipe.env = SimpleNamespace()
    recipe.robot = RobotView()
    home = torch.tensor(HOME).repeat(5, 1)
    positions = home.clone()
    positions[1, [0, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13]] += 0.5
    positions[2, [1, 10]] += torch.tensor([-0.1, 0.1])
    positions[3, [1, 10]] += torch.tensor([-0.3, 0.3])
    positions[4, [1, 10]] += torch.tensor([-0.25, 0.25])
    recipe.robot.data = SimpleNamespace(joint_pos=positions, default_joint_pos=home)
    recipe.scene = {"robot": recipe.robot}
    params = recipe.decode(term["params"])
    assert params["asset_cfg"].joint_ids == [1, 10]
    costs = recipe.base_reward(term["func"].rsplit(".", 1)[-1], params)
    torch.testing.assert_close(costs, torch.tensor([0.0, 0.0, 0.2, 0.6, 0.5]))
    assert term["local"] is True and term["weight"] < 0
    # Even near the observed limit posture, a small correction must improve the reward.
    assert (costs[4] * term["weight"]) > (costs[3] * term["weight"])
    for task, spec in json.loads(SNAPSHOT.read_text())["tasks"].items():
        if task != "velocity":
            assert not spec["rewards"].get("hip_roll_neutral", {}).get("local", False)


def test_target_limit_cost_can_improve_even_when_physical_joints_remain_at_stops():
    tasks = json.loads(SNAPSHOT.read_text())["tasks"]
    term = tasks["velocity"]["rewards"]["action_over_limit"]
    recipe = OfficialRecipe.__new__(OfficialRecipe)
    recipe.env = SimpleNamespace()
    home = torch.tensor(HOME)
    targets = home.repeat(4, 1)
    targets[1, [1, 10]] = torch.tensor([-0.65, 0.65])  # Allow PD headroom beyond the hard stops.
    targets[2, [1, 10]] = torch.tensor([-0.9, 1.2])
    targets[3, [1, 10]] = torch.tensor([-0.75, 0.85])
    limits = torch.tensor([-2.0, 2.0]).repeat(4, 14, 1)
    limits[:, [1, 10]] = torch.tensor([-0.4, 0.4])
    physical = home.repeat(4, 1)
    physical[:, [1, 10]] = torch.tensor([-0.4, 0.4])
    recipe.robot = SimpleNamespace(data=SimpleNamespace(joint_pos=physical, joint_pos_limits=limits))
    recipe.scene = {"robot": recipe.robot}
    recipe.action_manager = SimpleNamespace(
        get_term=lambda name: SimpleNamespace(raw_action=targets - home, scale=1.0, offset=home, target_ids=list(range(14)))
    )
    cost = recipe.base_reward(term["func"].rsplit(".", 1)[-1], term["params"])
    torch.testing.assert_close(cost, torch.tensor([0.0, 0.0, 0.7, 0.2]))
    assert term["local"] is True and term["weight"] < 0
    assert cost[3] * term["weight"] > cost[2] * term["weight"]
    for task, spec in tasks.items():
        if task != "velocity":
            assert not spec["rewards"].get("action_over_limit", {}).get("local", False)


@pytest.mark.parametrize("step,expected_standing", [(0, 0.20), (12001, 0.25), (24001, 0.30), (48001, 0.35)])
def test_velocity_standing_curriculum_samples_idle_commands_without_losing_turns(step, expected_standing):
    spec = deepcopy(json.loads(SNAPSHOT.read_text())["tasks"]["velocity"])
    recipe = OfficialRecipe.__new__(OfficialRecipe)
    count = 20000
    recipe.spec = spec
    # Exercise the real twist sampler on CPU; other command groups do not affect its distribution.
    spec["commands"] = {"twist": spec["commands"]["twist"]}
    recipe.common_step_counter = step
    recipe.step_dt = 0.02
    recipe.device = torch.device("cpu")
    recipe.env = SimpleNamespace(commands=torch.zeros(count, 13), num_envs=count, family="velocity", device=recipe.device)
    recipe.standing = torch.zeros(count, dtype=torch.bool)
    recipe.timers = {"twist": torch.zeros(count)}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        recipe.update_commands(torch.arange(count))
    twist = recipe.env.commands[:, :3]
    zero = (twist == 0).all(-1)
    turn = (twist[:, :2] == 0).all(-1) & (twist[:, 2] != 0)
    turn_probability = spec["commands"]["twist"]["rel_turn_in_place_envs"]
    assert abs(zero.float().mean().item() - expected_standing * (1 - turn_probability)) < 0.01
    assert abs(turn.float().mean().item() - turn_probability) < 0.01
    assert torch.equal(recipe.standing, zero)
    assert (twist[~zero & ~turn, :2].norm(dim=-1) > 0).all()


def test_local_standing_curriculum_is_scoped_and_export_preserves_it():
    import ast
    from pathlib import Path

    tasks = json.loads(SNAPSHOT.read_text())["tasks"]
    curriculum = tasks["velocity"]["curricula"]["standing_envs"]
    assert curriculum["local"] is True
    assert all(not task["curricula"].get("standing_envs", {}).get("local")
               for name, task in tasks.items() if name != "velocity")
    # The exporter runs in upstream's separate venv; execute its dependency-free merge helper here.
    tree = ast.parse((Path(__file__).parents[1] / "export_official_recipe.py").read_text())
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "preserve_local_terms")
    namespace = {}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), "export_local_terms", "exec"), namespace)
    original = {"tasks": {"velocity": {"rewards": {"custom": {"local": True, "weight": -1},
                                                   "official": {"weight": 999}},
                                       "curricula": {"standing_envs": curriculum}}}}
    exported = {"tasks": {"velocity": {"rewards": {"official": {"weight": 2}},
                                       "curricula": {"standing_envs": {"params": {}}}}}}
    namespace["preserve_local_terms"](exported, original)
    assert exported["tasks"]["velocity"]["curricula"]["standing_envs"] == curriculum
    assert exported["tasks"]["velocity"]["rewards"] == {"official": {"weight": 2},
                                                         "custom": {"local": True, "weight": -1}}


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                os.environ.get("MICRODUCK_GPU_TESTS") != "1" or not torch.cuda.is_available(),
                reason="Set MICRODUCK_GPU_TESTS=1 on a CUDA host",
            ),
        ),
    ],
)
def test_gait_terms_registered_logged_and_reset_safe(device):
    from examples.microduck.env import MicroDuckEnv

    env = MicroDuckEnv(
        task="velocity", solver="mujoco", device=device, num_envs=1 if device == "cpu" else 2
    )
    try:
        assert env.cfg["official_alignment"]["local_curricula"]["standing_envs"]["local"] is True
        names = (
            "standing_foot_motion", "straight_foot_heading", "hip_roll_neutral", "action_over_limit", "standing_foot_contact"
        )
        for name in names:
            assert name in env.official.supported and name not in env.official.omitted
            assert name in env.cfg["official_alignment"]["local_rewards"]
        env.commands[:, :3] = 0
        # Validate the authored site axes at HOME before relying on their heading.
        env.official.refresh(*env._kinematics(), torch.zeros_like(env.actions), env.actions)
        torch.testing.assert_close(
            env.official.base_reward(
                "straight_foot_heading_penalty", env.official.params["straight_foot_heading"]
            ),
            torch.zeros(env.num_envs, device=device),
        )
        _, reward, _, extras = env.step(torch.zeros_like(env.actions))
        assert torch.isfinite(reward).all()
        for name in names:
            assert torch.isfinite(extras["log"]["Task/" + name])
            assert extras["log"]["Task/" + name] <= 0
        previous_q = env._views()[0].clone()
        env._reset(torch.tensor([0], device=device))
        torch.testing.assert_close(env._views()[0][1:], previous_q[1:])
        _, reward, _, _ = env.step(torch.zeros_like(env.actions))
        assert torch.isfinite(reward).all()
    finally:
        env.close()
