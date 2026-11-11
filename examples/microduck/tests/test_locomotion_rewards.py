"""Deformable-relative slip, signed command progress, and task/episode boundaries."""

import ast
import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp

from examples.microduck.adapter import SceneEntityCfg
from examples.microduck.recipe import SNAPSHOT
from examples.microduck.reward_functions import (
    CommandedFootSwingHeight,
    CommandProgress,
    commanded_foot_clearance,
    commanded_foot_placement_reward,
    feet_surface_slip,
    head_angular_velocity_penalty,
    track_yaw_rate_fine,
)
from examples.microduck.sensors import ContactSensor, _soft_contact_flags
from examples.microduck.velocity_rewards import feet_slip


def test_velocity_fine_yaw_tracking_prices_drift_without_blocking_commanded_turns():
    rewards = json.loads(SNAPSHOT.read_text())["tasks"]["velocity"]["rewards"]
    term = rewards["track_yaw_fine"]
    command = torch.tensor([[0.2, 0., 0.], [0.2, 0., 0.], [0., 0., 0.8], [0., 0., -0.8]])
    angular = torch.tensor([[0., 0., 0.], [0., 0., 0.1], [1., 2., 0.8], [-1., -2., -0.8]])
    recipe = SimpleNamespace(
        scene={"robot": SimpleNamespace(data=SimpleNamespace(root_link_ang_vel_b=angular))},
        command_manager=SimpleNamespace(get_command=lambda name: command),
    )
    fine = track_yaw_rate_fine(recipe, **term["params"]) * term["weight"]
    # A small sustained error must matter, while matching either turn direction earns full credit.
    assert (fine[0] - fine[1]).item() > 0.075
    torch.testing.assert_close(fine[[0, 2, 3]], torch.full((3,), term["weight"]))
    coarse = rewards["track_angular_velocity"]
    assert coarse["weight"] == 2.0
    assert coarse["params"]["std"] == pytest.approx(0.5 ** 0.5)


@pytest.mark.parametrize("task", ["velocity", "velstand"])
def test_velocity_fine_yaw_tracking_reaches_environment_reward_and_logs(task):
    from examples.microduck.env import MicroDuckEnv

    env = MicroDuckEnv(solver="mujoco", num_envs=1, device="cpu", task=task)
    try:
        assert env.official.supported["track_yaw_fine"] == "track_yaw_rate_fine"
        _, reward, _, extras = env.step(torch.zeros(1, 14))
        assert torch.isfinite(reward).all()
        value = extras["log"]["Task/track_yaw_fine"]
        assert 0 <= value <= env.official.spec["rewards"]["track_yaw_fine"]["weight"]
        assert env.cfg["official_alignment"]["local_rewards"]["track_yaw_fine"]["local"]
    finally:
        env.close()


def test_head_stability_uses_world_motion_and_gates_recovery_commands_and_yaw():
    from examples.microduck.recipe import RobotView

    count = 6
    robot = RobotView()
    robot.body_names = ["trunk_base", "jaw_soft"]
    robot.data = SimpleNamespace(
        root_link_pos_w=torch.tensor([[0., 0., 0.12]]).repeat(count, 1),
        projected_gravity_b=torch.tensor([[0., 0., -1.]]).repeat(count, 1),
        body_link_ang_vel_w=torch.zeros(count, 2, 3),
    )

    class Scene(dict):
        pass

    scene = Scene(robot=robot)
    scene.terrain = SimpleNamespace(env_origins=torch.zeros(count, 3))
    command = torch.zeros(count, 4)
    recipe = SimpleNamespace(
        scene=scene, num_envs=count, device="cpu", step_dt=0.02, extras={"log": {}},
        command_manager=SimpleNamespace(get_command=lambda name: command),
    )
    params = json.loads(SNAPSHOT.read_text())["tasks"]["velocity"]["rewards"]["head_angular_velocity"]["params"]
    # A moving trunk with a stabilized head is free; a head shaking with fixed neck joints is not.
    robot.data.body_link_ang_vel_w[0, 0, :2] = 5
    robot.data.body_link_ang_vel_w[1, 1] = torch.tensor([3., 4., 20.])
    robot.data.body_link_ang_vel_w[2, 1, 2] = 20
    robot.data.body_link_ang_vel_w[3:, 1, :2] = 1
    robot.data.root_link_pos_w[3, 2] = 0.08
    robot.data.projected_gravity_b[4] = torch.tensor([1., 0., 0.])
    # The same clearance above raised terrain must yield the same upright gate.
    scene.terrain.env_origins[5, 2] = 0.3
    robot.data.root_link_pos_w[5, 2] += 0.3
    torch.testing.assert_close(head_angular_velocity_penalty(recipe, **params), torch.zeros(count))
    recipe._head_stability_age.fill_(1)
    torch.testing.assert_close(
        head_angular_velocity_penalty(recipe, **params), torch.tensor([0., 25., 0., 0., 0., 2.])
    )
    command[1, 0] = 0.2
    assert head_angular_velocity_penalty(recipe, **params)[1] == 0
    recipe._head_stability_age[1] = 0.4
    assert head_angular_velocity_penalty(recipe, **params)[1].item() == pytest.approx(12.5)
    recipe._head_stability_age[1] = 0.6
    assert head_angular_velocity_penalty(recipe, **params)[1].item() == pytest.approx(25)


@pytest.mark.parametrize("task", ["velocity", "velstand", "velocity_backlash", "velstand_backlash"])
def test_head_stability_body_binding_dispatch_and_reset(task):
    from examples.microduck.env import MicroDuckEnv

    env = MicroDuckEnv(solver="mujoco", num_envs=1, device="cpu", task=task)
    try:
        recipe = env.official
        assert recipe.supported["head_angular_velocity"] == "head_angular_velocity_penalty"
        env.step(torch.zeros(1, 14))
        # The mouth site is attached to the final head link, also on backlash models.
        assert recipe._head_stability_body_id == recipe.sites[0][0]
        recipe._head_stability_age.fill_(1)
        recipe.reset(torch.tensor([0]))
        torch.testing.assert_close(recipe._head_stability_age, torch.tensor([0.]))
    finally:
        env.close()


def test_head_stability_partial_reset_keeps_other_worlds_timer():
    from examples.microduck.recipe import OfficialRecipe

    stub = SimpleNamespace(reset=lambda ids: None)
    recipe = SimpleNamespace(
        observations=stub, sensors=stub, command_progress=stub,
        spec={"commands": {"twist": {"randomize_phase": False}}},
        phase_offset=torch.zeros(3), standing=torch.zeros(3, dtype=torch.bool),
        velocity_commands=None, update_commands=lambda ids: None,
        _head_stability_age=torch.tensor([0.7, 0.8, 0.9]),
    )
    OfficialRecipe.reset(recipe, torch.tensor([1]))
    torch.testing.assert_close(recipe._head_stability_age, torch.tensor([0.7, 0., 0.9]))


def progress_fixture(count):
    term = json.loads(SNAPSHOT.read_text())["tasks"]["velocity"]["rewards"]["command_progress"]
    recipe = SimpleNamespace(
        env=SimpleNamespace(commands=torch.zeros(count, 13), mixed_terrain=True),
        robot=SimpleNamespace(data=SimpleNamespace(
            root_link_lin_vel_b=torch.zeros(count, 3),
            projected_gravity_b=torch.tensor([[0., 0., -1.]]).repeat(count, 1),
        )),
        step_dt=0.02, extras={"log": {}},
    )
    progress = CommandProgress(count, "cpu")
    return recipe, progress, term["params"]


def commanded_foot_fixture(count=1):
    sensor = ContactSensor(count, 2, "cpu")
    data = SimpleNamespace(
        root_link_pos_w=torch.zeros(count, 3),
        root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(count, 1),
        root_link_lin_vel_b=torch.zeros(count, 3),
        site_pos_w=torch.tensor([[[0.0, 0.0, 0.1], [0.0, 0.03, 0.0], [0.0, -0.03, 0.0]]]).repeat(count, 1, 1),
        site_lin_vel_w=torch.zeros(count, 3, 3),
    )
    recipe = SimpleNamespace(
        num_envs=count,
        device=torch.device("cpu"),
        step_dt=0.02,
        scene={
            "robot": SimpleNamespace(data=data),
            "feet_ground_contact": sensor,
            "foot_height_scan": SimpleNamespace(data=SimpleNamespace(heights=torch.full((count, 2), 0.02))),
        },
        command_manager=SimpleNamespace(get_command=lambda name: torch.tensor([[0.2, 0.0, 0.0]]).repeat(count, 1)),
        extras={"log": {}},
    )
    return recipe, sensor, data


def test_commanded_foot_placement_uses_support_foot_and_touchdown_gate():
    recipe, sensor, data = commanded_foot_fixture()
    sensor.update(torch.tensor([[0.0, 1.0]]), recipe.step_dt)
    sensor.update(torch.tensor([[0.0, 1.0]]), recipe.step_dt)
    data.site_pos_w[0, 1] = torch.tensor([0.027, 0.03, 0.0])
    params = {
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.03,
        "placement_std": 0.025,
        "step_time_s": 0.18,
        "velocity_error_gain": 0.25,
        "max_step_m": 0.08,
    }
    sensor.update(torch.tensor([[1.0, 1.0]]), recipe.step_dt)
    reward = commanded_foot_placement_reward(recipe, **params)
    assert reward.item() > 0.99
    sensor.update(torch.tensor([[0.0, 1.0]]), recipe.step_dt)
    data.site_pos_w[0, 1, 0] += 0.05
    sensor.update(torch.tensor([[1.0, 1.0]]), recipe.step_dt)
    assert 0.0 < commanded_foot_placement_reward(recipe, **params).item() < 0.99
    sensor.update(torch.tensor([[0.0, 1.0]]), recipe.step_dt)
    assert commanded_foot_placement_reward(recipe, **params).item() == 0.0


@pytest.mark.parametrize("count", [1, 2, 5, 256])
@pytest.mark.parametrize("landing_foot", [0, 1])
def test_commanded_foot_placement_keeps_yaw_independent_per_environment(count, landing_foot):
    recipe, sensor, _ = commanded_foot_fixture(count)
    command = torch.zeros(count, 3)
    command[:, 0] = 0.2
    command[:, 2] = torch.linspace(-1.0, 1.0, count)
    recipe.command_manager.get_command = lambda name: command
    params = {
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.03,
        "placement_std": 0.025,
        "step_time_s": 0.18,
        "velocity_error_gain": 0.25,
        "max_step_m": 0.08,
    }
    contact = torch.ones(count, 2)
    contact[:, landing_foot] = 0.0
    sensor.update(contact, recipe.step_dt)
    sensor.update(contact, recipe.step_dt)
    sensor.update(torch.ones_like(contact), recipe.step_dt)

    # Stationary symmetric feet: translation asks for 2.7 cm; yaw shifts the
    # target forward for a left landing and backward for a right landing.
    side = 1.0 if landing_foot == 0 else -1.0
    expected = torch.exp(-0.5 * ((0.027 + side * 0.0027 * command[:, 2]) / 0.025).square())
    actual = commanded_foot_placement_reward(recipe, **params)
    assert actual.shape == (count,)
    torch.testing.assert_close(actual, expected)

    # The user's zero-command replay still evaluates this term before gating it.
    command.zero_()
    torch.testing.assert_close(commanded_foot_placement_reward(recipe, **params), torch.zeros(count))


def test_commanded_clearance_and_swing_height_follow_speed_command():
    recipe, sensor, data = commanded_foot_fixture()
    data.site_lin_vel_w[0, 1:, 0] = 0.2
    sensor.update(torch.tensor([[0.0, 0.0]]), recipe.step_dt)
    recipe.scene["foot_height_scan"].data.heights[:] = 0.027
    clearance_params = {
        "target_height": 0.02,
        "height_speed_gain": 0.035,
        "max_height": 0.045,
        "height_sensor_name": "foot_height_scan",
        "command_name": "twist",
        "command_threshold": 0.03,
    }
    assert commanded_foot_clearance(recipe, **clearance_params).item() > 0.0
    recipe.scene["foot_height_scan"].data.heights[:] = 0.027
    swing = CommandedFootSwingHeight(recipe)
    swing_params = {
        "sensor_name": "feet_ground_contact",
        "height_sensor_name": "foot_height_scan",
        "target_height": 0.02,
        "height_speed_gain": 0.035,
        "max_height": 0.045,
        "height_std": 0.015,
        "command_name": "twist",
        "command_threshold": 0.03,
    }
    swing(recipe, **swing_params)
    sensor.update(torch.tensor([[1.0, 1.0]]), recipe.step_dt)
    assert swing(recipe, **swing_params).item() > 1.99


def test_progress_prefers_tracking_to_stalling_to_reversal_in_both_signed_axes():
    recipe, progress, params = progress_fixture(16)
    directions = torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    recipe.env.commands[:, :2] = directions.repeat_interleave(4, 0) * 0.1
    # Correct, stationary, reverse, and overspeed; the existing tracker handles overspeed.
    recipe.robot.data.root_link_lin_vel_b[:, :2] = (
        directions[:, None] * torch.tensor([0.1, 0., -0.1, 0.2])[None, :, None]
    ).reshape(16, 2)
    assert not progress(recipe, **params).any()
    progress.age[:] = 1
    cost = progress(recipe, **params).reshape(4, 4)
    torch.testing.assert_close(cost, torch.tensor([[0., 0.64, 3.24, 0.]]).repeat(4, 1))


def test_progress_grace_ramp_idle_yaw_recovery_partial_reset_and_flat_task():
    recipe, progress, params = progress_fixture(5)
    recipe.env.commands[:, 0] = 0.1
    recipe.env.commands[1, :3] = torch.tensor([0., 0., 0.5])
    recipe.env.commands[2, :3] = 0
    recipe.robot.data.projected_gravity_b[3] = torch.tensor([1., 0., 0.])
    assert not progress(recipe, **params).any()
    progress.age[:] = params["grace_seconds"]
    assert not progress(recipe, **params).any()
    progress.age[:] = params["grace_seconds"] + params["ramp_seconds"] / 2
    torch.testing.assert_close(progress(recipe, **params), torch.tensor([0.32, 0., 0., 0., 0.32]))
    progress.age[:] = 1
    recipe.env.commands[0, 0] = -0.1
    assert progress(recipe, **params)[0] == 0  # Direction changes get a fresh grace period.
    progress.reset(torch.tensor([4]))
    actual = progress(recipe, **params)
    assert actual[4] == 0 and progress.age[1] > 1
    recipe.env.mixed_terrain = False
    assert not progress(recipe, **params).any()


def slip_fixture():
    data = SimpleNamespace(
        found=torch.tensor([[2., 1.], [2., 1.], [0., 0.], [2., 1.]]),
        soft_found=torch.tensor([[2., 1.], [1., 0.], [0., 0.], [2., 1.]]),
        soft_slip_squared_sum=torch.tensor([[0., 0.], [0.04, 0.], [0., 0.], [0.08, 0.04]]),
    )
    command = torch.tensor([[0.1, 0., 0.], [0.1, 0., 0.], [0.1, 0., 0.], [0., 0., 0.]])
    recipe = SimpleNamespace(
        env=SimpleNamespace(mixed_terrain=True),
        scene={"feet": SimpleNamespace(data=data),
               "robot": SimpleNamespace(data=SimpleNamespace(site_lin_vel_w=torch.full((4, 3, 3), 0.2)))},
        command_manager=SimpleNamespace(get_command=lambda name: command), extras={"log": {}},
    )
    params = {"sensor_name": "feet", "command_name": "twist", "command_threshold": 0.01,
              "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))}
    return recipe, params


def test_relative_slip_co_motion_mixed_edge_airborne_idle_and_flat_equivalence():
    recipe, params = slip_fixture()
    # World movement while stuck to the pad is free; a pad/ground edge blends each contact's squared speed.
    torch.testing.assert_close(feet_surface_slip(recipe, **params), torch.tensor([0., 0.14, 0., 0.]))
    assert recipe.extras["log"]["Metrics/world_slip_velocity_mean"] > 0
    recipe.env.mixed_terrain = False
    expected = feet_slip(recipe, **params)
    torch.testing.assert_close(feet_surface_slip(recipe, **params), expected, atol=0, rtol=0)
    recipe.env.mixed_terrain = True
    recipe.scene["feet"].data.soft_found.zero_()
    recipe.scene["feet"].data.soft_slip_squared_sum.zero_()
    torch.testing.assert_close(feet_surface_slip(recipe, **params), expected, atol=0, rtol=0)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"))])
def test_soft_slip_contact_points_barycentrics_tangent_and_com_rotation(device):
    def array(values, dtype):
        return wp.array(values, dtype=dtype, device=device)

    # Two contacts with equal/opposite sliding must not average to zero. World 1
    # co-moves with a face whose barycentric speed is 0.3. World 2 tests COM offset
    # and normal motion: angular velocity creates 0.2 tangential speed at the foot.
    feet, soft, squares = [wp.zeros((3, 2), device=device) for _ in range(3)]
    robot, head = [wp.zeros((3, 1), device=device) for _ in range(2)]
    wp.launch(_soft_contact_flags, dim=4, inputs=[
        array([4], int), array([[0, -1, -1], [1, -1, -1], [2, 3, 4], [5, -1, -1]], wp.vec3i),
        array([[1, 0, 0], [1, 0, 0], [0.2, 0.3, 0.5], [1, 0, 0]], wp.vec3),
        array([0, 0, 1, 2], int), array([[0, 0, 0.06]] * 4, wp.vec3),
        array([[0, 0, -1]] * 4, wp.vec3), array([0, 1, 2], int), array([0, 1, 2], int),
        array([0, 0, 0], float), array([1, 1, 1], int), array([0, 0, 1, 1, 1, 2], int),
        array([0.0015] * 6, float), array([1] * 6, int), array([[0, 0, 0.059]] * 6, wp.vec3),
        array([wp.transform_identity()] * 3, wp.transform),
        array([[0.2, 0, 0], [-0.2, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0]], wp.vec3),
        array([[0, 0, 0, 0, 0, 0], [0.3, 0, 0, 0, 0, 0], [0, 0, 2, 0, 0, 1]], wp.spatial_vector),
        array([[0, 0, 0], [0, 0, 0], [0.2, 0, 0]], wp.vec3), feet, robot, head, soft, squares,
    ], device=device)
    np.testing.assert_allclose(squares.numpy(), [[0.08, 0], [0, 0], [0.04, 0]], atol=1e-7)
    np.testing.assert_array_equal(soft.numpy(), [[2, 0], [1, 0], [1, 0]])
    np.testing.assert_array_equal(feet.numpy(), soft.numpy())


def test_local_reward_export_and_soft_sensor_partial_reset():
    snapshot = json.loads(SNAPSHOT.read_text())
    rewards = snapshot["tasks"]["velocity"]["rewards"]
    for name in ("foot_slip", "command_progress", "head_angular_velocity"):
        assert rewards[name]["local"] and rewards[name]["weight"] < 0
    for name in ("foot_clearance", "foot_swing_height"):
        assert rewards[name]["local"] and rewards[name]["weight"] > 0
    assert rewards["commanded_foot_placement"]["local"] and rewards["commanded_foot_placement"]["weight"] == 0
    assert rewards["foot_slip"]["weight"] == -0.1
    for task_name in ("velocity", "velstand", "velocity_rough", "velstand_rough"):
        task_rewards = snapshot["tasks"][task_name]["rewards"]
        assert task_rewards["commanded_foot_placement"]["local"]
        assert task_rewards["foot_clearance"]["func"] == "commanded_foot_clearance"
        assert task_rewards["foot_swing_height"]["func"] == "commanded_foot_swing_height"
    assert all("command_progress" not in spec["rewards"] for task, spec in snapshot["tasks"].items() if task != "velocity")
    source = ast.parse(SNAPSHOT.parents[1].joinpath("export_recipe.py").read_text())
    func = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "preserve_local_terms")
    namespace = {}
    # Extract the pure helper without importing the upstream-only export dependencies.
    exec(compile(ast.Module(body=[func], type_ignores=[]), "export_recipe.py", "exec"), namespace)  # noqa: S102
    exported = {"tasks": {"velocity": {"rewards": {"foot_slip": {"func": "upstream", "weight": -9}}}}}
    namespace["preserve_local_terms"](exported, deepcopy(snapshot))
    for name in ("foot_slip", "command_progress", "head_angular_velocity"):
        assert exported["tasks"]["velocity"]["rewards"][name] == rewards[name]
    selected = {name: spec for name, spec in snapshot["tasks"].items() if name.split("_")[0] in ("velocity", "velstand")}
    assert len(selected) == 8
    for task, spec in selected.items():
        expected = dict(rewards["head_angular_velocity"], weight=-0.025 if task in ("velocity", "velstand") else -0.05)
        assert spec["rewards"]["head_angular_velocity"] == expected
    sensor = ContactSensor(2, 2, "cpu")
    sensor.data.soft_found = torch.ones(2, 2)
    sensor.data.soft_slip_squared_sum = torch.ones(2, 2)
    sensor.reset(torch.tensor([0]))
    assert not sensor.data.soft_found[0].any() and not sensor.data.soft_slip_squared_sum[0].any()
    assert sensor.data.soft_found[1].all() and sensor.data.soft_slip_squared_sum[1].all()


def cadence_fixture(count=1):
    from examples.microduck.reward_functions import CommandedGaitCadence

    recipe, sensor, data = commanded_foot_fixture(count)
    recipe.step_dt = 0.01
    recipe.common_step_counter = 0
    command = torch.zeros(count, 3)
    command[:, 0] = 0.2
    recipe.command_manager.get_command = lambda _: command
    scene = type("Scene", (dict,), {})(recipe.scene)
    scene.terrain = SimpleNamespace(env_origins=torch.zeros(count, 3))
    recipe.scene = scene
    data.root_link_pos_w[:, 2] = 0.12
    data.projected_gravity_b = torch.tensor([[0., 0., -1.]]).repeat(count, 1)
    data.root_link_lin_vel_b[:, 0] = 0.2
    data.root_link_ang_vel_b = torch.zeros(count, 3)
    params = deepcopy(json.loads(SNAPSHOT.read_text())["tasks"]["velocity"]["rewards"]["gait_cadence"]["params"])
    recipe._gait_cadence = CommandedGaitCadence(recipe)
    return recipe, sensor, data, command, params


def cadence_tick(recipe, sensor, params, contact):
    from examples.microduck.reward_functions import commanded_gait_cadence, commanded_gait_overdue

    sensor.update(torch.as_tensor(contact, dtype=torch.float32).reshape(recipe.num_envs, 2), recipe.step_dt)
    recipe.common_step_counter += 1
    reward = commanded_gait_cadence(recipe, **params)
    penalty = commanded_gait_overdue(recipe, **params)
    return reward.clone(), penalty.clone()


def cadence_walk(recipe, sensor, params, interval, seconds=3.0, same_foot=False):
    values = []
    for step in range(round(seconds / recipe.step_dt)):
        period_steps = round(interval / recipe.step_dt)
        foot = 0 if same_foot else (step // period_steps) % 2
        contact = torch.ones(recipe.num_envs, 2)
        # Leave a stable double-support interval after each landing.
        if step % period_steps < round(0.6 * period_steps):
            contact[:, foot] = 0
        values.append(cadence_tick(recipe, sensor, params, contact)[0])
    return torch.stack(values)[-round(0.5 / recipe.step_dt):].mean(0)


def test_cadence_faster_commands_prefer_faster_alternation_and_do_not_pay_for_extra_events():
    from examples.microduck.reward_functions import commanded_step_frequency

    scores = {}
    for speed in (0.2, 0.4):
        for interval in (0.2, 0.25, 0.4, 0.1):
            recipe, sensor, data, command, params = cadence_fixture()
            command[:, 0] = speed
            data.root_link_lin_vel_b[:, 0] = speed
            scores[speed, interval] = cadence_walk(recipe, sensor, params, interval).item()
    assert scores[0.2, 0.25] > scores[0.2, 0.2] > scores[0.2, 0.1]
    assert scores[0.4, 0.2] > scores[0.4, 0.25] > scores[0.4, 0.4]
    assert scores[0.2, 0.25] == pytest.approx(scores[0.4, 0.2], abs=0.01)
    commands = torch.tensor([[0.2, 0., 0.], [-0.2, 0., 0.], [0., 0.2, 0.], [0., 0., 0.2], [0., 0., 0.]])
    frequency, active = commanded_step_frequency(commands, params["cadence"])
    torch.testing.assert_close(frequency, torch.tensor([4., 4., 4., 3.2, 3.]))
    assert active.tolist() == [True, True, True, True, False]


def test_cadence_accepts_support_transfer_without_double_support():
    recipe, sensor, _, _, params = cadence_fixture()
    for step in range(300):
        # The old support foot lifts as the new foot lands, before its 30 ms confirmation.
        foot = (step // 25) % 2
        contact = [[float(foot == 1), float(foot == 0)]]
        reward, _ = cadence_tick(recipe, sensor, params, contact)
    assert reward.item() > 0.99
    assert recipe._gait_cadence.interval.item() == pytest.approx(0.25)


def test_cadence_does_not_reuse_stale_support_for_unsupported_landing():
    recipe, sensor, _, _, params = cadence_fixture()
    for _ in range(10):
        cadence_tick(recipe, sensor, params, [[1., 1.]])
    for _ in range(30):
        cadence_tick(recipe, sensor, params, [[0., 0.]])
    for _ in range(10):
        cadence_tick(recipe, sensor, params, [[1., 0.]])
    assert recipe._gait_cadence.last_foot.item() == -1


@pytest.mark.parametrize("mode", ["same_foot", "hop", "low_clearance", "chatter", "no_progress", "wrong_direction"])
def test_cadence_rejects_invalid_steps_and_stationary_or_reverse_tapping(mode):
    recipe, sensor, data, command, params = cadence_fixture()
    if mode == "low_clearance":
        recipe.scene["foot_height_scan"].data.heights.zero_()
    if mode == "no_progress":
        data.root_link_lin_vel_b.zero_()
    if mode == "wrong_direction":
        data.root_link_lin_vel_b[:, 0] = -0.2
    if mode in ("hop", "chatter"):
        for step in range(200):
            contact = [[float(step % 25 >= 15)] * 2] if mode == "hop" else [[float(step % 2), 1.]]
            reward, _ = cadence_tick(recipe, sensor, params, contact)
            assert not reward.any()
    else:
        assert cadence_walk(recipe, sensor, params, 0.25, same_foot=mode == "same_foot").item() == 0


def test_cadence_overdue_expires_reward_and_gates_idle_recovery_and_command_changes():
    recipe, sensor, data, command, params = cadence_fixture(3)
    assert (cadence_walk(recipe, sensor, params, 0.25) > 0.99).all()
    # Both terms share a single update regardless of evaluation order.
    from examples.microduck.reward_functions import commanded_gait_cadence, commanded_gait_overdue

    elapsed = recipe._gait_cadence.elapsed.clone()
    commanded_gait_overdue(recipe, **params)
    commanded_gait_cadence(recipe, **params)
    torch.testing.assert_close(recipe._gait_cadence.elapsed, elapsed)
    for _ in range(100):
        reward, penalty = cadence_tick(recipe, sensor, params, [[1., 1.]] * 3)
    assert not reward.any() and (penalty == 1).all()
    command[0] = 0
    data.projected_gravity_b[1] = torch.tensor([1., 0., 0.])
    command[2, 0] = 0.4
    reward, penalty = cadence_tick(recipe, sensor, params, [[1., 1.]] * 3)
    assert not reward.any() and not penalty.any()
    assert (recipe._gait_cadence.last_foot == -1).all()


def test_cadence_turning_and_partial_reset_are_independent_between_worlds():
    recipe, sensor, data, command, params = cadence_fixture(2)
    command[1] = torch.tensor([0., 0., 1.])  # Both worlds request four steps/s.
    data.root_link_lin_vel_b[1] = 0
    data.root_link_ang_vel_b[1, 2] = 1
    assert (cadence_walk(recipe, sensor, params, 0.25) > 0.99).all()
    state = recipe._gait_cadence
    before = {name: getattr(state, name)[1].clone() for name in ("interval", "elapsed", "age", "last_foot", "armed")}
    state.reset(torch.tensor([0]))
    assert state.interval[0] == 0 and state.last_foot[0] == -1 and state.reward[0] == 0
    for name, value in before.items():
        torch.testing.assert_close(getattr(state, name)[1], value)


def test_cadence_air_window_and_placement_share_speed_dependent_period():
    from examples.microduck.reward_functions import commanded_feet_air_time, commanded_step_frequency

    recipe, sensor, data, command, params = cadence_fixture(5)
    command[:, 0] = torch.tensor([0.2, 0.4, -0.4, 0., 0.])
    command[3, 2] = 0.2
    frequency, active = commanded_step_frequency(command, params["cadence"])
    sensor.data.current_air_time[:] = 0.11
    air = commanded_feet_air_time(recipe, "feet_ground_contact", "twist", params["cadence"])
    torch.testing.assert_close(air, torch.tensor([0., 2., 2., 0., 0.]))
    placement = deepcopy(json.loads(SNAPSHOT.read_text())["tasks"]["velocity"]["rewards"]["commanded_foot_placement"])
    # Match the intended left-foot landing exactly, including the yaw correction, for different batch commands.
    time_scale = (params["cadence"]["placement_time_fraction"] / frequency).clamp(0.12, 0.24)
    data.root_link_lin_vel_b[:, :2] = command[:, :2]
    data.site_pos_w[:, 1, 0] = time_scale * (0.5 * command[:, 0] + 0.015 * command[:, 2])
    sensor.update(torch.tensor([[0., 1.]]).repeat(5, 1), 0.04)
    sensor.update(torch.ones(5, 2), 0.01)
    # Decode the JSON entity selector just as the real recipe does.
    from examples.microduck.recipe import OfficialRecipe
    decoder = OfficialRecipe.__new__(OfficialRecipe)
    decoded = decoder.decode(placement["params"])
    actual = commanded_foot_placement_reward(recipe, **decoded)
    torch.testing.assert_close(actual, active.float())


def test_cadence_config_scope_and_real_recipe_reset_and_dispatch():
    from examples.microduck.env import MicroDuckEnv

    tasks = json.loads(SNAPSHOT.read_text())["tasks"]
    selected = [name for name in tasks if name.split("_")[0] in ("velocity", "velstand")]
    assert len(selected) == 8
    for name, spec in tasks.items():
        rewards = spec["rewards"]
        assert ("foot_swing_progress" in rewards) == (name in ("velocity", "velstand"))
        if name in ("velocity", "velstand"):
            progress = rewards["foot_swing_progress"]
            assert progress["local"] and progress["weight"] == 0
            assert progress["params"]["cadence"] == rewards["gait_cadence"]["params"]["cadence"]
        if name not in selected:
            assert "gait_cadence" not in rewards and "gait_overdue" not in rewards
            continue
        cadence = rewards["gait_cadence"]["params"]["cadence"]
        for term in ("gait_cadence", "gait_overdue", "air_time", "foot_clearance", "foot_swing_height",
                     "commanded_foot_placement"):
            assert rewards[term]["local"] and rewards[term]["params"]["cadence"] == cadence
        if name == "velocity":
            assert rewards["gait_cadence"]["weight"] == rewards["gait_overdue"]["weight"] == 0
        else:
            assert rewards["gait_cadence"]["weight"] > 0 > rewards["gait_overdue"]["weight"]
    for task in ("velocity", "velstand"):
        env = MicroDuckEnv(task=task, solver="mujoco", device="cpu", num_envs=1)
        try:
            _, reward, _, extras = env.step(torch.zeros_like(env.actions))
            assert torch.isfinite(reward).all()
            if task == "velstand":
                assert extras["log"]["Task/gait_cadence"] >= 0
                assert extras["log"]["Task/gait_overdue"] <= 0
            else:
                assert "Task/gait_cadence" not in extras["log"]
                assert "Task/gait_overdue" not in extras["log"]
            if task == "velocity":
                assert "Task/foot_swing_progress" not in extras["log"]
                assert "Metrics/foot_swing_score_left" not in extras["log"]
                assert not hasattr(env.official, "_foot_swing_progress")
            else:
                state = env.official._gait_cadence
                state.interval.fill_(0.25)
            env._reset(torch.tensor([0]))
            if task == "velstand":
                assert not state.interval.any() and (state.last_foot == -1).all()
        finally:
            env.close()


def swing_progress_fixture(count=1):
    recipe, sensor, data, command, cadence_params = cadence_fixture(count)
    params = dict(cadence_params, advance_time_fraction=0.6, min_advance=0.008, max_advance=0.05,
                  freshness_cycles=1.5, asset_cfg=SimpleNamespace(name="robot", site_ids=[1, 2]))
    return recipe, sensor, data, command, params


def swing_progress_walk(recipe, sensor, data, command, params, mode="walk", seconds=3.0):
    from examples.microduck.reward_functions import commanded_foot_swing_progress

    direction = command[:, :2] / command[:, :2].norm(dim=-1, keepdim=True).clamp_min(1.e-6)
    qw, _, _, qz = data.root_link_quat_w.unbind(-1)
    yaw = 2 * torch.atan2(qz, qw)
    world_direction = torch.stack((yaw.cos() * direction[:, 0] - yaw.sin() * direction[:, 1],
                                   yaw.sin() * direction[:, 0] + yaw.cos() * direction[:, 1]), -1)
    for step in range(round(seconds / recipe.step_dt)):
        foot = (step // 25) % 2
        airborne = step % 25 < 15
        contact = torch.ones(recipe.num_envs, 2)
        if airborne and mode != "drag":
            contact[:, foot] = 0
        root_delta = world_direction * 0.002
        data.root_link_pos_w[:, :2] += root_delta
        if mode == "passive":
            data.site_pos_w[:, 1:3, :2] += root_delta[:, None]
        elif airborne:
            distance = -0.006 if mode == "reverse" else 0.006
            if mode not in ("vertical", "drag") and not (mode == "one_foot" and foot == 1):
                data.site_pos_w[:, foot + 1, :2] += world_direction * distance
        sensor.update(contact, recipe.step_dt)
        recipe.common_step_counter += 1
        reward = commanded_foot_swing_progress(recipe, **params)
    return reward


@pytest.mark.parametrize("mode", ["walk", "one_foot", "vertical", "passive", "reverse", "drag"])
def test_swing_progress_requires_each_foot_to_advance_after_qualified_swing(mode):
    recipe, sensor, data, command, params = swing_progress_fixture()
    reward = swing_progress_walk(recipe, sensor, data, command, params, mode)
    assert reward.item() == pytest.approx(1.0 if mode == "walk" else 0.0)
    state = recipe._foot_swing_progress
    if mode == "one_foot":
        assert state.score[0, 0] > 0.99 and state.score[0, 1] == 0
    assert "Metrics/foot_swing_advance_left" in recipe.extras["log"]
    assert "Metrics/foot_swing_score_right" in recipe.extras["log"]


def test_swing_progress_direction_yaw_cache_expiry_and_partial_reset():
    from examples.microduck.reward_functions import commanded_foot_swing_progress

    recipe, sensor, data, command, params = swing_progress_fixture(4)
    command[:] = torch.tensor([[0.2, 0., 0.], [-0.2, 0., 0.], [0., 0.2, 0.], [0., 0., 1.]])
    data.root_link_lin_vel_b[:, :2] = command[:, :2]
    data.root_link_quat_w[2] = torch.tensor([2 ** -0.5, 0., 0., 2 ** -0.5])
    reward = swing_progress_walk(recipe, sensor, data, command, params)
    torch.testing.assert_close(reward, torch.tensor([1., 1., 1., 0.]))
    state = recipe._foot_swing_progress
    age = state.score_age.clone()
    commanded_foot_swing_progress(recipe, **params)
    torch.testing.assert_close(state.score_age, age)
    score = state.score[1:].clone()
    state.reset(torch.tensor([0]))
    assert not state.score[0].any()
    torch.testing.assert_close(state.score[1:], score)
    for _ in range(100):
        sensor.update(torch.ones(4, 2), recipe.step_dt)
        recipe.common_step_counter += 1
        reward = commanded_foot_swing_progress(recipe, **params)
    assert not reward.any()


@pytest.mark.parametrize("mode", ["idle", "tilt", "command_change", "no_progress", "wrong_direction"])
def test_swing_progress_cannot_keep_reward_after_gating_changes(mode):
    from examples.microduck.reward_functions import commanded_foot_swing_progress

    recipe, sensor, data, command, params = swing_progress_fixture()
    assert swing_progress_walk(recipe, sensor, data, command, params).item() > 0.99
    if mode == "idle":
        command.zero_()
    elif mode == "tilt":
        data.projected_gravity_b[:] = torch.tensor([1., 0., 0.])
    elif mode == "command_change":
        command[:, 0] = 0.4
    else:
        data.root_link_lin_vel_b[:, 0] = 0 if mode == "no_progress" else -0.2
    recipe.common_step_counter += 1
    assert commanded_foot_swing_progress(recipe, **params).item() == 0


def test_swing_progress_freezes_displacement_before_contact_settling():
    from examples.microduck.reward_functions import commanded_foot_swing_progress

    recipe, sensor, data, _, params = swing_progress_fixture()

    def tick(contact):
        sensor.update(torch.tensor([contact]), recipe.step_dt)
        recipe.common_step_counter += 1
        commanded_foot_swing_progress(recipe, **params)

    for _ in range(10):
        tick([1., 1.])
    for _ in range(12):
        data.site_pos_w[:, 1, 0] += 0.001
        tick([0., 1.])
    tick([1., 1.])
    for _ in range(5):
        # Large stance slip during contact confirmation must not become swing credit.
        data.site_pos_w[:, 1, 0] += 0.1
        tick([1., 1.])
    state = recipe._foot_swing_progress
    assert state.last_advance[0, 0] == pytest.approx(0.011, abs=1.e-6)
    assert state.score[0, 0] == pytest.approx(0.011 / 0.03, abs=1.e-6)


def test_swing_progress_preserves_lift_position_across_brief_contact_chatter():
    from examples.microduck.reward_functions import commanded_foot_swing_progress

    recipe, sensor, data, _, params = swing_progress_fixture()
    for step in range(45):
        contact = [[1., 1.]]
        if 10 <= step < 35:
            data.site_pos_w[:, 1, 0] += 0.002
            # The one-frame contact does not meet the 30 ms stable-contact requirement.
            if step != 20:
                contact = [[0., 1.]]
        sensor.update(torch.tensor(contact), recipe.step_dt)
        recipe.common_step_counter += 1
        commanded_foot_swing_progress(recipe, **params)
    assert recipe._foot_swing_progress.last_advance[0, 0] == pytest.approx(0.048, abs=1.e-6)
    assert recipe._foot_swing_progress.score[0, 0] == 1


def test_swing_progress_same_foot_repetitions_expire_the_other_foot_sample():
    from examples.microduck.reward_functions import commanded_foot_swing_progress

    recipe, sensor, data, command, params = swing_progress_fixture()
    assert swing_progress_walk(recipe, sensor, data, command, params).item() > 0.99
    for step in range(150):
        contact = [[0., 1.]] if step % 25 < 15 else [[1., 1.]]
        data.site_pos_w[:, 1, 0] += 0.006
        sensor.update(torch.tensor(contact), recipe.step_dt)
        recipe.common_step_counter += 1
        reward = commanded_foot_swing_progress(recipe, **params)
    assert reward.item() == 0
    assert recipe._foot_swing_progress.score_age[0, 1] > 0.75
