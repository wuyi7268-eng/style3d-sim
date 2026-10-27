"""Formula provenance and recipe boundaries that can be checked without a GPU or credentials."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import warp as wp

from examples.microduck import official_reward_functions as formulas
from examples.microduck import official_velocity_rewards as velocity_formulas
from examples.microduck.common import setup
from examples.microduck.official_adapter import SceneEntityCfg, link_origin_velocity
from examples.microduck.official_recipe import OfficialRecipe, recipe_fingerprint
from examples.microduck.official_sensors import (
    ContactSensor,
    TerrainHeightSensor,
    _contact_flags,
    whole_body_angular_momentum,
)
from examples.microduck.policy_spec import build_robot, filter_authored_collider

ROOT = Path(__file__).parents[1]


def test_exported_function_bodies_match_latest_upstream_definitions():
    source = ast.parse((ROOT / "microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py").read_text())
    upstream = {n.name: n for n in source.body if isinstance(n, ast.FunctionDef)}
    copied = ast.parse((ROOT / "official_reward_functions.py").read_text())
    functions = [n for n in copied.body if isinstance(n, ast.FunctionDef)]
    assert len(functions) >= 35
    for node in functions:
        expected = ast.parse(ast.unparse(upstream[node.name])).body[0]
        actual = ast.parse(ast.unparse(node)).body[0]
        for tree in (expected, actual):
            for child in ast.walk(tree):
                if isinstance(child, ast.ImportFrom) and child.module == "mjlab.sensor":
                    child.module = "official_sensors"
                    child.level = 1
        assert ast.dump(actual, include_attributes=False) == ast.dump(expected, include_attributes=False)


def test_default_feet_selection_and_com_velocity_frame():
    cfg = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))
    assert cfg.site_ids == [1, 2]
    velocity = link_origin_velocity(
        torch.tensor([[0.0, 1.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 1.0]]),
        torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        torch.tensor([1.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(velocity, torch.zeros(1, 3))


def test_ball_target_and_overshoot():
    env = SimpleNamespace(
        num_envs=4,
        device="cpu",
        scene={
            "ball": SimpleNamespace(
                data=SimpleNamespace(
                    root_link_lin_vel_w=torch.tensor(
                        [
                            [-1.0, 0.0, 0.0],
                            [0.5, 0.0, 0.0],
                            [1.0, 0.0, 0.0],
                            [2.0, 0.0, 0.0],
                        ]
                    )
                )
            )
        },
    )
    torch.testing.assert_close(formulas.ball_forward_velocity(env, max_speed=1.0), torch.tensor([0.0, 0.5, 1.0, 1.0]))
    torch.testing.assert_close(formulas.ball_speed_overshoot_penalty(env), torch.tensor([0.0, 0.0, 0.0, 1.0]))


def test_partial_reset_potential_does_not_leak_between_episodes():
    data = SimpleNamespace(root_link_pos_w=torch.tensor([[0.0, 0.0, 0.1], [0.0, 0.0, 0.1]]))
    scene = type("Scene", (dict,), {})({"robot": SimpleNamespace(data=data)})
    scene.terrain = SimpleNamespace(env_origins=torch.zeros(2, 3))
    env = SimpleNamespace(scene=scene, episode_length_buf=torch.tensor([2, 2]))
    assert not formulas.height_progress(env).any()
    data.root_link_pos_w[:, 2] += 0.01
    env.episode_length_buf[:] = torch.tensor([1, 3])
    torch.testing.assert_close(formulas.height_progress(env), torch.tensor([0.0, 0.01]))


def test_curriculum_strict_stage_boundary():
    env = SimpleNamespace(common_step_counter=10)
    stages = [{"step": 0, "value": 1}, {"step": 10, "value": 2}]
    assert OfficialRecipe.stage(env, stages)["value"] == 1
    env.common_step_counter = 11
    assert OfficialRecipe.stage(env, stages)["value"] == 2


def test_recipe_checkpoint_guard(tmp_path):
    checkpoint = tmp_path / "model_0.pt"
    checkpoint.touch()
    metadata = {
        "observation_contract": "official_asymmetric_pd_v2",
        "solver": "synreal",
        "recipe": "official_adapted_v1",
        "task": "groundpick",
        "ppo": {},
        "official_alignment": {"recipe_fingerprint": recipe_fingerprint()},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(metadata))
    args, _, _ = setup("train", ["--solver", "synreal", "--resume", str(checkpoint)])
    assert args.recipe == "official_adapted_v1" and args.task == "groundpick"
    with pytest.raises(SystemExit):
        setup("train", ["--solver", "synreal", "--resume", str(checkpoint), "--recipe", "legacy"])
    metadata["official_alignment"]["recipe_fingerprint"] = "different"
    path.write_text(json.dumps(metadata))
    with pytest.raises(SystemExit):
        setup("train", ["--solver", "synreal", "--resume", str(checkpoint)])


def test_groundpick_phase_boundaries():
    phase = torch.tensor([0.0, 0.375, 0.425, 0.6125, 0.8, 0.99])
    torch.testing.assert_close(
        formulas.phase_pose_blend(phase, 0.375, 0.425, 0.8), torch.tensor([0.0, 1.0, 1.0, 0.5, 0.0, 0.0])
    )
    torch.testing.assert_close(formulas.phase_rise_gate(phase, 0.425, 0.8), torch.tensor([0.0, 0.0, 0.0, 0.5, 1.0, 1.0]))


def test_velocity_reward_source_provenance():
    path = ROOT / "microduck_rl-develop/.venv/lib/python3.12/site-packages/mjlab/tasks/velocity/mdp/rewards.py"
    source = ast.parse(path.read_text())
    upstream = {n.name: n for n in source.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    generated = ast.parse((ROOT / "official_velocity_rewards.py").read_text())
    nodes = [n for n in generated.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    assert len(nodes) == 6
    for node in nodes:
        # Ruff reindents multiline docstrings; executable AST must remain identical.
        for tree in (node, upstream[node.name]):
            for child in ast.walk(tree):
                if isinstance(child, (ast.FunctionDef, ast.ClassDef)) and ast.get_docstring(child) is not None:
                    child.body.pop(0)
        assert ast.dump(node, include_attributes=False) == ast.dump(upstream[node.name], include_attributes=False)


def test_contact_air_time_touchdown_and_partial_reset():
    sensor = ContactSensor(2, 2, "cpu")
    for _ in range(30):
        sensor.update(torch.tensor([[0.0, 1.0], [1.0, 0.0]]), 0.005)
    torch.testing.assert_close(sensor.data.current_air_time, torch.tensor([[0.15, 0.0], [0.0, 0.15]]))
    sensor.update(torch.ones(2, 2), 0.005)
    torch.testing.assert_close(sensor.data.last_air_time, torch.tensor([[0.155, 0.0], [0.0, 0.155]]))
    assert sensor.compute_first_contact(0.02).tolist() == [[True, False], [False, True]]
    before = sensor.data.current_contact_time[1].clone()
    sensor.reset(torch.tensor([0]))
    assert not sensor.data.current_contact_time[0].any()
    torch.testing.assert_close(sensor.data.current_contact_time[1], before)


def test_angular_momentum_spin_orbit_and_translation_invariance():
    pose = torch.tensor([[[-1.0, 0, 0, 0, 0, 0, 1], [1.0, 0, 0, 0, 0, 0, 1]]])
    velocity = torch.tensor([[[0.0, -2, 0, 0, 0, 3], [0.0, 2, 0, 0, 0, 3]]])
    mass = torch.tensor([[1.0, 1.0]])
    inertia = torch.eye(3).expand(1, 2, 3, 3)
    local_com = torch.zeros(1, 2, 3)
    expected = torch.tensor([[0.0, 0, 10]])
    torch.testing.assert_close(whole_body_angular_momentum(pose, velocity, local_com, mass, inertia), expected)
    pose[..., :3] += torch.tensor([5.0, 9.0, -2.0])
    velocity[..., :3] += torch.tensor([3.0, 4.0, 7.0])
    torch.testing.assert_close(whole_body_angular_momentum(pose, velocity, local_com, mass, inertia), expected)


def test_swing_height_landing_cost_and_command_gate():
    sensor = ContactSensor(1, 2, "cpu")
    height = TerrainHeightSensor()
    height.data.heights = torch.tensor([[0.04, 0.02]])
    scene = {"feet": sensor, "height": height}
    env = SimpleNamespace(
        scene=scene,
        num_envs=1,
        device="cpu",
        step_dt=0.02,
        extras={"log": {}},
        command_manager=SimpleNamespace(get_command=lambda _: torch.tensor([[0.2, 0.0, 0.0]])),
    )
    reward = velocity_formulas.feet_swing_height(SimpleNamespace(params={"height_sensor_name": "height"}), env)
    params = {
        "sensor_name": "feet",
        "height_sensor_name": "height",
        "target_height": 0.02,
        "command_name": "twist",
        "command_threshold": 0.01,
    }
    sensor.update(torch.zeros(1, 2), 0.02)
    torch.testing.assert_close(reward(env, **params), torch.zeros(1))
    sensor.update(torch.ones(1, 2), 0.005)
    torch.testing.assert_close(reward(env, **params), torch.ones(1))
    assert not reward.peak_heights.any()


def test_contact_flags_exclude_gaps_ball_and_other_worlds():
    feet = wp.zeros((2, 2), dtype=float, device="cpu")
    self_contact = wp.zeros((2, 1), dtype=float, device="cpu")
    robot_ground = wp.zeros((2, 1), dtype=float, device="cpu")
    head_ground = wp.zeros((2, 1), dtype=float, device="cpu")

    def array(values, dtype=wp.int32):
        return wp.array(values, dtype=dtype, device="cpu")

    wp.launch(
        _contact_flags,
        dim=6,
        inputs=[
            array([6]),
            array([1, 1, 1, 1, 1, 4]),
            array([0, 0, 2, 3, 4, 0]),
            array([[0, 0, 0]] * 6, wp.vec3),
            array([[0, 0, -0.01], [0, 0, 0.1]] + [[0, 0, -0.01]] * 4, wp.vec3),
            array([[0, 0, 1]] * 6, wp.vec3),
            array([0] * 6, float),
            array([0] * 6, float),
            array([-1, 0, 1, 2, 3]),
            array([-1, 0, 0, 0, 1]),
            array([-1, 1, 2, -2, 1]),
            array([[0, 0, 0, 0, 0, 0, 1]] * 4, wp.transform),
            feet,
            self_contact,
            robot_ground,
            head_ground,
        ],
        device="cpu",
    )
    torch.testing.assert_close(wp.to_torch(feet), torch.tensor([[1.0, 0.0], [1.0, 0.0]]))
    torch.testing.assert_close(wp.to_torch(self_contact), torch.tensor([[1.0], [0.0]]))
    torch.testing.assert_close(wp.to_torch(robot_ground), torch.tensor([[1.0], [1.0]]))
    torch.testing.assert_close(wp.to_torch(head_ground), torch.zeros(2, 1))


@pytest.mark.parametrize("enabled", [False, True])
def test_self_collision_only_meshes_cannot_hit_authored_terrain(enabled):
    robot, _, _ = build_robot(enable_self_collisions=enabled)
    ground = robot.add_ground_plane()
    filter_authored_collider(robot, ground)
    types = robot.custom_attributes["mujoco:contype"]
    affinities = robot.custom_attributes["mujoco:conaffinity"]
    count = 0
    for shape in range(ground):
        ctype = int(types.values.get(shape, types.default))
        affinity = int(affinities.values.get(shape, affinities.default))
        if ctype == affinity == 2:
            count += 1
            assert (shape, ground) in robot.shape_collision_filter_pairs
    assert count > 0


@pytest.mark.parametrize("task", ["velocity", "velstand", "groundpick", "ballkick"])
def test_fixed_friction_keeps_foot_ground_collision(task):
    from examples.microduck.official_randomization import randomize_foot_friction

    robot, _, _ = build_robot(task=task, enable_self_collisions=False)
    builder = type(robot)(up_axis=2)
    builder.replicate(robot, 2)
    ground = builder.add_ground_plane()
    filter_authored_collider(builder, ground)
    feet = [
        i
        for i, label in enumerate(builder.shape_label)
        if label.rsplit("/", 1)[-1] in ("left_foot_collision", "right_foot_collision")
    ]
    assert len(feet) == 4
    assert all((i, ground) not in builder.shape_collision_filter_pairs for i in feet)
    model = builder.finalize(device="cpu")
    before = wp.to_torch(model.shape_material_mu).clone()
    metadata = randomize_foot_friction(model, task, 2, "cpu")
    actual = wp.to_torch(model.shape_material_mu)
    torch.testing.assert_close(actual[feet], torch.ones(4))
    other = torch.ones(len(actual), dtype=torch.bool)
    other[feet] = False
    torch.testing.assert_close(actual[other], before[other])
    assert metadata["enabled"] is False and metadata["per_env"] is False
