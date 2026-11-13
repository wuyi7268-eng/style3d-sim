"""Bridge anchoring, deformed support, selective reset, and walking command contracts."""

import os
from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import patch

import newton
import numpy as np
import pytest
import torch
import warp as wp

from examples.microduck.scenes import SCENE_CONFIG, SceneTerrain, add_scene


@pytest.mark.skipif(os.environ.get("MICRODUCK_GPU_TESTS") != "1", reason="Requires SynReal login and CUDA")
def test_unloaded_bridge_motion_decays_at_checkpoint_timestep():
    import style3d_sim as s3d
    from examples.microduck.env import PhysicsConfig

    # Freeze the light, fine mesh and 5 ms playback solve that exposed growing oscillations.
    config = replace(SCENE_CONFIG, elastic_bridge_length=.8, elastic_bridge_width=.3,
                     elastic_bridge_height=.65, elastic_bridge_sag=.04,
                     elastic_bridge_cells_x=40, elastic_bridge_cells_y=10,
                     elastic_bridge_density=.1, elastic_bridge_stretch=5000.,
                     elastic_bridge_bend=0., elastic_bridge_damping=1.)
    physics = PhysicsConfig(physics_dt=.005, decimation=4, nonlinear_iterations=5, linear_iterations=35,
                            joint_constraint_primal_iterations=1, affine_body_block_preconditioner=False,
                            air_density=1.225)
    builder = s3d.ModelBuilder()
    add_scene(builder, "elastic_bridge", config)
    model = builder.finalize(device="cuda:0")
    model.set_gravity((0, 0, -9.81))
    solver = s3d.NewtonSolver(model)
    physics.configure_synreal(solver)
    collision = newton.CollisionPipeline(model, enable_rigid_soft_full_surface_contact=True,
                                         soft_contact_margin=.01, soft_contact_max=8192)
    contacts = collision.contacts()
    solver.prepare_contacts(contacts)
    state, next_state = model.state(), model.state()
    control = model.control()
    reference = model.particle_q.numpy()
    fixed = model.particle_mass.numpy() == 0
    tail = []
    max_tail_speed = 0.
    for step in range(600):
        state.clear_forces()
        collision.collide(state, contacts)
        solver.step(state, next_state, control, contacts, physics.physics_dt)
        state, next_state = next_state, state
        positions = state.particle_q.numpy()
        assert np.isfinite(positions).all()
        np.testing.assert_array_equal(positions[fixed], reference[fixed])
        if step >= 400:
            tail.append(positions)
            max_tail_speed = max(max_tail_speed, float(np.linalg.norm(state.particle_qd.numpy(), axis=-1).max()))
    # Natural gravity response must remain enabled, but unloaded motion must dissipate.
    assert np.linalg.norm(positions - reference, axis=-1).max() > .001
    assert np.linalg.norm(np.asarray(tail) - tail[0], axis=-1).max() < .001
    assert max_tail_speed < .01


@pytest.mark.skipif(os.environ.get("MICRODUCK_GPU_TESTS") != "1", reason="Requires SynReal login and CUDA")
@pytest.mark.parametrize("interactive,controller", [(False, "policy"), (True, "home"), (True, "policy")])
def test_bridge_starts_and_resets_without_presimulation(interactive, controller):
    import style3d_sim as s3d
    from examples.microduck.env import MicroDuckEnv

    # Construction and reset must not advance physics or run a hidden relaxation solver.
    with patch.object(s3d.NewtonSolver, "step", side_effect=AssertionError("Unexpected initialization step")):
        env = MicroDuckEnv(solver="synreal", num_envs=1, device="cuda:0", terrain="elastic_bridge",
                           interactive=interactive, controller=controller, randomize_foot_friction=False)
    try:
        indices = env.terrain.cloth_indices
        reference = wp.to_torch(env.model.particle_q)[indices].clone()
        fixed = wp.to_torch(env.model.particle_mass)[indices] == 0
        for _ in range(2):
            for state in env.states:
                torch.testing.assert_close(wp.to_torch(state.particle_q)[indices], reference, rtol=0, atol=0)
                assert not torch.count_nonzero(wp.to_torch(state.particle_qd)[indices])
                env._views(state)[0][:, 0] = -3.0
                env.sync_selected_state(state, torch.tensor([0], device=env.device))
            actions = torch.zeros_like(env.actions) if controller == "policy" else None
            for _ in range(3):
                env.advance_physics(actions)
            current = wp.to_torch(env.states[0].particle_q)[indices]
            assert torch.isfinite(current).all()
            assert (current[~fixed] - reference[~fixed]).norm(dim=-1).max() > 1e-4
            torch.testing.assert_close(current[fixed], reference[fixed], rtol=0, atol=1e-7)
            with patch.object(s3d.NewtonSolver, "step", side_effect=AssertionError("Unexpected reset step")):
                env.reset()
    finally:
        env.close()


def test_only_short_ends_are_fixed_and_middle_span_has_no_rigid_support():
    builder = newton.ModelBuilder()
    add_scene(builder, "elastic_bridge")
    nx, ny = SCENE_CONFIG.elastic_bridge_cells_x + 1, SCENE_CONFIG.elastic_bridge_cells_y + 1
    active = (np.asarray(builder.particle_flags).reshape(nx, ny) & int(newton.ParticleFlags.ACTIVE)) != 0
    assert active[1:-1].all()
    assert not active[[0, -1]].any()
    mass = np.asarray(builder.particle_mass).reshape(nx, ny)
    assert (mass[1:-1] > 0).all()
    assert (mass[[0, -1]] == 0).all()
    points = np.asarray(builder.particle_q).reshape(nx, ny, 3)
    np.testing.assert_allclose(points[0, :, 2], SCENE_CONFIG.elastic_bridge_height)
    np.testing.assert_allclose(points[-1, :, 2], SCENE_CONFIG.elastic_bridge_exit_height)
    mid_height = (SCENE_CONFIG.elastic_bridge_height + SCENE_CONFIG.elastic_bridge_exit_height) / 2
    np.testing.assert_allclose(points[nx // 2, :, 2], mid_height - SCENE_CONFIG.elastic_bridge_sag)
    assert len(builder.shape_body) == 11
    for label, transform, scale in zip(builder.shape_label, builder.shape_transform, builder.shape_scale, strict=True):
        if label.endswith("floor"):
            assert transform[2] + scale[2] == pytest.approx(0)
        elif label == "elastic_bridge_ramp":
            continue
        else:
            assert abs(transform[0]) - scale[0] >= SCENE_CONFIG.elastic_bridge_length / 2 - 1e-6
    approach = builder.shape_label.index("elastic_bridge_approach")
    anchor = builder.shape_label.index("elastic_bridge_anchor")
    deck_pose, deck_half = builder.shape_transform[approach], builder.shape_scale[approach]
    beam_pose, beam_half = builder.shape_transform[anchor], builder.shape_scale[anchor]
    assert deck_pose[0] + deck_half[0] == pytest.approx(beam_pose[0] - beam_half[0])
    assert deck_pose[2] + deck_half[2] == pytest.approx(beam_pose[2] + beam_half[2])
    assert beam_pose[0] + beam_half[0] == pytest.approx(-SCENE_CONFIG.elastic_bridge_length / 2)
    exit_index = builder.shape_label.index("elastic_bridge_exit")
    exit_pose, exit_half = builder.shape_transform[exit_index], builder.shape_scale[exit_index]
    assert exit_pose[0] == pytest.approx(-deck_pose[0])
    np.testing.assert_allclose(exit_half[:2], deck_half[:2])
    assert exit_pose[2] + exit_half[2] == pytest.approx(SCENE_CONFIG.elastic_bridge_exit_height)
    outlet_anchor = next(i for i, label in enumerate(builder.shape_label)
                         if label == "elastic_bridge_anchor" and builder.shape_transform[i][0] > 0)
    assert (builder.shape_transform[outlet_anchor][2] + builder.shape_scale[outlet_anchor][2]
            == pytest.approx(SCENE_CONFIG.elastic_bridge_exit_height))
    assert exit_pose[0] - exit_half[0] == pytest.approx(SCENE_CONFIG.elastic_bridge_length / 2 + .08)


@pytest.mark.parametrize("exit_height", [.55, .65, .75])
def test_end_ramps_support_cloth_without_initial_penetration(exit_height):
    config = replace(SCENE_CONFIG, elastic_bridge_exit_height=exit_height)
    builder = newton.ModelBuilder()
    metadata = add_scene(builder, "elastic_bridge", config)
    points = np.asarray(builder.particle_q)
    ramps = [i for i, label in enumerate(builder.shape_label) if label == "elastic_bridge_ramp"]
    assert len(ramps) == 2
    for index, ramp in zip(ramps, metadata["ramps"], strict=True):
        pose, half = builder.shape_transform[index], builder.shape_scale[index]
        corners = np.array([wp.transform_point(pose, wp.vec3(x, y, half[2]))
                            for x in (-half[0], half[0]) for y in (-half[1], half[1])])
        expected = np.array([ramp["start"], ramp["end"]])
        np.testing.assert_allclose(np.sort(corners[:, 0])[::2], np.sort(expected[:, 0]), atol=1e-7)
        np.testing.assert_allclose(np.sort(corners[:, 2])[::2], np.sort(expected[:, 2]), atol=1e-7)
        assert abs(ramp["end"][0]) == pytest.approx(config.elastic_bridge_length / 2 - config.elastic_bridge_ramp_length)
        assert half[1] == pytest.approx(config.elastic_bridge_width / 2)
        inverse = wp.transform_inverse(pose)
        local = np.array([wp.transform_point(inverse, wp.vec3(*point)) for point in points])
        over = (np.abs(local[:, 0]) <= half[0] + 1e-6) & (np.abs(local[:, 1]) <= half[1] + 1e-6)
        assert over.any()
        assert (local[over, 2] - half[2]).min() >= .003 - 1e-6
        assert builder.shape_body[index] == -1


def test_end_ramps_can_be_disabled():
    builder = newton.ModelBuilder()
    metadata = add_scene(builder, "elastic_bridge", replace(SCENE_CONFIG, elastic_bridge_ramp_length=0))
    assert not metadata["ramps"]
    assert "elastic_bridge_ramp" not in builder.shape_label


@pytest.mark.parametrize("field,value", [("cells_x", 1), ("cells_y", True), ("width", 0.05), ("density", float("nan")),
                                         ("damping", -1), ("sag", 1.0), ("approach_width", .1),
                                         ("approach_length", .1), ("spawn_distance", .75),
                                         ("approach_width", float("nan")), ("approach_length", float("inf")),
                                         ("ramp_length", -.01), ("ramp_length", .4), ("ramp_length", float("nan")),
                                         ("ramp_drop", -.01), ("ramp_drop", float("inf"))])
def test_invalid_bridge_parameters(field, value):
    with pytest.raises(ValueError):
        add_scene(newton.ModelBuilder(), "elastic_bridge", replace(SCENE_CONFIG, **{f"elastic_bridge_{field}": value}))


def test_support_and_reset_use_deformed_cloth_without_touching_other_world_or_robot():
    import style3d_sim as s3d
    from examples.microduck.policy_spec import build_robot

    template, _, _ = build_robot(s3d.ModelBuilder)
    with patch("examples.microduck.scenes.provision_pad_contacts"):
        metadata = add_scene(template, "elastic_bridge")
    builder = s3d.ModelBuilder()
    newton.solvers.SolverStyle3D.register_custom_attributes(builder)
    builder.replicate(template, 2)
    model = builder.finalize(device="cpu")
    env = SimpleNamespace(model=model, num_envs=2, device=torch.device("cpu"), cfg={}, _sync=lambda: None,
                          states=[model.state(), model.state()])
    terrain = SceneTerrain(env, "elastic_bridge", metadata)
    terrain.finalize()
    indices = terrain.cloth_indices.numpy()
    assert not np.intersect1d(indices, model.synreal_affine_body_indices.numpy()).size
    assert wp.to_torch(terrain.pad_particle).sum() == len(indices)
    selected = indices[terrain.cloth_worlds.numpy() == 0]
    other = np.setdiff1d(np.arange(model.particle_count), selected)
    for state in env.states:
        wp.to_torch(state.particle_q)[terrain.cloth_indices, 2] -= 0.05
        wp.to_torch(state.particle_qd).fill_(0.2)
    points = torch.tensor([[[0., 0., 0.8], [0., 0.5, 0.8]]] * 2)
    rest = (SCENE_CONFIG.elastic_bridge_height + SCENE_CONFIG.elastic_bridge_exit_height) / 2 - SCENE_CONFIG.elastic_bridge_sag
    torch.testing.assert_close(terrain.surface_heights(points), torch.tensor([[rest - .05, 0.]] * 2))
    before = [state.particle_q.numpy().copy() for state in env.states]
    terrain.reset(torch.tensor([0]))
    for state, original in zip(env.states, before, strict=True):
        np.testing.assert_array_equal(state.particle_q.numpy()[selected], model.particle_q.numpy()[selected])
        np.testing.assert_array_equal(state.particle_qd.numpy()[selected], 0)
        np.testing.assert_array_equal(state.particle_q.numpy()[other], original[other])
        np.testing.assert_allclose(state.particle_qd.numpy()[other], .2)
    torch.testing.assert_close(terrain.surface_heights(points), torch.tensor([[rest, 0.], [rest - .05, 0.]]))
    assert terrain.out_of_bounds(torch.tensor([[0., 0., .8], [0., .2, .8]])).tolist() == [False, True]
    # A rigid approach stays level even while the adjacent membrane deforms.
    inlet = -SCENE_CONFIG.elastic_bridge_length / 2
    points = torch.tensor([[[inlet - .4, 0., .8], [inlet, 0., .8], [inlet + .05, 0., .8]]] * 2)
    heights = terrain.surface_heights(points)
    torch.testing.assert_close(heights[:, :2], torch.full((2, 2), SCENE_CONFIG.elastic_bridge_height))
    assert (heights[:, 2] < SCENE_CONFIG.elastic_bridge_height).all()
    # The lowered second world's cloth exposes the rigid inlet ramp.
    ramp = metadata["ramps"][0]
    fraction = .05 / SCENE_CONFIG.elastic_bridge_ramp_length
    assert heights[1, 2].item() == pytest.approx(ramp["start"][2] + fraction * (ramp["end"][2] - ramp["start"][2]))
    assert not terrain.out_of_bounds(torch.tensor([[inlet - .001, 0., .8], [inlet + .001, 0., .8]])).any()
    assert terrain.side_out_of_bounds(torch.tensor([[inlet - .2, .25, .8], [inlet + .2, .25, .8]])).tolist() == [False, True]
    assert terrain.out_of_bounds(torch.tensor([[inlet - .81, 0., .8], [1.3, 0., .8]])).all()
    points = torch.tensor([[[.8, .2, .8], [.4, 0., .8]]] * 2)
    torch.testing.assert_close(terrain.surface_heights(points), torch.tensor(
        [[SCENE_CONFIG.elastic_bridge_exit_height, SCENE_CONFIG.elastic_bridge_exit_height]] * 2))
    assert not terrain.out_of_bounds(points[:, 0]).any()
    # Height queries must keep the outlet anchor and deck flush.
    seam = SCENE_CONFIG.elastic_bridge_length / 2 + .08
    seam_points = torch.tensor([[[seam - .001, 0., .8], [seam + .001, 0., .8]]] * 2)
    torch.testing.assert_close(terrain.surface_heights(seam_points), torch.tensor(
        [[SCENE_CONFIG.elastic_bridge_exit_height, SCENE_CONFIG.elastic_bridge_exit_height]] * 2))
    env.task_logic = SimpleNamespace(spawn=torch.tensor([[0., 0., .125], [0., 0., .125]]))
    spawn_x = inlet - SCENE_CONFIG.elastic_bridge_spawn_distance
    spawn_points = torch.tensor([[[spawn_x, 0., 0.]]] * 2)
    spawn_z = terrain.surface_heights(spawn_points)[0, 0].item() + .125
    assert inlet < spawn_x < 0
    assert spawn_z < SCENE_CONFIG.elastic_bridge_height + .125
    terrain.adjust_spawn(torch.tensor([0]))
    torch.testing.assert_close(env.task_logic.spawn[0], torch.tensor([spawn_x, 0., spawn_z]))
    torch.testing.assert_close(env.task_logic.spawn[1], torch.tensor([0., 0., .125]))
    assert not terrain.out_of_bounds(env.task_logic.spawn).any()
    reference = model.particle_q.numpy().copy()
    for state in env.states:
        wp.to_torch(state.particle_q)[terrain.cloth_indices, 2] -= .03
        wp.to_torch(state.particle_qd).fill_(.2)
        wp.to_torch(state.particle_f).fill_(.1)
    before = [(state.particle_q.numpy().copy(), state.particle_qd.numpy().copy(), state.particle_f.numpy().copy())
              for state in env.states]
    terrain.reset(torch.tensor([0]))
    for state, original in zip(env.states, before, strict=True):
        np.testing.assert_array_equal(state.particle_q.numpy()[selected], reference[selected])
        np.testing.assert_array_equal(state.particle_qd.numpy()[selected], 0)
        np.testing.assert_array_equal(state.particle_f.numpy()[selected], 0)
        for array, saved in zip((state.particle_q, state.particle_qd, state.particle_f), original, strict=True):
            np.testing.assert_array_equal(array.numpy()[other], saved[other])
    env.task_logic.spawn[:] = torch.tensor([0., 0., .125])
    terrain.adjust_spawn(torch.tensor([0]))
    torch.testing.assert_close(env.task_logic.spawn[0], torch.tensor([spawn_x, 0., spawn_z]))

    # Playback and manual resets query the cloth height at the selected spawn position.
    terrain.playback_center_spawn = True
    env.initial_q = torch.tensor([[0., 0., .125, 0., 0., 0., 1.]] * 2)
    env.task_logic = SimpleNamespace(spawn=env.initial_q.clone(), spawn_velocity=torch.ones(2, 6))
    env.task_logic.spawn[0, :2] = torch.tensor([-.55, .1])
    terrain.adjust_spawn(torch.tensor([0]))
    torch.testing.assert_close(env.task_logic.spawn[0], torch.tensor([spawn_x, 0., spawn_z, 0., 0., 0., 1.]))
    assert not env.task_logic.spawn_velocity[0].any()
    assert env.task_logic.spawn_velocity[1].eq(1).all()


def test_bridge_recipe_is_isolated_and_resampling_keeps_commands_along_bridge():
    from examples.microduck.env import MicroDuckEnv
    from examples.microduck.export_recipe import preserve_local_terms
    from examples.microduck.policy_spec import build_robot
    from examples.microduck.recipe import OfficialRecipe

    env = MicroDuckEnv(solver="mujoco", num_envs=1, device="cpu", task="velocity")
    try:
        ordinary = deepcopy(env.official.spec)
        env.terrain.kind = "elastic_bridge"
        env.terrain.cfg = {"config": asdict(SCENE_CONFIG)}
        robot, _, _ = build_robot(task="velocity")
        recipe = OfficialRecipe(env, robot)
        assert recipe.alignment["recipe_profile"] == "elastic_bridge_velocity_v1"
        assert recipe.reward_overrides["cloth_contact_slip"].__module__ == "examples.microduck.bridge_rewards"
        assert recipe.velocity_commands is None
        # The CPU fixture starts with flat-terrain pose randomization; place it on the path first.
        for state in env.states:
            env._views(state)[0][:, :2] = 0
            env._views(state)[0][:, 3:7] = torch.tensor([0., 0., 0., 1.])
            env.sync_selected_state(state, torch.arange(env.num_envs))
        recipe.bridge_path.reset(torch.arange(env.num_envs))
        assert recipe.spec["events"]["reset_base"]["params"]["pose_range"]["yaw"] == [-.05, .05]
        for step in (0, 12001, 48001):
            recipe.common_step_counter = step
            recipe.update_commands(torch.arange(env.num_envs))
            assert (env.commands[:, 1:3] == 0).all()
            vx = env.commands[:, 0]
            assert ((vx >= .15) & (vx <= .30)).all()
        value, terms = recipe.reward(*env._kinematics(), torch.zeros_like(env.actions), torch.zeros_like(env.actions))
        assert torch.isfinite(value).all()
        assert {"cloth_velocity", "cloth_contact_slip", "air_time"} <= terms.keys()
        env.terrain.kind = "default"
        restored = OfficialRecipe(env, robot)
        assert restored.spec == ordinary
        assert restored.reward_overrides == {}
        assert restored.velocity_commands is not None
        exported = {"tasks": {"velocity": {}}}
        preserve_local_terms(exported, {"tasks": {"velocity": ordinary}})
        assert "terrain_overrides" not in exported["tasks"]["velocity"]
    finally:
        env.close()


def test_bridge_uses_cloth_progress_reward():
    from examples.microduck import reward_functions as rewards
    from examples.microduck.tests.test_cloth_learning import cloth_fixture

    recipe, _, data, _, _ = cloth_fixture(2)
    recipe.env.terrain.kind = "elastic_bridge"
    data.root_link_lin_vel_b[:, 0] = torch.tensor([0., .2])
    value = rewards.cloth_command_progress(recipe, "twist", .03, 35., .09, .10, .35)
    assert value[1] > value[0]


def test_side_exit_is_failure_and_playback_does_not_reset():
    from examples.microduck.env import MicroDuckEnv

    env = MicroDuckEnv(solver="mujoco", num_envs=1, device="cpu", task="velocity")
    try:
        # Isolate termination semantics using CPU rigid physics and an empty soft-contact mask.
        env.terrain.kind = "elastic_bridge"
        env.terrain.cfg = {"config": asdict(SCENE_CONFIG)}
        env.terrain.side_out_of_bounds = lambda p: SceneTerrain.side_out_of_bounds(env.terrain, p)
        env.terrain.pad_particle = wp.zeros(env.model.particle_count, dtype=int, device="cpu")
        for state in env.states:
            env._views(state)[0][:, 1] = .21
            env.sync_selected_state(state, torch.tensor([0]))
        _, _, done, extras = env.step(torch.zeros(1, 14), apply_interval_events=False, auto_reset=False)
        assert done.item() and not extras["time_outs"].item()
        assert env.total_resets == 0
        env.step_playback(torch.zeros(1, 14))
        assert env.total_resets == 0
        assert env._views()[2][0, 1] > .18
    finally:
        env.close()


def test_json_scene_roundtrip_can_resume_but_changed_material_cannot(tmp_path):
    import json

    from examples.microduck.common import setup
    from examples.microduck.env import environment_config
    from examples.microduck.recipe import recipe_fingerprint

    checkpoint = tmp_path / "model_1.pt"
    checkpoint.touch()
    config = {
        "solver": "synreal", "task": "velocity", "recipe": "official_adapted_v1",
        "observation_contract": "official_asymmetric_pd_v2",
        "physics": asdict(environment_config("elastic_bridge").physics),
        "terrain": {"kind": "elastic_bridge", "config": asdict(SCENE_CONFIG)},
        "official_alignment": {"recipe_fingerprint": recipe_fingerprint("velocity", "elastic_bridge")},
        "ppo": {"policy": {"class_name": "ActorCritic"}},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    options = ["--solver", "synreal", "--resume", str(checkpoint), "--use-checkpoint-environment"]
    args, _, _ = setup("train", options)
    assert args.terrain == "elastic_bridge"
    config["terrain"]["config"]["elastic_bridge_stretch"] *= 2
    path.write_text(json.dumps(config))
    with pytest.raises(SystemExit):
        setup("train", options)
