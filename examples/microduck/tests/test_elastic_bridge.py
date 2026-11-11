"""Bridge anchoring, deformed support, selective reset, and walking command contracts."""

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


def test_only_short_ends_are_fixed_and_span_has_no_rigid_support():
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
    np.testing.assert_allclose(points[[0, -1], :, 2], SCENE_CONFIG.elastic_bridge_height)
    np.testing.assert_allclose(points[nx // 2, :, 2], SCENE_CONFIG.elastic_bridge_height - SCENE_CONFIG.elastic_bridge_sag)
    assert len(builder.shape_body) == 8
    for label, transform, scale in zip(builder.shape_label, builder.shape_transform, builder.shape_scale, strict=True):
        if label.endswith("floor"):
            assert transform[2] + scale[2] == pytest.approx(0)
        else:
            assert abs(transform[0]) - scale[0] >= SCENE_CONFIG.elastic_bridge_length / 2 - 1e-6
    approach = builder.shape_label.index("elastic_bridge_approach")
    anchor = builder.shape_label.index("elastic_bridge_anchor")
    deck_pose, deck_half = builder.shape_transform[approach], builder.shape_scale[approach]
    beam_pose, beam_half = builder.shape_transform[anchor], builder.shape_scale[anchor]
    assert deck_pose[0] + deck_half[0] == pytest.approx(beam_pose[0] - beam_half[0])
    assert deck_pose[2] + deck_half[2] == pytest.approx(beam_pose[2] + beam_half[2])
    assert beam_pose[0] + beam_half[0] == pytest.approx(-SCENE_CONFIG.elastic_bridge_length / 2)


@pytest.mark.parametrize("field,value", [("cells_x", 1), ("cells_y", True), ("width", 0.05), ("density", float("nan")),
                                         ("damping", -1), ("sag", 1.0), ("approach_width", .1),
                                         ("approach_length", .1), ("spawn_distance", .75),
                                         ("approach_width", float("nan")), ("approach_length", float("inf"))])
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
    rest = SCENE_CONFIG.elastic_bridge_height - SCENE_CONFIG.elastic_bridge_sag
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
    assert not terrain.out_of_bounds(torch.tensor([[inlet - .001, 0., .8], [inlet + .001, 0., .8]])).any()
    assert terrain.side_out_of_bounds(torch.tensor([[inlet - .2, .25, .8], [inlet + .2, .25, .8]])).tolist() == [False, True]
    assert terrain.out_of_bounds(torch.tensor([[inlet - .81, 0., .8], [1., 0., .8]])).all()
    env.task_logic = SimpleNamespace(spawn=torch.tensor([[0., 0., .125], [0., 0., .125]]))
    terrain.adjust_spawn(torch.tensor([0]))
    torch.testing.assert_close(env.task_logic.spawn[0], torch.tensor([0., 0., rest + .125]))
    torch.testing.assert_close(env.task_logic.spawn[1], torch.tensor([0., 0., .125]))
    assert not terrain.out_of_bounds(env.task_logic.spawn).any()
    # The relaxed shape is a reset target, never a replacement for the material's stress-free reference.
    reference = model.particle_q.numpy().copy()
    rest = reference[selected]
    relaxed = rest.copy()
    movable = (model.particle_flags.numpy()[selected] & int(newton.ParticleFlags.ACTIVE)) != 0
    relaxed[movable, 2] -= .02
    env.solver, env.physics = object(), object()
    with patch("examples.microduck.scenes.settled_bridge_state", return_value=(rest, relaxed, {"release_drift_m": 0.})):
        terrain.initialize_rest_state()
    np.testing.assert_array_equal(model.particle_q.numpy(), reference)
    before = [state.particle_q.numpy().copy() for state in env.states]
    for state in env.states:
        wp.to_torch(state.particle_q)[selected, 2] -= .03
    terrain.reset(torch.tensor([0]))
    for state, original in zip(env.states, before, strict=True):
        np.testing.assert_array_equal(state.particle_q.numpy()[selected], relaxed)
        np.testing.assert_array_equal(state.particle_q.numpy()[other], original[other])
    for state in env.states:
        state.particle_qd.zero_()
        state.clear_forces()
    shape = int(np.flatnonzero(model.shape_body.numpy() >= 0)[0])
    contacts = SimpleNamespace(soft_contact_count=wp.array([0], dtype=int, device="cpu"),
                               soft_contact_indices=wp.array([[int(selected[1]), -1, -1]], dtype=wp.vec3i, device="cpu"),
                               soft_contact_shape=wp.array([shape], dtype=int, device="cpu"))
    terrain.wake_bridge_contacts(contacts)
    assert terrain.bridge_sleeping.tolist() == [True, True]
    wp.to_torch(contacts.soft_contact_count)[0] = 1
    terrain.wake_bridge_contacts(contacts)
    assert terrain.bridge_sleeping.tolist() == [False, True]
    for state in env.states:
        wp.to_torch(state.particle_q)[terrain.cloth_indices, 2] -= .01
    awake = env.states[0].particle_q.numpy()[selected].copy()
    terrain.hold_sleeping_bridges()
    np.testing.assert_array_equal(env.states[0].particle_q.numpy()[selected], awake)
    sleeping = terrain.cloth_indices[terrain.cloth_worlds == 1]
    torch.testing.assert_close(wp.to_torch(env.states[0].particle_q)[sleeping], terrain.reset_particle_q[sleeping])
    wp.to_torch(env.states[0].particle_f)[sleeping[1], 2] = -.1
    terrain.wake_bridge_contacts(contacts)
    assert not terrain.bridge_sleeping.any()
    terrain.reset(torch.tensor([0]))
    assert terrain.bridge_sleeping.tolist() == [True, False]
    env.task_logic.spawn[:] = torch.tensor([0., 0., .125])
    terrain.adjust_spawn(torch.tensor([0]))
    torch.testing.assert_close(env.task_logic.spawn[0], torch.tensor([0., 0.,
                               SCENE_CONFIG.elastic_bridge_height - SCENE_CONFIG.elastic_bridge_sag - .02 + .125]))
    torch.testing.assert_close(env.task_logic.spawn[1], torch.tensor([0., 0., .125]))
    for state in env.states:
        np.testing.assert_array_equal(state.particle_qd.numpy()[selected], 0)


def test_bridge_recipe_is_isolated_and_resampling_keeps_commands_along_bridge():
    from examples.microduck.env import MicroDuckEnv
    from examples.microduck.export_recipe import preserve_local_terms
    from examples.microduck.policy_spec import build_robot
    from examples.microduck.recipe import OfficialRecipe

    env = MicroDuckEnv(solver="mujoco", num_envs=1, device="cpu", task="velocity")
    try:
        ordinary = deepcopy(env.official.spec)
        env.terrain.kind = "elastic_bridge"
        robot, _, _ = build_robot(task="velocity")
        recipe = OfficialRecipe(env, robot)
        assert recipe.velocity_commands is None
        assert recipe.spec["events"]["reset_base"]["params"]["pose_range"]["yaw"] == [-.05, .05]
        for step in (0, 12001, 48001):
            recipe.common_step_counter = step
            recipe.update_commands(torch.arange(env.num_envs))
            assert (env.commands[:, 1:3] == 0).all()
            vx = env.commands[:, 0]
            assert ((vx == 0) | ((vx >= .05) & (vx <= .20))).all()
        env.terrain.kind = "default"
        restored = OfficialRecipe(env, robot)
        assert restored.spec == ordinary
        assert restored.velocity_commands is not None
        exported = {"tasks": {"velocity": {}}}
        preserve_local_terms(exported, {"tasks": {"velocity": ordinary}})
        assert exported["tasks"]["velocity"]["terrain_overrides"] == ordinary["terrain_overrides"]
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
        "official_alignment": {"recipe_fingerprint": recipe_fingerprint()},
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
