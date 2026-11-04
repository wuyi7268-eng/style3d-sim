"""Optional training and playback scenes share one authored scene module."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import newton
import numpy as np
import pytest
import torch

from examples.microduck.common import setup
from examples.microduck.scenes import SCENE_CONFIG, SCENE_KINDS, add_scene


@pytest.mark.skipif(
    os.environ.get("MICRODUCK_GPU_TESTS") != "1" or not torch.cuda.is_available(),
    reason="Static scene preview needs CUDA texture SDF construction; enable MICRODUCK_GPU_TESTS=1",
)
def test_standalone_scene_preview_without_solver_or_login():
    process_env = {key: value for key, value in os.environ.items()
                   if key not in ("STYLE3D_ACCOUNT", "STYLE3D_PASSWORD", "STYLE3D_SIM_ACCOUNT", "STYLE3D_SIM_PASSWORD")}
    result = subprocess.run(
        [sys.executable, "-m", "examples.microduck.scenes", "--preview", "--viewer", "null",
         "--device", "cuda:0", "--num-frames", "1", "--cells", "4", "4", "2", "--test"],
        cwd=Path(__file__).resolve().parents[3], env=process_env, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "INITIAL LAYOUT ONLY (no physics)" in result.stdout


@pytest.mark.parametrize("kind", SCENE_KINDS)
def test_scene_builder_authors_expected_support(kind):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    metadata = add_scene(builder, kind)
    assert metadata["kind"] == kind
    assert metadata["config"]
    if kind == "hammock":
        assert len(builder.particle_q) == SCENE_CONFIG.hammock_resolution**2
        flags = np.asarray(builder.particle_flags)
        assert not (flags[0] & int(newton.ParticleFlags.ACTIVE))
        assert not (flags[-1] & int(newton.ParticleFlags.ACTIVE))
        assert metadata["support"] == "sagging_cloth_between_fixed_ends"
    elif kind == "sand":
        assert metadata["grain_count_per_world"] == len(builder.particle_q)
        assert SCENE_CONFIG.sand_spacing == pytest.approx(0.002)
        assert SCENE_CONFIG.sand_length == pytest.approx(0.32)
        assert SCENE_CONFIG.sand_width == pytest.approx(0.24)
        assert metadata["grain_radius"] == pytest.approx(0.00096)
        assert metadata["grain_count_per_world"] == 160 * 120 * round(SCENE_CONFIG.sand_depth / SCENE_CONFIG.sand_spacing)
        assert metadata["grain_radius"] * 2 == pytest.approx(0.00192)
    elif kind == "pool":
        assert len(builder.shape_body) == 5
        assert metadata["fluid_count_per_world"] == len(builder.particle_q) == 5400
        assert metadata["support"] == "native_particle_fluid"
        assert np.all(np.asarray(builder.particle_mass) > 0)
    else:
        assert len(builder.shape_body) == SCENE_CONFIG.step_count + 1
        assert metadata["support"] == "rigid_stair_treads"


@pytest.mark.parametrize("kind", SCENE_KINDS)
def test_scene_selection_is_available_for_synreal_training_and_playback(kind):
    train, _, _ = setup("train", ["--solver", "synreal", "--terrain", kind])
    assert train.terrain == kind
    assert train.environment_profile == kind
    if kind in ("sand", "pool"):
        assert train.num_envs == 1
    with pytest.raises(SystemExit):
        setup("play", ["--solver", "synreal", "--terrain", kind])


def test_sand_scene_registers_native_grains_after_world_replication():
    from examples.microduck.scenes import SceneTerrain

    template = newton.ModelBuilder(up_axis=newton.Axis.Z)
    metadata = add_scene(template, "sand")
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.replicate(template, 2)
    model = builder.finalize(device="cpu")
    env = SimpleNamespace(model=model, num_envs=2, device=torch.device("cpu"), cfg={})
    terrain = SceneTerrain(env, "sand", metadata)
    terrain.finalize()
    assert terrain.cfg["granular_particle_count"] == metadata["grain_count_per_world"] * 2
    assert len(model.synreal_granular_particle_indices) == terrain.cfg["granular_particle_count"]


def test_pool_fluid_registration_and_selected_world_reset():
    import warp as wp
    import style3d_sim as s3d
    from examples.microduck.policy_spec import build_robot
    from examples.microduck.scenes import SceneTerrain

    template, _, _ = build_robot(s3d.ModelBuilder)
    metadata = add_scene(template, "pool")
    builder = s3d.ModelBuilder()
    builder.replicate(template, 2)
    env = SimpleNamespace(model=builder.finalize(device="cpu"), num_envs=2, device=torch.device("cpu"),
                          cfg={}, _sync=lambda: None)
    terrain = SceneTerrain(env, "pool", metadata)
    terrain.finalize()
    model = env.model
    indices = model.synreal_fluid_particle_indices.numpy()
    assert len(indices) == 2 * metadata["fluid_count_per_world"]
    assert not np.intersect1d(indices, model.synreal_affine_body_indices.numpy()).size
    assert model.synreal_fluid_rest_density == 1000.0
    boundaries = model.synreal_fluid_boundary_positions.numpy()
    assert len(boundaries) > 0
    assert np.all(boundaries[:, 2] < SCENE_CONFIG.pool_wall_height)
    env.states = [model.state(), model.state()]
    selected = indices[model.particle_world.numpy()[indices] == 0]
    other = np.setdiff1d(np.arange(model.particle_count), selected)
    for state in env.states:
        wp.to_torch(state.particle_q).add_(0.03)
        wp.to_torch(state.particle_qd).fill_(0.2)
    before = [state.particle_q.numpy().copy() for state in env.states]
    terrain.reset(torch.tensor([0]))
    for state, original in zip(env.states, before, strict=True):
        np.testing.assert_array_equal(state.particle_q.numpy()[selected], model.particle_q.numpy()[selected])
        np.testing.assert_array_equal(state.particle_qd.numpy()[selected], 0)
        np.testing.assert_array_equal(state.particle_q.numpy()[other], original[other])
        np.testing.assert_allclose(state.particle_qd.numpy()[other], 0.2)
    env.task_logic = SimpleNamespace(spawn=torch.zeros(2, 21))
    terrain.adjust_spawn(torch.tensor([0]))
    assert env.task_logic.spawn[0, 2] == pytest.approx(SCENE_CONFIG.pool_depth)
    assert terrain.origins[0, 2] == 0  # Water must not become a fake rigid contact surface.


@pytest.mark.parametrize("world_count", [1, 3])
def test_sand_with_robot_excludes_affine_proxies(world_count):
    import style3d_sim as s3d
    from examples.microduck.policy_spec import build_robot
    from examples.microduck.scenes import SceneTerrain

    template, _, _ = build_robot(s3d.ModelBuilder)
    metadata = add_scene(template, "sand")
    authored = np.asarray(template.particle_q).copy()
    builder = s3d.ModelBuilder(up_axis=newton.Axis.Z)
    builder.replicate(template, world_count)
    model = builder.finalize(device="cpu")
    env = SimpleNamespace(model=model, num_envs=world_count, device=torch.device("cpu"), cfg={})
    terrain = SceneTerrain(env, "sand", metadata)
    terrain.finalize()
    grains = model.synreal_granular_particle_indices.numpy()
    proxies = model.synreal_affine_body_indices.numpy().ravel()
    assert len(proxies) > 0
    assert not np.intersect1d(grains, proxies).size
    worlds = model.particle_world.numpy()
    for world in range(world_count):
        selected = grains[worlds[grains] == world]
        assert len(selected) == metadata["grain_count_per_world"]
        np.testing.assert_allclose(model.particle_q.numpy()[selected], authored, atol=1e-7)


@pytest.mark.skipif(
    os.environ.get("MICRODUCK_GPU_TESTS") != "1" or not torch.cuda.is_available(),
    reason="Hammock mesh SDF construction requires CUDA; enable MICRODUCK_GPU_TESTS=1",
)
def test_hammock_environment_generates_triangle_interior_contacts():
    import warp as wp
    from examples.microduck.env import MicroDuckEnv

    env = MicroDuckEnv(solver="synreal", num_envs=1, device="cuda:0", terrain="hammock",
                       interactive=True, controller="home", preview=True, randomize_foot_friction=False)
    try:
        assert env.cfg["collision_options"]["enable_rigid_soft_full_surface_contact"]
        q = env.model.joint_q.numpy()
        q[2] = 1.07
        env.model.joint_q.assign(q)
        newton.eval_fk(env.model, env.model.joint_q, env.model.joint_qd, env.states[0])
        env.collision.collide(env.states[0], env.contacts)
        wp.synchronize_device(env.model.device)
        count = int(env.contacts.soft_contact_count.numpy()[0])
        interior = env.contacts.soft_contact_particle.numpy()[:count] < 0
        assert interior.any(), "A rendered cloth face must collide between its vertices"
        indices = env.contacts.soft_contact_indices.numpy()[:count][interior]
        # Edge records use -1 for their unused third vertex; face records have three IDs.
        assert (indices >= 0).all(axis=1).any()
        assert np.isin(indices[indices >= 0], np.unique(env.model.tri_indices.numpy())).all()
    finally:
        env.close()


@pytest.mark.parametrize("kind", SCENE_KINDS)
def test_scene_checkpoint_infers_profile_and_rejects_changed_scene_config(kind, tmp_path):
    checkpoint = tmp_path / "model_1.pt"
    checkpoint.touch()
    metadata = {
        "solver": "synreal",
        "task": "velocity",
        "recipe": "official_adapted_v1",
        "observation_contract": "official_asymmetric_pd_v2",
        "official_alignment": {"recipe_fingerprint": "unused"},
        "ppo": {},
        "terrain": {"kind": kind, "config": {"changed": True}},
    }
    (tmp_path / "config.json").write_text(json.dumps(metadata))
    with pytest.raises(SystemExit):
        setup("train", ["--solver", "synreal", "--resume", str(checkpoint)])
    args, _, _ = setup("play", ["--solver", "synreal", "--checkpoint", str(checkpoint),
                                "--allow-recipe-mismatch"])
    assert args.terrain == kind
    assert args.environment_profile == kind
