"""Offline pad checks and an opt-in authenticated SynReal contact regression."""

import os
import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import newton
import numpy as np
import pytest
import torch  # Load the bundled CUDA runtime before importing the native adapter.
import warp as wp

from examples.microduck.scenes import SoftPadConfig, add_soft_pad, tetrahedral_box_mesh


def boundary_faces(tetrahedra):
    faces = np.concatenate([tetrahedra[:, face] for face in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))])
    unique, counts = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True)
    assert np.isin(counts, (1, 2)).all(), "A mesh face has more than two incident tetrahedra"
    return unique[counts == 1]


def test_mesh_density_tracks_size_and_rounds_up_without_float_boundary_errors():
    config = SoftPadConfig(size=(1.1, 0.171, 0.001), cells_per_meter=100.0)
    assert config.cells is None
    assert config.resolved_cells == (110, 18, 1)
    resized = replace(config, size=(0.55, 0.2, 0.1))
    assert resized.resolved_cells == (55, 20, 10)
    anisotropic = SoftPadConfig(size=(1.2, 1.2, 0.12), cells_per_meter=(37.5, 37.5, 50.0))
    assert anisotropic.resolved_cells == (45, 45, 6)
    assert replace(anisotropic, size=(0.8, 0.8, 0.12)).resolved_cells == (30, 30, 6)


@pytest.mark.parametrize("density", [0, -1, float('nan'), float('inf'), True, (20, 30), (20, 0, 30), (20, True, 30)])
def test_invalid_mesh_density_is_rejected(density):
    with pytest.raises(ValueError, match="cells_per_meter"):
        SoftPadConfig(cells_per_meter=density)


def test_density_checkpoint_roundtrip_and_explicit_legacy_cells():
    config = SoftPadConfig(size=(1.2, 1.2, 0.12), cells_per_meter=(37.5, 37.5, 50.0))
    metadata = json.loads(json.dumps({'pad': asdict(config)}))
    restored = SoftPadConfig.from_checkpoint(metadata)
    assert restored.cells is None and restored.resolved_cells == (45, 45, 6)
    assert replace(restored, size=(0.8, 0.8, 0.12)).resolved_cells == (30, 30, 6)
    legacy = SoftPadConfig.from_checkpoint({'pad': {'size': [1.2, 1.2, 0.12], 'cells': [45, 45, 6]}})
    assert legacy.cells_per_meter is None and legacy.resolved_cells == (45, 45, 6)
    assert replace(legacy, size=(0.8, 0.8, 0.12)).resolved_cells == (45, 45, 6)
    assert SoftPadConfig.from_checkpoint({'pad': {}}).resolved_cells == (24, 16, 3)
    # Existing callers can explicitly choose a small mesh for a diagnostic or test.
    assert replace(config, cells=(2, 2, 1)).resolved_cells == (2, 2, 1)


def test_density_mesh_matches_equivalent_explicit_geometry():
    config = SoftPadConfig(size=(0.4, 0.3, 0.15), cells_per_meter=20.0)
    automatic, explicit = newton.ModelBuilder(), newton.ModelBuilder()
    add_soft_pad(automatic, config)
    add_soft_pad(explicit, replace(config, cells=(8, 6, 3)))
    assert len(automatic.tet_indices) == 6 * 8 * 6 * 3
    for name in ('particle_q', 'particle_mass', 'tri_indices', 'tet_indices'):
        np.testing.assert_array_equal(np.asarray(getattr(automatic, name)), np.asarray(getattr(explicit, name)))


@pytest.mark.parametrize('flags,expected', [
    ([], (16, 12, 2)),
    (['--cells-per-meter', '40'], (32, 24, 4)),
    (['--cells-per-meter', '40', '30', '20'], (32, 18, 2)),
    (['--cells', '2', '3', '1'], (2, 3, 1)),
])
def test_scene_size_and_resolution_overrides(monkeypatch, flags, expected):
    from examples.microduck import env as environment, scenes

    profile = environment.environment_config('mixed_soft_pad')
    profile = replace(profile, pad=replace(profile.pad, cells=None, cells_per_meter=20.0))
    monkeypatch.setitem(environment.ENVIRONMENT_CONFIGS, 'mixed_soft_pad', profile)

    class BeforeConstruction(Exception):
        pass

    def capture(**kwargs):
        assert kwargs['pad_config'].resolved_cells == expected
        raise BeforeConstruction

    monkeypatch.setattr("examples.microduck.env.MicroDuckEnv", capture)
    args = scenes.create_parser().parse_args(['--preview', '--device', 'cpu', '--size', '0.8', '0.6', '0.1'] + flags)
    with pytest.raises(BeforeConstruction):
        scenes.SoftPadScene(None, args)


@pytest.mark.parametrize("base_z", [0, 0.0, -0.06])
def test_scene_bottom_check_accepts_integer_height_and_detects_motion(base_z):
    from examples.microduck.env import MicroDuckEnv

    config = replace(SoftPadConfig(), base_z=base_z, cells=(2, 2, 1))
    builder = newton.ModelBuilder()
    builder.add_body(mass=1.0)
    add_soft_pad(builder, config)
    model = builder.finalize(device="cpu")
    state = model.state()
    positions = wp.to_torch(state.particle_q)
    rest = positions.clone()
    terrain = SimpleNamespace(config=config, rest=rest, pad_indices=torch.arange(len(rest)), check_state=lambda: None)
    env = SimpleNamespace(_sync=lambda: None, states=[state], terrain=terrain, device="cpu")
    MicroDuckEnv.check_scene_state(env)
    bottom = torch.argmin(rest[:, 2])
    positions[bottom, 0] += 0.001
    with pytest.raises(AssertionError, match="bonded bottom"):
        MicroDuckEnv.check_scene_state(env)


@pytest.mark.parametrize("cells", [(1, 1, 1), (3, 2, 4), SoftPadConfig().cells])
def test_tetrahedra_fill_box_without_internal_boundary_faces(cells):
    size = SoftPadConfig().size
    vertices, tetrahedra = tetrahedral_box_mesh(size, cells)
    points = vertices[tetrahedra].astype(np.float64)
    volumes = (
        np.einsum("ij,ij->i", points[:, 1] - points[:, 0], np.cross(points[:, 2] - points[:, 0], points[:, 3] - points[:, 0]))
        / 6.0
    )
    assert np.all(volumes > 0.0)
    assert volumes.sum() == pytest.approx(np.prod(size), rel=1.0e-6)

    boundary = boundary_faces(tetrahedra)
    faces = vertices[boundary]
    on_outer_plane = np.zeros(len(faces), dtype=bool)
    for axis, length in enumerate(size):
        for sign in (-1, 1):
            on_outer_plane |= np.isclose(faces[:, :, axis], sign * length / 2.0, atol=1.0e-7).all(axis=1)
    assert on_outer_plane.all(), "A crack or unmatched interior face is present in the pad"
    nx, ny, nz = cells
    assert len(boundary) == 4 * (nx * ny + nx * nz + ny * nz)


@pytest.mark.parametrize("cells", [(2, 3, 2), (6, 4, 3)])
def test_free_pad_mass_matches_density_times_volume(cells):
    config = replace(SoftPadConfig(), cells=cells, fix_bottom=False)
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    pad = add_soft_pad(builder, config)
    assert np.sum(builder.particle_mass) == pytest.approx(config.density * np.prod(config.size), rel=1.0e-6)
    assert not len(pad["fixed_particles"])
    assert all(int(flags) & int(newton.ParticleFlags.ACTIVE) for flags in builder.particle_flags)


def test_only_z_bottom_face_is_fixed_and_ranges_exclude_prior_geometry():
    config = replace(SoftPadConfig(), cells=(3, 2, 3), base_z=0.17)
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.add_particle(pos=wp.vec3(2.0), vel=wp.vec3(0.0), mass=1.0)
    pad = add_soft_pad(builder, config)
    start, stop = pad["particle_range"]
    assert start == 1
    positions = np.asarray(builder.particle_q)
    masses = np.asarray(builder.particle_mass)
    flags = np.asarray(builder.particle_flags, dtype=np.int32)
    expected = np.flatnonzero(np.isclose(positions[:, 2], config.base_z, atol=1.0e-7))
    np.testing.assert_array_equal(pad["fixed_particles"], expected)
    assert len(expected) == (config.cells[0] + 1) * (config.cells[1] + 1)
    assert np.all(masses[expected] == 0.0)
    assert np.all((flags[expected] & int(newton.ParticleFlags.ACTIVE)) == 0)
    free = np.setdiff1d(np.arange(start, stop), expected)
    assert np.all(masses[free] > 0.0)
    assert np.all((flags[free] & int(newton.ParticleFlags.ACTIVE)) != 0)
    assert masses[0] == 1.0
    assert pad["triangle_range"] == (0, len(builder.tri_indices))
    assert pad["tetrahedron_range"] == (0, len(builder.tet_indices))
    assert positions[start:stop, 2].max() == pytest.approx(config.top_z)


def test_finalized_surface_is_closed_and_excludes_affine_proxy_particles():
    import style3d_sim as s3d

    builder = s3d.ModelBuilder(up_axis=newton.Axis.Z)
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()))
    builder.add_shape_sphere(body, radius=0.025)
    pad = add_soft_pad(builder, replace(SoftPadConfig(), cells=(3, 2, 2)))
    model = builder.finalize(device="cpu")
    start, stop = pad["triangle_range"]
    surface = model.tri_indices.numpy()[start:stop]
    tet_start, tet_stop = pad["tetrahedron_range"]
    expected = boundary_faces(model.tet_indices.numpy()[tet_start:tet_stop])
    actual = np.sort(surface, axis=1)
    np.testing.assert_array_equal(actual[np.lexsort(actual.T[::-1])], expected[np.lexsort(expected.T[::-1])])

    proxy_indices = model.synreal_affine_body_indices.numpy().ravel()
    assert len(proxy_indices) == 4
    assert not np.isin(surface, proxy_indices).any()
    assert not np.isin(model.tet_indices.numpy()[tet_start:tet_stop], proxy_indices).any()


@pytest.mark.parametrize("base_z", [-0.06, 0.0])
def test_scene_preserves_robot_and_floor_contact_materials(monkeypatch, base_z):
    from newton.viewer import ViewerNull

    import style3d_sim as s3d
    from examples.microduck import env as environment
    from examples.microduck.policy_spec import build_robot
    from examples.microduck.scenes import SoftPadScene, create_parser

    profile = environment.environment_config("mixed_soft_pad")
    monkeypatch.setitem(environment.ENVIRONMENT_CONFIGS, "mixed_soft_pad",
                        replace(profile, pad=replace(profile.pad, base_z=base_z)))

    reference, _, _ = build_robot(s3d.ModelBuilder)
    floor = reference.add_ground_plane()
    expected = {name: np.asarray(getattr(reference, f"shape_material_{name}")) for name in ("ke", "kd", "mu")}
    args = create_parser().parse_args(["--preview", "--device", "cpu", "--viewer", "null", "--cells", "2", "2", "2"])

    class MaterialsValidated(Exception):
        pass

    def inspect_materials(builder, *args, **kwargs):
        # Inspect the actual scene builder before CUDA-only SDF construction.
        for name, values in expected.items():
            actual = np.asarray(getattr(builder, f"shape_material_{name}"))
            np.testing.assert_allclose(actual[:floor], values[:floor], rtol=1.0e-6)
            np.testing.assert_allclose(actual[floor:], values[floor], rtol=1.0e-6)
        expected_geometry = newton.GeoType.BOX if base_z < 0 else newton.GeoType.PLANE
        assert all(kind == expected_geometry for kind in builder.shape_type[floor:])
        raise MaterialsValidated

    monkeypatch.setattr(s3d.ModelBuilder, "finalize", inspect_materials)
    viewer = ViewerNull(num_frames=1)
    try:
        with pytest.raises(MaterialsValidated):
            SoftPadScene(viewer, args)
    finally:
        viewer.close()


@pytest.mark.skipif(
    os.environ.get("MICRODUCK_TEST_SOFT_PAD_GPU") != "1" or not torch.cuda.is_available(),
    reason="Set MICRODUCK_TEST_SOFT_PAD_GPU=1 on a CUDA host; SynReal login credentials are required",
)
def test_soft_pad_gpu_stability_and_reset():
    """Exercise the real rigid/soft contact path on a small pad when explicitly enabled."""
    from newton.viewer import ViewerNull

    from examples.microduck.scenes import SoftPadScene, create_parser

    # Keep this regression inexpensive while retaining the old contact geometry that
    # reproduced the instability. Authentication remains the scene's normal login path.
    previous_device = wp.get_device()
    wp.set_device("cuda:0")
    args = create_parser().parse_args(
        [
            "--viewer",
            "null",
            "--device",
            "cuda:0",
            "--num-frames",
            "150",
            "--size",
            "0.9",
            "0.55",
            "0.06",
            "--cells",
            "12",
            "8",
            "3",
            "--substeps",
            "8",
        ]
    )
    viewer = ViewerNull(num_frames=150)
    scene = None

    def assert_bounded_state():
        particles = scene.state_0.particle_q.numpy()
        body_velocity = scene.state_0.body_qd.numpy()
        volumes = scene.six_volumes(particles) / scene.rest_volumes
        pad_displacement = np.linalg.norm(particles[scene.pad_indices] - scene.rest[scene.pad_indices], axis=1)
        pad_speed = np.linalg.norm(scene.state_0.particle_qd.numpy()[scene.pad_indices], axis=1)
        body_angular_speed = np.linalg.norm(body_velocity[:, 3:], axis=1)

        scene.check_state()
        assert np.isfinite(volumes).all()
        assert np.isfinite(pad_displacement).all()
        assert np.isfinite(pad_speed).all()
        assert np.isfinite(body_angular_speed).all()
        assert volumes.min() > 0.0, "The sponge contains an inverted tetrahedron"
        # HOME targets do not balance a falling robot. These limits allow its impact
        # but reject the numerical explosion observed before the solver fix.
        assert pad_displacement.max() < 0.1, "The sponge displacement became unbounded"
        assert pad_speed.max() < 10.0, "The sponge velocity became unbounded"
        assert body_angular_speed.max() < 100.0, "Rigid body angular velocity became explosive"

    try:
        scene = SoftPadScene(viewer, args)
        for _ in range(120):
            scene.step()
            assert_bounded_state()

        scene.reset()
        assert scene.steps == 0
        assert scene.sim_time == 0.0
        for state in (scene.state_0, scene.state_1):
            for field in ("particle_q", "particle_qd", "body_q", "body_qd", "joint_q", "joint_qd"):
                np.testing.assert_allclose(
                    getattr(state, field).numpy(),
                    getattr(scene.initial_state, field).numpy(),
                    atol=1.0e-6,
                    rtol=0.0,
                    err_msg=f"Reset did not restore {field}",
                )
        np.testing.assert_array_equal(scene.control.joint_target_q.numpy(), scene.initial_targets.numpy())
        # Advance past the original 0.2-0.35 s failure window again so stale drive
        # history cannot hide behind an otherwise correct state-buffer reset.
        for _ in range(30):
            scene.step()
            assert_bounded_state()
    finally:
        viewer.close()
        wp.set_device(previous_device)
