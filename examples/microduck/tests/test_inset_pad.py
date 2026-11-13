"""A buried pad must have real clearance, visible negative heights and no hidden floor."""
from dataclasses import replace
from types import SimpleNamespace

import newton
import numpy as np
import pytest
import torch
import warp as wp

from examples.microduck.env import PAD_CONFIG
from examples.microduck.scenes import MixedTerrain, SoftPadConfig, add_inset_floor, add_soft_pad

# Test the buried-pad contract independently of the user's active experiment geometry.
INSET_PAD = replace(PAD_CONFIG, size=(1.2, 1.2, 0.06), cells=(4, 4, 3), base_z=-0.06, allow_edge_crossing=False)


def fixture(config=INSET_PAD):
    flat = newton.ModelBuilder()
    add_inset_floor(flat, config, has_pad=False)
    pad = newton.ModelBuilder()
    add_soft_pad(pad, config)
    add_inset_floor(pad, config, has_pad=True)
    builder = newton.ModelBuilder()
    builder.replicate(flat, 1)
    builder.replicate(pad, 1)
    env = SimpleNamespace(num_envs=2, device=torch.device('cpu'), cfg={}, _sync=lambda: None)
    terrain = MixedTerrain(env, builder, config, sampling='fixed_groups_center')
    env.model = builder.finalize(device='cpu')
    terrain.finalize()
    env.states = [env.model.state(), env.model.state()]
    return env, terrain


def test_buried_surface_is_flush_and_old_checkpoint_keeps_raised_pad():
    assert INSET_PAD.embedded and INSET_PAD.top_z == pytest.approx(0)
    old = SoftPadConfig.from_checkpoint({'pad': {'base_z': 0.0015}})
    assert not old.embedded and old.top_z == pytest.approx(0.0615)
    assert not old.allow_edge_crossing
    with pytest.raises(ValueError):
        replace(PAD_CONFIG, base_z=float('nan'))


@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('sign', [-1, 1])
def test_current_training_pad_is_flush_and_crosses_both_directions(axis, sign):
    assert PAD_CONFIG.embedded and PAD_CONFIG.top_z == pytest.approx(0)
    assert PAD_CONFIG.allow_edge_crossing
    env, terrain = fixture(replace(PAD_CONFIG, cells=(4, 4, 3)))
    assert env.cfg['terrain']['pad_edge_timeout_margin'] is None
    assert env.cfg['terrain']['boundary_curriculum'] == 'cross_surface_locomotion'
    restored = SoftPadConfig.from_checkpoint(env.cfg['terrain'])
    assert restored.allow_edge_crossing
    edge = terrain.config.size[axis] / 2
    for distance in (0.0, edge - 0.07, edge, edge + 0.1, edge, 0.0, 2.5, 2.51):
        pose = torch.zeros(2, 3)
        pose[:, axis] = sign * distance
        pose[:, 2] = 0.12
        terrain.update_support(pose)
        assert terrain.out_of_bounds(pose).tolist() == [distance > 2.5] * 2
    assert terrain.has_pad.tolist() == [False, True]


def test_floor_has_hole_only_in_pad_world_and_rigid_support_outside():
    env, terrain = fixture()
    model = env.model
    assert model.shape_count == 6
    assert not (model.shape_type.numpy() == newton.GeoType.PLANE).any()
    worlds = model.shape_world.numpy()
    centers = model.shape_transform.numpy()[:, :3]
    half = model.shape_scale.numpy()

    def rigid_height(world, x, y):
        inside = (worlds == world) & (np.abs(centers[:, 0] - x) <= half[:, 0])
        inside &= np.abs(centers[:, 1] - y) <= half[:, 1]
        return (centers[inside, 2] + half[inside, 2]).max()

    assert rigid_height(0, 0, 0) == pytest.approx(0)
    assert rigid_height(1, 0, 0) == pytest.approx(terrain.config.cavity_bottom_z)
    x_out, y_out = (half + 0.1 for half in terrain.config.cavity_half_extents)
    for world in (0, 1):
        for x, y in ((x_out, 0), (-x_out, 0), (0, y_out), (0, -y_out)):
            assert rigid_height(world, x, y) == pytest.approx(0)
    rest = terrain.rest[terrain.pad_indices[1]]
    assert rest[:, 2].min().item() == pytest.approx(-0.06)
    assert rest[:, 2].max().item() == pytest.approx(0)
    assert (rest[:, 2] >= terrain.config.base_z - 1e-7).all()


def test_depressed_surface_is_not_clamped_to_ground_or_other_world():
    env, terrain = fixture()
    q = wp.to_torch(env.states[0].particle_q)
    ids = terrain.pad_indices[1]
    # Compress the entire top while preserving the bonded bottom.
    q[ids, 2] = terrain.config.base_z + (q[ids, 2] - terrain.config.base_z) * 0.75
    x_out = terrain.config.cavity_half_extents[0] + 0.1
    queries = torch.tensor([[[0., 0., 0.1], [x_out, 0., 0.1]]] * 2)
    torch.testing.assert_close(terrain.surface_heights(queries), torch.tensor([[0., 0.], [-0.015, 0.]]))
    torch.testing.assert_close(terrain.heights(queries), torch.tensor([[0.1, 0.1], [0.115, 0.1]]))
    terrain.update_support(queries[:, 0])
    torch.testing.assert_close(terrain.origins[:, 2], torch.tensor([0., -0.015]))
    terrain.reset(torch.tensor([1]))
    torch.testing.assert_close(terrain.surface_heights(queries), torch.zeros(2, 2))


@pytest.mark.parametrize('axis,edge', [(axis, INSET_PAD.size[axis] / 2) for axis in (0, 1)])
@pytest.mark.parametrize('sign', [-1, 1])
def test_pad_group_times_out_before_each_edge_but_flat_group_does_not(axis, edge, sign):
    env, terrain = fixture()
    for distance in (0.0, edge - 0.09, edge - 0.07, edge, edge + 0.01, 1.0, 0.0):
        pose = torch.zeros(2, 3)
        pose[:, axis] = sign * distance
        pose[:, 2] = 0.12
        terrain.update_support(pose)
        assert terrain.out_of_bounds(pose).tolist() == [False, distance > edge - 0.08]
    assert env.cfg['terrain']['pad_edge_timeout_margin'] == 0.08
    assert env.cfg['terrain']['boundary_curriculum'] == 'surface_locomotion_before_edge_crossing'
    assert terrain.out_of_bounds(torch.tensor([[2.5, 0., 0.12]] * 2)).tolist() == [False, True]
    assert terrain.out_of_bounds(torch.tensor([[2.51, 0., 0.12]] * 2)).all()
    assert terrain.out_of_bounds(torch.tensor([[1.8, 1.8, 0.12]] * 2)).all()
    assert terrain.spawn_on_pad.tolist() == [False, True]


@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('sign', [-1, 1])
def test_playback_crosses_pad_edges_but_keeps_scene_boundary(axis, sign):
    from examples.microduck.play import configure_playback_boundaries

    env, terrain = fixture()
    env.mixed_terrain = True
    configure_playback_boundaries(env)
    for distance in (0.0, 0.51, 0.60, 0.70, 0.0, 2.5, 2.51):
        pose = torch.zeros(2, 3)
        pose[:, axis] = sign * distance
        assert terrain.out_of_bounds(pose).tolist() == [distance > 2.5] * 2
    # Historical per-episode sampling may spawn a ground cohort beside an existing pad.
    terrain.has_pad.fill_(True)
    assert not terrain.out_of_bounds(torch.zeros(2, 3)).any()


def test_collision_has_no_rigid_lid_over_pad():
    # A foot-sized rigid probe penetrates ground z=0. Only the flat world should
    # produce a rigid-ground contact at that location; the cavity floor is deeper.
    cfg = replace(INSET_PAD, cells=(2, 2, 1))
    builder = newton.ModelBuilder()
    for has_pad in (False, True):
        template = newton.ModelBuilder()
        body = template.add_body(xform=wp.transform(wp.vec3(0, 0, 0.01), wp.quat_identity()), mass=1.0)
        template.add_shape_sphere(body, radius=0.02)
        add_inset_floor(template, cfg, has_pad=has_pad)
        builder.replicate(template, 1)
    model = builder.finalize(device='cpu')
    state = model.state()
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    count = int(contacts.rigid_contact_count.numpy()[0])
    assert count > 0
    shape_world = model.shape_world.numpy()
    shape_body = model.shape_body.numpy()
    points = []
    for side in (0, 1):
        shapes = getattr(contacts, f'rigid_contact_shape{side}').numpy()[:count]
        local = getattr(contacts, f'rigid_contact_point{side}').numpy()[:count]
        points.append(np.asarray([
            state.body_q.numpy()[shape_body[shape], :3] + point
            if shape_body[shape] >= 0 else point for shape, point in zip(shapes, local, strict=True)
        ]))
    # Broadphase keeps separated speculative candidates; only negative gaps bear load.
    gap = ((points[1] - points[0]) * contacts.rigid_contact_normal.numpy()[:count]).sum(-1)
    gap -= contacts.rigid_contact_margin0.numpy()[:count] + contacts.rigid_contact_margin1.numpy()[:count]
    touching = gap <= 0
    assert touching.any()
    for shapes in (contacts.rigid_contact_shape0, contacts.rigid_contact_shape1):
        assert (shape_world[shapes.numpy()[:count][touching]] == 0).all()
