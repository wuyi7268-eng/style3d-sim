"""A thin collider inside a large pad face still needs an upward support constraint."""

from types import SimpleNamespace

import newton
import numpy as np
import pytest
import torch
import warp as wp

from examples.microduck.scenes import MixedTerrain, SoftPadConfig, add_soft_pad


def scene(*, x=0.03, z=0.043, capacity=16):
    config = SoftPadConfig(size=(6.0, 6.0, 0.06), cells=(1, 1, 1), base_z=0.0)
    template = newton.ModelBuilder()
    body = template.add_body(xform=wp.transform(wp.vec3(x, 0.02, z), wp.quat_identity()), mass=1.0)
    # A closed tetrahedron has a unique lowest vertex, completely inside the pad volume.
    mesh = newton.Mesh(
        vertices=np.array([[0, 0, -0.002], [-0.02, -0.02, 0.002], [0.02, -0.02, 0.002], [0, 0.02, 0.002]], np.float32),
        indices=np.array([0, 2, 1, 0, 3, 2, 0, 1, 3, 1, 2, 3], np.int32),
    )
    template.add_shape_mesh(body, mesh=mesh)
    visual = template.default_shape_cfg.copy()
    visual.has_shape_collision = visual.has_particle_collision = False
    template.add_shape_mesh(body, mesh=mesh, cfg=visual)
    builder = newton.ModelBuilder()
    builder.replicate(template, 1)
    add_soft_pad(template, config)
    builder.replicate(template, 1)
    env = SimpleNamespace(num_envs=2, device=torch.device("cpu"), cfg={}, _sync=lambda: None)
    env.terrain = MixedTerrain(env, builder, config, 0.5, sampling="fixed_groups_center")
    env.model = builder.finalize(device="cpu")
    env.terrain.finalize()
    env.states = [env.model.state(), env.model.state()]
    env.contacts = newton.Contacts(16, capacity, device="cpu")
    env.contacts._enable_rigid_soft_full_surface_contact = True
    return env


def test_submerged_mesh_gets_support_from_its_own_pad_without_moving_state():
    env = scene()
    state = env.states[0]
    before = {name: getattr(state, name).numpy().copy() for name in ("body_q", "particle_q")}
    env.terrain.check_contacts()
    c = env.contacts
    assert c.soft_contact_count.numpy()[0] == 1
    shape = c.soft_contact_shape.numpy()[0]
    assert env.model.shape_world.numpy()[shape] == 1
    tri = c.soft_contact_indices.numpy()[0]
    assert (env.model.particle_world.numpy()[tri] == 1).all()
    weights = c.soft_contact_barycentric.numpy()[0]
    assert weights.sum() == pytest.approx(1.0)
    assert weights.min() >= 0
    point = (state.particle_q.numpy()[tri] * weights[:, None]).sum(0)
    np.testing.assert_allclose(point, [0.03, 0.02, 0.06], atol=1e-6)
    np.testing.assert_allclose(c.soft_contact_normal.numpy()[0], [0, 0, -1], atol=1e-6)
    np.testing.assert_allclose(c.soft_contact_body_pos.numpy()[0], [0, 0, -0.002], atol=1e-6)
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(state, name).numpy(), value)


@pytest.mark.parametrize("x,z", [(3.2, 0.043), (0.03, 0.09), (0.03, -0.03)])
def test_no_infinite_lid_or_support_below_pad(x, z):
    env = scene(x=x, z=z)
    env.terrain.check_contacts()
    assert env.contacts.soft_contact_count.numpy()[0] == 0


def test_support_follows_deformed_surface_and_not_rest_height():
    env = scene()
    positions = env.states[0].particle_q.numpy()
    top = np.unique(env.terrain.top_faces.numpy()[1].ravel())
    positions[top, 2] = 0.05 + positions[top, 0] * 0.002
    env.states[0].particle_q.assign(positions)
    env.terrain.check_contacts()
    c = env.contacts
    assert c.soft_contact_count.numpy()[0] == 1
    tri = c.soft_contact_indices.numpy()[0]
    point = (positions[tri] * c.soft_contact_barycentric.numpy()[0, :, None]).sum(0)
    assert point[2] == pytest.approx(0.05006, abs=1e-6)
    normal = c.soft_contact_normal.numpy()[0]
    assert np.linalg.norm(normal) == pytest.approx(1.0, abs=1e-6)
    assert normal[2] < 0


def test_support_contact_overflow_fails_without_out_of_bounds_write():
    env = scene(capacity=1)
    env.contacts.soft_contact_count.assign(np.array([1], np.int32))
    with pytest.raises(RuntimeError, match="soft contact capacity exceeded"):
        env.terrain.check_contacts()


def test_rechecking_one_collision_generation_does_not_duplicate_contacts():
    env = scene(x=0.02)
    env.terrain.check_contacts()
    assert env.contacts.soft_contact_count.numpy()[0] == 1
    env.terrain.check_contacts()
    assert env.contacts.soft_contact_count.numpy()[0] == 1
    env.contacts.clear()
    env.terrain.check_contacts()
    assert env.contacts.soft_contact_count.numpy()[0] == 1
