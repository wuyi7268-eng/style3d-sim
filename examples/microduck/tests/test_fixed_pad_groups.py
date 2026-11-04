"""Sparse soft topology, fixed center resets, and isolation between heterogeneous worlds."""
from types import SimpleNamespace

import newton
import numpy as np
import pytest
import torch
import warp as wp

from examples.microduck.scenes import MixedTerrain, pad_world_count, SoftPadConfig, add_soft_pad


def fixture(fraction=0.5, count=4):
    config = SoftPadConfig(cells=(2, 2, 1))
    robot = newton.ModelBuilder()
    body = robot.add_body(mass=1.0)
    robot.add_shape_sphere(body, radius=0.01)
    builder = newton.ModelBuilder()
    pads = pad_world_count(count, fraction)
    builder.replicate(robot, count - pads)
    add_soft_pad(robot, config)
    builder.replicate(robot, pads)
    env = SimpleNamespace(num_envs=count, device=torch.device('cpu'), cfg={}, _sync=lambda: None,
                          solver=SimpleNamespace(), actions=torch.zeros(count, 14), commands=torch.zeros(count, 13),
                          episode_length_buf=torch.zeros(count), official=SimpleNamespace(common_step_counter=0))
    env.terrain = MixedTerrain(env, builder, config, fraction, sampling='fixed_groups_center')
    env.model = builder.finalize(device='cpu')
    env.terrain.finalize()
    env.states = [env.model.state(), env.model.state()]
    env.task_logic = SimpleNamespace(spawn=torch.ones(count, 21))
    return env, env.terrain


@pytest.mark.parametrize('fraction,expected', [(0, 0), (0.5, 2), (1, 4), (0.3, 1)])
def test_only_selected_worlds_contain_tets_and_reset_at_center(fraction, expected):
    env, terrain = fixture(fraction)
    assert env.model.tet_count == expected * 24
    expected_mask = torch.arange(4) >= 4 - expected
    torch.testing.assert_close(terrain.has_pad, expected_mask)
    for _ in range(5):
        env.task_logic.spawn.fill_(0.33)
        terrain.reset(torch.arange(4))
        terrain.adjust_spawn(torch.arange(4))
        torch.testing.assert_close(terrain.spawn_on_pad, expected_mask)
        assert (env.task_logic.spawn[expected_mask, :2] == 0).all()
        assert (env.task_logic.spawn[~expected_mask, :2] == 0.33).all()
        assert (terrain.origins[:, :2] == 0).all()
    heights = terrain.surface_heights(torch.tensor([[[0., 0., 0.2]]] * 4))
    torch.testing.assert_close(heights[:, 0], expected_mask.float() * terrain.config.top_z)
    assert not terrain.out_of_bounds(torch.zeros(4, 3)).any()
    assert terrain.out_of_bounds(torch.tensor([[3., 0., 0.]] * 4)).all()
    terrain.measure_volume()
    torch.testing.assert_close(terrain.minimum_volume, torch.ones(4))


def test_partial_reset_flat_world_does_not_write_minus_one_particle():
    env, terrain = fixture()
    for state in env.states:
        wp.to_torch(state.particle_q).add_(0.01)
        wp.to_torch(state.particle_qd).fill_(0.1)
    before = [wp.to_torch(s.particle_q).clone() for s in env.states]
    terrain.reset(torch.tensor([0, 1]))
    for s, expected in zip(env.states, before, strict=True):
        torch.testing.assert_close(wp.to_torch(s.particle_q), expected, atol=0, rtol=0)
    terrain.reset(torch.tensor([2]))
    for s, expected in zip(env.states, before, strict=True):
        q = wp.to_torch(s.particle_q)
        torch.testing.assert_close(q[terrain.pad_indices[3]], expected[terrain.pad_indices[3]], atol=0, rtol=0)
        torch.testing.assert_close(q[terrain.pad_indices[2]], terrain.rest[terrain.pad_indices[2]], atol=0, rtol=0)


def test_guard_restores_pad_world_without_touching_flat_bodies_or_neighbor_pad(tmp_path):
    env, terrain = fixture()
    guard = terrain.guard
    guard.diagnostic_dir = tmp_path
    guard.begin_step()
    for state in env.states:
        wp.to_torch(state.body_q)[:, 0] += 0.01
        wp.to_torch(state.particle_q)[terrain.pad_indices[3], 0] += 0.001
        wp.to_torch(state.particle_q)[terrain.pad_indices[2], 2] *= -1
    bodies = [wp.to_torch(s.body_q).clone() for s in env.states]
    neighbor = [wp.to_torch(s.particle_q)[terrain.pad_indices[3]].clone() for s in env.states]
    guard.after_substep(0)
    assert guard.failed.tolist() == [False, False, True, False]
    assert guard.invalid.tolist() == [False, False, True, False]
    for i, state in enumerate(env.states):
        torch.testing.assert_close(wp.to_torch(state.body_q)[[0, 1, 3]], bodies[i][[0, 1, 3]], atol=0, rtol=0)
        torch.testing.assert_close(wp.to_torch(state.particle_q)[terrain.pad_indices[3]], neighbor[i], atol=0, rtol=0)
        torch.testing.assert_close(wp.to_torch(state.particle_q)[terrain.pad_indices[2]],
                                   terrain.rest[terrain.pad_indices[2]], atol=0, rtol=0)
    assert len(list(tmp_path.glob('*.pt'))) == 1


def test_zero_pad_guard_and_single_world_rounding():
    assert pad_world_count(1, 0.5) == 1
    assert pad_world_count(3, 0.5) == 2
    env, terrain = fixture(0)
    terrain.guard.begin_step()
    terrain.guard.after_substep(0)
    terrain.guard.finish_step()
    assert not terrain.guard.failed.any()
    assert env.model.tet_count == 0
    np.testing.assert_array_equal(terrain.minimum_volume.numpy(), np.ones(4))


def test_sparse_pad_render_offsets_do_not_translate_physics():
    env, terrain = fixture()
    state = env.states[0]
    before = wp.to_torch(state.particle_q).clone()
    offsets = torch.tensor([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.], [2., 2., 0.]])
    display = terrain.display_state(SimpleNamespace(world_offsets=wp.from_torch(offsets, dtype=wp.vec3)), state)
    worlds = wp.to_torch(env.model.particle_world).long()
    torch.testing.assert_close(wp.to_torch(display.particle_q), before + offsets[worlds])
    torch.testing.assert_close(wp.to_torch(state.particle_q), before, atol=0, rtol=0)
