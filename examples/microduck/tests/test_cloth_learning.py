"""Cloth locomotion must use the moving support, including actual soft contacts."""

from types import SimpleNamespace
from copy import deepcopy
import json

import newton
import numpy as np
import pytest
import torch
import warp as wp

from examples.microduck.scenes import SceneTerrain, add_scene
from examples.microduck import reward_functions as rewards
from examples.microduck.tests.test_locomotion_rewards import cadence_fixture
from examples.microduck.recipe import SNAPSHOT


def test_cloth_clearance_tracks_deformation_and_world_reset():
    template = newton.ModelBuilder()
    metadata = add_scene(template, "cloth")
    builder = newton.ModelBuilder()
    newton.solvers.SolverStyle3D.register_custom_attributes(builder)
    builder.replicate(template, 2)
    model = builder.finalize(device="cpu")
    env = SimpleNamespace(model=model, num_envs=2, device=torch.device("cpu"), cfg={}, _sync=lambda: None)
    env.states = [model.state(), model.state()]
    terrain = SceneTerrain(env, "cloth", metadata)
    terrain.finalize()
    points = torch.tensor([[[0., 0., 1.]], [[0., 0., 1.]]])
    initial = terrain.heights(points).clone()
    selected = terrain.cloth_indices[terrain.cloth_worlds == 0]
    for state in env.states:
        wp.to_torch(state.particle_q)[selected, 2] -= .05
    changed = terrain.heights(points)
    torch.testing.assert_close(changed, initial + torch.tensor([[.05], [0.]]))
    terrain.reset(torch.tensor([0]))
    torch.testing.assert_close(terrain.heights(points), initial)
    mask = terrain.pad_particle.numpy().astype(bool)
    np.testing.assert_array_equal(np.flatnonzero(mask), terrain.cloth_indices.numpy())


def cloth_fixture(count=6):
    recipe, sensor, data, command, params = cadence_fixture(count)
    recipe.env = SimpleNamespace(mixed_terrain=False, terrain=SimpleNamespace(
        kind="cloth", out_of_bounds=lambda p: (p[:, :2].abs() > .56).any(-1),
    ))
    command[:, 0] = .1
    data.root_link_lin_vel_b.zero_()
    sensor.data.soft_found = torch.ones(count, 2)
    sensor.data.soft_slip_squared_sum = torch.zeros(count, 2)
    sensor.data.found.fill_(1)
    recipe.scene['robot_ground_contact'] = SimpleNamespace(data=SimpleNamespace(found=torch.full((count, 1), 2.)))
    params['cadence']['cloth_motion'] = {'exploration_floor': .25, 'yaw_std': .35, 'lateral_std': .08}
    return recipe, sensor, data, command, params


def tracking(recipe):
    return rewards.cloth_velocity_tracking(recipe, 'twist', .04, .5, .12, .35, .01, 35., .09, .25)


def test_cloth_tracking_rewards_progress_and_rejects_drift_falls_and_exits():
    recipe, _, data, command, _ = cloth_fixture()
    data.root_link_lin_vel_b[1:, 0] = .1
    data.root_link_ang_vel_b[2:4, 2] = .5
    command[3, 2] = .5
    data.projected_gravity_b[4, 2] = 0
    data.root_link_pos_w[5, 0] = .6
    value = tracking(recipe)
    assert 0 < value[0] < .08
    assert value[1] == pytest.approx(1)
    assert value[2] < value[3] * .2
    assert value[3] == pytest.approx(1)
    assert not value[4:].any()
    command *= -1
    data.root_link_lin_vel_b *= -1
    data.root_link_ang_vel_b *= -1
    torch.testing.assert_close(tracking(recipe), value)
    command.zero_()
    assert not tracking(recipe).any()


def test_cloth_slip_is_relative_bounded_and_airborne_has_no_cost():
    recipe, sensor, data, _, _ = cloth_fixture(3)
    data.site_lin_vel_w[:, 1:3, 0] = 1.  # Cloth co-motion must not be charged as slip.
    sensor.data.soft_slip_squared_sum[1:] = .15**2
    sensor.data.soft_found[2] = 0
    sensor.data.soft_slip_squared_sum[2] = 0
    value = rewards.cloth_contact_slip(recipe, 'feet_ground_contact', .15)
    assert value[0] == 0 and value[2] == 0
    assert value[1] == pytest.approx(1 - np.exp(-1))
    recipe.scene['robot_ground_contact'].data.found[1] = 3
    torch.testing.assert_close(rewards.cloth_nonfoot_contact(recipe, 'feet_ground_contact', 'robot_ground_contact'),
                               torch.tensor([0., 1., 0.]))


def test_cloth_progress_rewards_command_direction_and_prices_stalling():
    recipe, _, data, command, _ = cloth_fixture(4)
    command[:, 0] = .2
    data.root_link_lin_vel_b[:, 0] = torch.tensor([0., .1, .2, -.1])
    progress = rewards.cloth_command_progress(recipe, 'twist', .03, 35., .09, .10, .35)
    stall = rewards.cloth_stall_penalty(recipe, 'twist', .03, 35., .09, .10, .35)
    assert progress[0] == pytest.approx(0.)
    assert progress[1] < progress[2]
    assert progress[3] == pytest.approx(0.)
    assert stall[0] > stall[1] > stall[2]
    command[:, 0] = 0
    command[:, 2] = .5
    assert not rewards.cloth_command_progress(recipe, 'twist', .03, 35., .09, .10, .35).any()
    assert not rewards.cloth_stall_penalty(recipe, 'twist', .03, 35., .09, .10, .35).any()


@pytest.mark.parametrize('kind', ['default', 'mixed_soft_pad', 'trampoline'])
def test_new_rewards_are_zero_outside_cloth(kind):
    recipe, _, _, _, _ = cloth_fixture()
    recipe.env.terrain.kind = kind
    assert not tracking(recipe).any()
    assert not rewards.cloth_contact_slip(recipe, 'feet_ground_contact', .15).any()
    assert not rewards.cloth_nonfoot_contact(recipe, 'feet_ground_contact', 'robot_ground_contact').any()


def test_cloth_foot_shaping_prefers_useful_motion_and_keeps_exploration():
    recipe, sensor, data, command, params = cloth_fixture(3)
    sensor.data.current_air_time.fill_(.2)
    data.root_link_lin_vel_b[1, 0] = .1
    data.root_link_lin_vel_b[2, 0] = -.1
    value = rewards.commanded_feet_air_time(recipe, 'feet_ground_contact', 'twist', params['cadence'])
    assert value[1] > 0
    assert value[0] == pytest.approx(value[1] * .25)
    assert value[2] == pytest.approx(value[0])
    command[:, 0] = 0
    command[:, 2] = -.5
    torch.testing.assert_close(rewards._soft_pad_motion_gate(recipe, command, params['cadence']), torch.ones(3))


@pytest.mark.parametrize('mixed', [False, True])
def test_cloth_options_leave_flat_and_pad_shaping_identical(mixed):
    from examples.microduck.tests.test_soft_pad_learning import fixture

    recipe, _, data, command, params = fixture()
    recipe.env.mixed_terrain = mixed
    data.root_link_lin_vel_b[:, 0] = torch.linspace(-.1, .2, 6)
    data.root_link_ang_vel_b[:, 2] = torch.linspace(-.5, .5, 6)
    old = deepcopy(params['cadence'])
    old.pop('cloth_motion', None)
    new = deepcopy(old)
    new['cloth_motion'] = {'exploration_floor': .25, 'yaw_std': .35, 'lateral_std': .08}
    torch.testing.assert_close(rewards._soft_pad_motion_gate(recipe, command, old),
                               rewards._soft_pad_motion_gate(recipe, command, new), rtol=0, atol=0)


def test_cloth_reward_configuration_is_velocity_only():
    tasks = json.loads(SNAPSHOT.read_text())['tasks']
    for name, task in tasks.items():
        for key in ('cloth_velocity', 'cloth_command_progress', 'cloth_stall',
                    'cloth_contact_slip', 'cloth_nonfoot_contact'):
            assert (key in task['rewards']) == (name == 'velocity')
    assert tasks['velocity']['rewards']['gait_cadence']['weight'] == 0
    assert tasks['velocity']['rewards']['gait_overdue']['weight'] == 0


def test_cloth_training_viewer_keeps_command_resampling_but_playback_does_not():
    from examples.microduck.env import MicroDuckEnv

    env = MicroDuckEnv(solver='mujoco', num_envs=1, device='cpu', task='velocity')
    try:
        # Use the CPU dynamics to isolate the viewer branch without a GL context.
        env.terrain.kind = 'cloth'
        env.terrain.pad_particle = wp.zeros(env.model.particle_count, dtype=int, device='cpu')
        env.viewer = SimpleNamespace(close=lambda: None)
        env.render = lambda: None
        calls = []
        env.official.update_commands = lambda: calls.append(True)
        env.step(torch.zeros(1, 14))
        assert len(calls) == 1
        env.step(torch.zeros(1, 14), auto_reset=False)
        assert len(calls) == 1
    finally:
        env.close()
