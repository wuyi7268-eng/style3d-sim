"""Soft-pad objectives must distinguish useful locomotion from stationary foot motion."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from examples.microduck.tests.test_locomotion_rewards import cadence_fixture
from examples.microduck import reward_functions as rewards
from examples.microduck.recipe import SNAPSHOT


def fixture(count=6):
    recipe, sensor, data, command, params = cadence_fixture(count)
    recipe.env = SimpleNamespace(mixed_terrain=True, terrain=SimpleNamespace(
        has_pad=torch.ones(count, dtype=torch.bool),
        config=SimpleNamespace(size=(3., 3., .12)),
    ))
    command[:, 0] = .1
    data.root_link_lin_vel_b.zero_()
    params['cadence']['soft_pad_motion'] = {
        'exploration_floor': .25, 'yaw_std': .35, 'lateral_std': .08,
    }
    return recipe, sensor, data, command, params


def test_pad_fine_tracking_prices_low_speed_stalling_and_preserves_signed_turns():
    recipe, _, data, command, _ = fixture()
    data.root_link_lin_vel_b[1, 0] = .1
    data.root_link_lin_vel_b[2, 0] = -.1
    data.root_link_lin_vel_b[3, :2] = torch.tensor([.1, .15])
    data.root_link_lin_vel_b[4:, 0] = .1
    data.root_link_ang_vel_b[4:, 2] = .5
    command[5, 2] = .5
    value = rewards.soft_pad_velocity_tracking(recipe, 'twist', .04, .5, .12, .35, .01, 35., .25)
    assert value[0] < .08
    assert value[1] == pytest.approx(1)
    assert value[2] < value[0]
    assert value[3] < .08
    assert value[4] < .2 * value[5]
    assert value[5] == pytest.approx(1)
    command[:, :2] *= -1
    data.root_link_lin_vel_b[:, :2] *= -1
    torch.testing.assert_close(rewards.soft_pad_velocity_tracking(recipe, 'twist', .04, .5, .12, .35, .01, 35., .25), value)


def test_pad_tracking_rewards_partial_progress_despite_large_yaw_error():
    recipe, _, data, command, _ = fixture(4)
    command[:, 0] = .35
    data.root_link_lin_vel_b[:, 0] = torch.tensor([0., .05, .10, .15])
    data.root_link_ang_vel_b[:, 2] = .8
    value = rewards.soft_pad_velocity_tracking(recipe, 'twist', .04, .5, .12, .35, .01, 35., .25)
    assert value[0] > .01
    assert (value[1:] > value[:-1]).all()
    assert value[-1] - value[1] > .006


def test_pad_objectives_ignore_flat_off_pad_idle_and_fallen_states():
    recipe, _, data, command, _ = fixture()
    data.root_link_lin_vel_b[:, 0] = .1
    recipe.env.terrain.has_pad[1] = False
    data.root_link_pos_w[2, 0] = 2
    command[3] = 0
    command[4] = torch.tensor([0., 0., .5])
    data.projected_gravity_b[5, 2] = 0
    value = rewards.soft_pad_velocity_tracking(recipe, 'twist', .04, .5, .12, .35, .01, 35., .25)
    torch.testing.assert_close(value, torch.tensor([1., 0., 0., 0., 0., 0.]))
    recipe.env.mixed_terrain = False
    assert not rewards.soft_pad_velocity_tracking(recipe, 'twist', .04, .5, .12, .35, .01, 35., .25).any()


def test_foot_shaping_retains_exploration_but_prefers_forward_progress_without_drift():
    recipe, sensor, data, command, params = fixture()
    command[:, 0] = .2
    data.root_link_lin_vel_b[1:, 0] = .2
    data.root_link_ang_vel_b[2, 2] = .7
    data.root_link_lin_vel_b[3, 1] = .2
    data.root_link_lin_vel_b[4, 0] = -.2
    recipe.env.terrain.has_pad[5] = False
    sensor.data.current_air_time.fill_(.2)
    original = deepcopy(params['cadence'])
    original.pop('soft_pad_motion')
    old = rewards.commanded_feet_air_time(recipe, 'feet_ground_contact', 'twist', original)
    new = rewards.commanded_feet_air_time(recipe, 'feet_ground_contact', 'twist', params['cadence'])
    assert new[0] == pytest.approx(old[0] * .25)
    assert new[1] == old[1]
    assert new[2] < new[1] * .3
    assert new[3] < new[1] * .3
    assert new[4] == pytest.approx(new[0])
    assert new[5] == old[5]
    # Pure yaw keeps its old shaping, including either turn direction.
    command[:, 0] = 0
    command[:, 2] = -.5
    torch.testing.assert_close(
        rewards.commanded_feet_air_time(recipe, 'feet_ground_contact', 'twist', original),
        rewards.commanded_feet_air_time(recipe, 'feet_ground_contact', 'twist', params['cadence']),
    )


def test_clearance_and_swing_peak_use_the_same_progress_gate():
    recipe, sensor, data, command, params = fixture(3)
    command[:, 0] = .2
    data.root_link_lin_vel_b[1:, 0] = .2
    data.site_lin_vel_w[:, 1:3, 0] = .2
    recipe.env.terrain.has_pad[2] = False
    height = recipe.scene['foot_height_scan'].data.heights
    height.fill_(.037)
    cadence = params['cadence']
    kwargs = dict(target_height=.03, height_speed_gain=.035, max_height=.05,
                  height_sensor_name='foot_height_scan', command_name='twist', command_threshold=.01,
                  cadence=cadence)
    clearance = rewards.commanded_foot_clearance(recipe, **kwargs)
    assert clearance[0] == pytest.approx(clearance[1] * .25)
    assert clearance[2] == clearance[1]
    peak = rewards.CommandedFootSwingHeight(recipe)
    sensor.update(torch.zeros(3, 2), recipe.step_dt)
    peak(recipe, sensor_name='feet_ground_contact', height_std=.015, **kwargs)
    sensor.update(torch.ones(3, 2), recipe.step_dt)
    value = peak(recipe, sensor_name='feet_ground_contact', height_std=.015, **kwargs)
    assert value[0] == pytest.approx(value[1] * .25)
    assert value[2] == value[1]
    assert not peak.peak_heights.any()


def test_config_scope_and_cpu_environment_dispatch():
    from examples.microduck.env import MicroDuckEnv

    tasks = json.loads(SNAPSHOT.read_text())['tasks']
    for name, task in tasks.items():
        assert ('soft_pad_velocity' in task['rewards']) == (name == 'velocity')
        if name != 'velocity':
            assert all('soft_pad_motion' not in r['params'].get('cadence', {}) for r in task['rewards'].values())
    env = MicroDuckEnv(solver='mujoco', num_envs=1, device='cpu', task='velocity')
    try:
        _, reward, _, extras = env.step(torch.zeros(1, 14))
        assert torch.isfinite(reward).all()
        assert extras['log']['Task/soft_pad_velocity'] == 0
        assert env.official.supported['soft_pad_velocity'] == 'soft_pad_velocity_tracking'
        assert env.cfg['official_alignment']['local_rewards']['soft_pad_velocity']['local']
    finally:
        env.close()
