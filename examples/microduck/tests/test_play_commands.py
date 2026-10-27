"""Playback must preserve task padding and commands across automatic resets."""

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from examples.microduck.play import update_playback_observation, update_velocity_command


def test_ballkick_playback_preserves_trained_zero_padding():
    snapshot = json.loads((Path(__file__).parents[1] / 'configs/official_snapshot.json').read_text())
    env = SimpleNamespace(
        commands=torch.ones(1, 13), device='cpu',
        official=SimpleNamespace(observations=SimpleNamespace(spec=snapshot['tasks']['ballkick']['observations'])),
    )
    obs = torch.ones(1, 61)
    update_playback_observation(env, obs)
    torch.testing.assert_close(obs[:, 51:], torch.zeros(1, 10))
    torch.testing.assert_close(obs[:, :48], torch.ones(1, 48))
    torch.testing.assert_close(obs[:, 48:51], env.commands[:, :3])


def test_velocity_command_is_restored_after_reset():
    env = SimpleNamespace(commands=torch.ones(1, 13), device='cpu')
    command = [0.2, 0.0, 0.0]
    update_velocity_command(env, None, command, previous_command=command)
    torch.testing.assert_close(env.commands[:, :3], torch.tensor([command]))
