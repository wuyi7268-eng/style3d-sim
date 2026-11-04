"""A deterministic training diagnostic must never sample actions or update weights."""
from types import SimpleNamespace

import pytest
import torch

from examples.microduck.common import setup
from examples.microduck.play import HEAD_UP_COMMAND
from examples.microduck.train import run_deterministic_diagnostic


def test_diagnostic_uses_training_step_and_restores_commands_after_reset():
    env = SimpleNamespace(
        viewer=None, device='cpu', commands=torch.zeros(2, 13), total_resets=0,
        official=SimpleNamespace(common_step_counter=0, observations=SimpleNamespace(spec={
            'actor': {'enable_corruption': True, 'terms': {
                'command': {'func': 'generated_commands'},
                'head_command': {'func': 'generated_commands'},
                'body_command': {'func': 'mdp.zero_command_padding'},
            }},
        })),
    )
    observation = torch.zeros(2, 61)
    calls = []

    def reset():
        assert env.official.common_step_counter == 101 * 24
        assert not env.official.observations.spec['actor']['enable_corruption']

    def step(actions, *, apply_interval_events, apply_viewer_forces):
        assert not apply_interval_events and apply_viewer_forces
        torch.testing.assert_close(actions, torch.full((2, 14), 0.25))
        calls.append(actions.clone())
        # Simulate a reset replacing commands and observations between frames.
        env.commands.fill_(3)
        observation.fill_(5)
        env.total_resets += 1
        return observation, None, None, None

    def load(path, *, load_optimizer):
        assert path == 'model.pt' and load_optimizer is False

    def inference(*, device):
        assert device == 'cpu'

        def policy(obs):
            assert torch.is_inference_mode_enabled()
            torch.testing.assert_close(obs[:, 48:51], torch.tensor([[0.3, 0., 0.]] * 2))
            torch.testing.assert_close(obs[:, 51:55], torch.tensor([HEAD_UP_COMMAND] * 2))
            assert not obs[:, 55:61].any()
            return torch.full((2, 14), 0.25)
        return policy

    env.reset, env.step = reset, step
    env.get_observations = lambda: (observation, {})
    # No learn/optimizer or act(sample) method: any use fails the test.
    runner = SimpleNamespace(load=load, current_learning_iteration=100, get_inference_policy=inference)
    args = SimpleNamespace(init_from='model.pt', resume=None, device='cpu', command=[0.3, 0., 0.], num_frames=3)
    run_deterministic_diagnostic(env, runner, args, 24)
    assert len(calls) == 3


def test_diagnostic_requires_a_checkpoint():
    with pytest.raises(SystemExit):
        setup('train', ['--diagnostic-no-exploration'])


def test_normal_training_does_not_enable_diagnostic():
    args, _, _ = setup('train', [])
    assert not args.diagnostic_no_exploration
