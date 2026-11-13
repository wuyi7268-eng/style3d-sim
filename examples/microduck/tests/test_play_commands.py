"""Playback keeps commands and history coherent across manual resets only."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from examples.microduck.common import setup
from examples.microduck.play import (
    manual_reset_requested,
    update_playback_observation,
    update_velocity_command,
)


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


def test_playback_can_disable_fall_resets():
    args, _, _ = setup("play", ["--viewer", "null"])
    assert args.fall_reset is False


def test_legacy_flag_cannot_reenable_automatic_playback_resets():
    args, _, _ = setup("play", ["--viewer", "null", "--fall-reset"])
    assert args.fall_reset is False


def test_manual_reset_key_is_edge_triggered_and_works_while_paused():
    pressed = False
    viewer = SimpleNamespace(is_key_down=lambda key: pressed and key == "r", is_paused=lambda: True)
    keys = {}
    assert not manual_reset_requested(viewer, keys)
    pressed = True
    assert manual_reset_requested(viewer, keys)
    assert not manual_reset_requested(viewer, keys)
    pressed = False
    assert not manual_reset_requested(viewer, keys)
    pressed = True
    assert manual_reset_requested(viewer, keys)
    assert not manual_reset_requested(None, keys)


@pytest.mark.parametrize("manual_command", [False, True])
def test_checkpoint_loop_resets_all_worlds_while_paused_and_refreshes_observation(monkeypatch, manual_command):
    from examples.microduck import play, ppo

    # Hold R for two paused frames, release it, then resume. Pausing must not consume the replay budget.
    frames = iter([(True, True), (True, True), (False, True), (False, False), (False, False)])
    current = [False, False]
    resets, inferred = [], []
    observation = torch.zeros(3, 61)

    def running():
        value = next(frames, None)
        if value is None:
            return False
        current[:] = value
        return True

    viewer = SimpleNamespace(
        is_running=running, is_key_down=lambda key: current[0] and key == "r", is_paused=lambda: current[1]
    )

    def reset():
        resets.append(True)
        observation.fill_(len(resets))

    def policy(obs):
        inferred.append(obs.clone())
        return torch.zeros(3, 14)

    env = SimpleNamespace(
        mixed_terrain=False, terrain=SimpleNamespace(kind="default"), official=None, viewer=viewer, family="standup", reset=reset,
        get_observations=lambda: (observation.clone(), {}), render=lambda: None, close=lambda: None,
        cfg={"policy_dt": 0.02}, graph=None, total_resets=0,
    )
    runner = SimpleNamespace(load=lambda *args, **kwargs: None, get_inference_policy=lambda **kwargs: policy)
    args = SimpleNamespace(
        policy=None, checkpoint="model.pt", device="cpu", num_frames=2, benchmark=True,
        diagnostics=False, command=[0, 0, 0], keep_open=False, record_dir=None, record_until_pad_edge=False,
        manual_command=manual_command,
    )
    # A checkpoint initializes once; the additional reset must originate from the R key.
    env.official = SimpleNamespace(
        common_step_counter=0, bridge_path=None, observations=SimpleNamespace(spec={"actor": {"enable_corruption": True}})
    )
    if manual_command:
        # A sentinel would fail if either the reset path or inference loop retained automatic feedback.
        env.terrain.kind = "elastic_bridge"
        env.official.bridge_path = SimpleNamespace(config={"spawn_distance_m": .15})
    runner.current_learning_iteration = 0
    monkeypatch.setattr(play, "setup", lambda *_: (args, None, {"num_steps_per_env": 24}))
    monkeypatch.setattr(play, "create_env", lambda args: env)
    monkeypatch.setattr(ppo, "create_runner", lambda *args, **kwargs: runner)
    monkeypatch.setattr(play, "PolicyPlayback", lambda env: SimpleNamespace(step=lambda actions: observation.clone()))
    monkeypatch.setattr(play, "update_playback_observation", lambda *args: None)
    monkeypatch.setattr(play.time, "sleep", lambda seconds: None)
    assert play.main([]) == 0
    if manual_command:
        assert env.official.bridge_path is None
        assert env.terrain.playback_center_spawn
    assert len(resets) == 2
    assert len(inferred) == 2
    assert all(torch.equal(obs, torch.full((3, 61), 2.0)) for obs in inferred)


@pytest.mark.parametrize("task", ["velocity", "velstand", "groundpick", "ballkick", "standup"])
def test_playback_ignores_termination_timeout_and_bounds_but_training_resets(task, monkeypatch):
    from examples.microduck import events
    from examples.microduck.env import MicroDuckEnv
    from examples.microduck.observations import configure_policy_observations

    env = MicroDuckEnv(solver="mujoco", task=task, num_envs=1, device="cpu", seed=42, reset_on_fall=True)
    try:
        configure_policy_observations(env, "history")
        env.max_episode_length = 1
        monkeypatch.setattr(events, "termination", lambda *_: torch.ones(1, dtype=torch.bool))
        monkeypatch.setattr(env.terrain, "out_of_bounds", lambda *_: torch.ones(1, dtype=torch.bool))
        actions = torch.full((1, 14), 0.01)
        for _ in range(3):
            assert torch.isfinite(env.step_playback(actions)).all()
        assert env.total_resets == 0
        assert env.episode_length_buf.tolist() == [3]
        obs, _ = env.reset()
        assert env.episode_length_buf.tolist() == [0]
        assert not env.actions.any()
        torch.testing.assert_close(obs[:, 61:].reshape(1, 10, 48), obs[:, None, :48].expand(-1, 10, -1))
        _, _, done, _ = env.step(actions, apply_interval_events=False)
        assert done.all()
        assert env.total_resets == 1
        assert env.episode_length_buf.tolist() == [0]
    finally:
        env.close()
