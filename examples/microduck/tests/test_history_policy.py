"""Past-frame timing, independent encoders, and real PPO/checkpoint integration on CPU."""

import copy
import json

import pytest
import torch

from examples.microduck.checkpoint import checkpoint_policy_variant
from examples.microduck.common import setup
from examples.microduck.observations import (
    CONTRACT,
    HISTORY_SPEC,
    ObservationHistory,
    configure_policy_observations,
)
from examples.microduck.policy import HistoryActorCritic, HistoryNormalization
from examples.microduck.ppo import create_runner, microduck_symmetry, ppo_config
from examples.microduck.recipe import RECIPE, recipe_fingerprint


def test_history_excludes_current_and_retains_ten_past_frames():
    history = ObservationHistory(2, "cpu")
    actor = torch.zeros(2, 61)
    actor[1] = 100
    first = history.read(actor).clone()
    assert first.shape == (2, 480)
    torch.testing.assert_close(first.reshape(2, 10, 48), actor[:, None, :48].expand(-1, 10, -1))
    for step in range(1, 13):
        history.advance()
        actor += 1
        result = history.read(actor).clone()
        expected = torch.tensor([max(0, step - 1 - lag) for lag in range(10)], dtype=torch.float)
        torch.testing.assert_close(result[0].reshape(10, 48), expected[:, None].expand(-1, 48))
        torch.testing.assert_close(history.read(actor), result)
        torch.testing.assert_close(result[1], result[0] + 100)


def test_partial_reset_fills_only_selected_environment():
    history = ObservationHistory(3, "cpu")
    actor = torch.randn(3, 61)
    history.read(actor)
    history.advance()
    actor += 1
    before = history.read(actor).clone()
    history.reset(torch.tensor([1]))
    actor[1] = -5
    after = history.read(actor)
    torch.testing.assert_close(after[[0, 2]], before[[0, 2]])
    torch.testing.assert_close(after[1], torch.full((480,), -5.0))
    history.advance()
    actor += 1
    torch.testing.assert_close(history.read(actor)[1], torch.full((480,), -5.0))


def test_encoders_are_independent_and_receive_their_own_gradients():
    policy = HistoryActorCritic(541, 550, 14)
    assert policy.actor[0].in_features == 125
    assert policy.critic[0].in_features == 134
    assert policy.actor_history_encoder[0].in_features == 480
    assert policy.critic_history_encoder[-1].out_features == 64
    actor_parameters = list(policy.actor_history_encoder.parameters())
    critic_parameters = list(policy.critic_history_encoder.parameters())
    assert not {id(p) for p in actor_parameters} & {id(p) for p in critic_parameters}
    actor_obs, critic_obs = torch.randn(3, 541), torch.randn(3, 550)
    assert policy.act(actor_obs).shape == (3, 14)
    assert policy.act_inference(actor_obs).shape == (3, 14)
    policy.act_inference(actor_obs).square().mean().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in actor_parameters)
    assert all(p.grad is None for p in critic_parameters)
    policy.zero_grad(set_to_none=True)
    assert policy.evaluate(critic_obs).shape == (3, 1)
    policy.evaluate(critic_obs).square().mean().backward()
    assert all(p.grad is None for p in actor_parameters)
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in critic_parameters)


def test_history_normalization_reuses_actor_stats_without_counting_past_frames():
    actor_norm = HistoryNormalization(61)
    critic_norm = HistoryNormalization(70, actor_norm)
    current = torch.randn(4, 61) * 2 + 3
    past = torch.randn(4, 10, 48) * 5 - 10
    actor = actor_norm(torch.cat((current, past.flatten(1)), -1))
    stats = actor_norm.current
    expected = (past - stats.mean[:48]) / (stats.std[:48] + stats.eps)
    torch.testing.assert_close(actor[:, 61:].reshape(4, 10, 48), expected)
    assert stats.count == 4
    torch.testing.assert_close(stats.mean, current.mean(0))
    critic = critic_norm(torch.cat((torch.randn(4, 70) + 50, past.flatten(1)), -1))
    torch.testing.assert_close(critic[:, 70:], actor[:, 61:])
    assert stats.count == critic_norm.current.count == 4
    actor_norm.eval()
    critic_norm.eval()
    actor_norm(torch.randn(4, 541))
    critic_norm(torch.randn(4, 550))
    assert stats.count == critic_norm.current.count == 4


def test_symmetry_mirrors_each_past_frame_without_reordering_time():
    obs = torch.randn(3, 541)
    mirrored, _ = microduck_symmetry(None, obs)
    twice, _ = microduck_symmetry(None, mirrored[3:])
    torch.testing.assert_close(twice[3:], obs)
    frames = obs[:, 61:].reshape(-1, 48)
    padded = torch.cat((frames, torch.zeros(len(frames), 13)), -1)
    expected, _ = microduck_symmetry(None, padded)
    torch.testing.assert_close(mirrored[3:, 61:].reshape(-1, 48), expected[len(frames):, :48])


class HistoryEnv:
    num_envs, num_actions, num_privileged_obs, device = 4, 14, 550, "cpu"
    max_episode_length = 100

    def __init__(self):
        self.cfg = {}
        self.episode_length_buf = torch.zeros(4, dtype=torch.long)
        self.current = torch.randn(4, 61)
        self.history = ObservationHistory(4, "cpu")

    def get_observations(self):
        history = self.history.read(self.current)
        actor = torch.cat((self.current, history), -1)
        critic = torch.cat((self.current, torch.ones(4, 9), history), -1)
        return actor, {"observations": {"critic": critic}}

    def step(self, actions):
        self.history.advance()
        self.current = torch.randn(4, 61)
        actor, extras = self.get_observations()
        return actor, actions.square().sum(-1), torch.zeros(4, dtype=torch.long), extras


def test_ppo_updates_both_encoders_and_restores_checkpoint(tmp_path):
    cfg = ppo_config(4, "history")
    cfg["num_steps_per_env"] = 2
    cfg["policy"].update(actor_hidden_dims=[16], critic_hidden_dims=[16])
    cfg["algorithm"].update(num_learning_epochs=1, num_mini_batches=1)
    env = HistoryEnv()
    runner = create_runner(env, cfg, str(tmp_path), "cpu")
    policy = runner.alg.policy
    actor_before = policy.actor_history_encoder[0].weight.detach().clone()
    critic_before = policy.critic_history_encoder[0].weight.detach().clone()
    runner.learn(1)
    assert runner.alg.storage.observations.shape == (2, 4, 541)
    assert runner.alg.storage.privileged_observations.shape == (2, 4, 550)
    assert not torch.equal(actor_before, policy.actor_history_encoder[0].weight)
    assert not torch.equal(critic_before, policy.critic_history_encoder[0].weight)
    observation, extras = env.get_observations()
    inference = runner.get_inference_policy()
    expected = inference(observation)
    expected_value = policy.evaluate(runner.privileged_obs_normalizer(extras["observations"]["critic"]))
    restored = create_runner(HistoryEnv(), cfg, device="cpu")
    restored.load(str(tmp_path / "model_0.pt"))
    torch.testing.assert_close(restored.get_inference_policy()(observation), expected)
    torch.testing.assert_close(
        restored.alg.policy.evaluate(restored.privileged_obs_normalizer(extras["observations"]["critic"])),
        expected_value,
    )
    assert restored.alg.optimizer.state
    for name, value in runner.obs_normalizer.state_dict().items():
        torch.testing.assert_close(restored.obs_normalizer.state_dict()[name], value)
    # The critic still borrows the restored actor statistics, rather than stale/copy statistics.
    assert restored.privileged_obs_normalizer._actor_stats() is restored.obs_normalizer.current


def test_cli_selects_variant_and_checkpoint_rejects_mismatches(tmp_path):
    assert setup("train", [])[0].policy_variant == "standard"
    assert setup("train", ["--policy-variant", "history"])[0].policy_variant == "history"
    path = tmp_path / "model_0.pt"
    path.touch()
    metadata = {
        "policy_variant": "history", "history": HISTORY_SPEC,
        "observation_contract": CONTRACT, "solver": "mujoco", "recipe": RECIPE, "task": "velocity",
        "ppo": ppo_config(4, "history"), "official_alignment": {"recipe_fingerprint": recipe_fingerprint()},
    }
    (tmp_path / "config.json").write_text(json.dumps(metadata))
    assert setup("train", ["--resume", str(path)])[0].policy_variant == "history"
    assert setup("play", ["--checkpoint", str(path)])[0].policy_variant == "history"
    with pytest.raises(SystemExit):
        setup("train", ["--resume", str(path), "--policy-variant", "standard"])
    invalid = copy.deepcopy(metadata)
    invalid["history"]["length"] = 9
    with pytest.raises(ValueError, match="history"):
        checkpoint_policy_variant(invalid)
    invalid = copy.deepcopy(metadata)
    del invalid["policy_variant"]
    with pytest.raises(ValueError, match="policy class"):
        checkpoint_policy_variant(invalid)


def test_real_observations_fast_playback_and_reset_use_same_history():
    from examples.microduck.env import MicroDuckEnv
    from examples.microduck.play import PolicyPlayback, update_playback_observation

    env = MicroDuckEnv(device="cpu", num_envs=1)
    try:
        configure_policy_observations(env, "history")
        assert (env.num_obs, env.num_privileged_obs) == (541, 550)
        actor, extras = env.get_observations()
        torch.testing.assert_close(actor[:, 61:], extras["observations"]["critic"][:, 70:])
        torch.testing.assert_close(actor[:, 61:].reshape(1, 10, 48), actor[:, None, :48].expand(-1, 10, -1))
        frozen = actor.clone()
        update_playback_observation(env, actor)
        torch.testing.assert_close(actor[:, 61:], frozen[:, 61:])
        torch.testing.assert_close(env.get_observations()[0][:, 61:], frozen[:, 61:])
        actor, _, _, _ = env.step(torch.zeros(1, 14))
        torch.testing.assert_close(actor[:, 61:109], frozen[:, :48])
        env._reset(torch.tensor([0]))
        reset, _ = env.get_observations()
        torch.testing.assert_close(reset[0, 61:].reshape(10, 48), reset[0, None, :48].expand(10, -1))
        torch.testing.assert_close(PolicyPlayback(env).observe(), reset)
        previous = reset.clone()
        next_actor = PolicyPlayback(env).step(torch.zeros(1, 14))
        torch.testing.assert_close(next_actor[:, 61:109], previous[:, :48])
    finally:
        env.close()


def test_history_preserves_rewards_commands_curricula_and_terminations():
    from examples.microduck.env import MicroDuckEnv

    def rollout(variant):
        env = MicroDuckEnv(device="cpu", num_envs=1, seed=123)
        records = []
        try:
            configure_policy_observations(env, variant)
            # Exercise auto-reset and later curricula without changing production settings.
            env.max_episode_length = 12
            for stage in (0, 14401, 48001):
                env.official.common_step_counter = stage
                env.reset()
                for step in range(16):
                    if step == 3:
                        for timer in env.official.pushes.timers.values():
                            timer.zero_()
                    actions = torch.sin(torch.arange(14).float() + step)[None] * 0.05
                    obs, reward, done, extras = env.step(actions)
                    assert env.official.common_step_counter == stage + step + 1
                    terms = {key: torch.as_tensor(value).clone() for key, value in extras["log"].items()}
                    weighted_sum = sum(value for key, value in terms.items() if key.startswith("Task/"))
                    torch.testing.assert_close(reward[0], weighted_sum * env.step_dt)
                    records.append({
                        "actor": obs[:, :61].clone(),
                        "critic": extras["observations"]["critic"][:, :70].clone(),
                        "reward": reward.clone(),
                        "done": done.clone(),
                        "timeout": extras["time_outs"].clone(),
                        "commands": env.commands.clone(),
                        "episode_length": env.episode_length_buf.clone(),
                        **terms,
                    })
            assert env.total_resets > 0
            return records, env.official.spec
        finally:
            env.close()

    standard, standard_spec = rollout("standard")
    history, history_spec = rollout("history")
    assert standard_spec == history_spec
    for original, augmented in zip(standard, history, strict=True):
        assert original.keys() == augmented.keys()
        for name in original:
            torch.testing.assert_close(original[name], augmented[name], rtol=0, atol=0, msg=name)
