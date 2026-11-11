"""Basketball blind/recurrent contracts, physical reset isolation and failure gates."""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp

from examples.microduck.basketball import BasketballEnv, keyboard_command, load_config, runner_config, validate_config
from examples.microduck.basketball_policy import BasketballOnnxPolicy, pretrained_policy
from examples.microduck.basketball_task import advance_hold, task_terms


def states(count=1):
    root = torch.tensor([[0, 0, 0.365]]).repeat(count, 1)
    ball = torch.tensor([[0, 0, 0.12]]).repeat(count, 1)
    gravity = torch.tensor([[0, 0, -1.0]]).repeat(count, 1)
    zeros = torch.zeros(count, 3)
    feet = ball[:, None, :].repeat(1, 2, 1)
    feet[:, :, 2] += 0.132
    return root, gravity, zeros.clone(), zeros.clone(), ball, zeros.clone(), feet, zeros.clone()


def test_reward_optimum_and_fall_gates():
    config = load_config()
    data = states(4)
    terms, failed, _ = task_terms(config, *data)
    for name in ("centered", "height", "feet_on_ball", "upright", "track_linear", "track_angular"):
        torch.testing.assert_close(terms[name], torch.ones(4))
    assert not failed.any()
    data[0][1, 0] = 0.161
    data[0][2, 2] = 0.289
    data[1][3] = torch.tensor([1.0, 0, 0])
    terms, failed, _ = task_terms(config, *data)
    assert failed.tolist() == [False, True, True, True]
    assert terms["centered"][1] < 1e-6
    assert terms["height"][2] < 0.01


def test_feet_reward_requires_both_feet_and_tracks_moving_ball():
    config = load_config()
    data = states()
    data[6][:, 0, 2] += 0.1
    terms, _, _ = task_terms(config, *data)
    assert terms["feet_on_ball"].item() == pytest.approx(0.5)
    translated = [value.clone() for value in data]
    for index in (0, 4, 6):
        translated[index][..., :2] += torch.tensor([1.2, -0.3])
    moved, _, _ = task_terms(config, *translated)
    for name in terms:
        torch.testing.assert_close(moved[name], terms[name])


def test_curriculum_promotes_demotes_and_stops_at_free_ball():
    hold = load_config()["hold"]
    levels = torch.tensor([0, 0, 3, 3, 5])
    duration = torch.tensor([0.2, 6.0, 1.0, 3.0, 10.0])
    assert advance_hold(levels, duration, hold).tolist() == [0, 1, 2, 3, 5]


@pytest.mark.parametrize("section,key,value", [
    ("geometry", "radius", -1), ("hold", "levels", [1, 0.1]),
    ("hold", "levels", [1, 0, 0.5]), ("noise", "position", float("nan")),
    ("commands", "limits", [1, 2]), ("physics", "decimation", 9),
])
def test_invalid_config(section, key, value):
    config = load_config()
    config[section][key] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_lstm_training_contract():
    config = runner_config(load_config(), 2)
    assert config["policy"]["class_name"] == "ActorCriticRecurrent"
    assert config["policy"]["rnn_hidden_dim"] == 256
    assert config["policy"]["rnn_type"] == "lstm"
    assert config["empirical_normalization"]
    assert config["algorithm"]["num_mini_batches"] == 2


def test_keyboard_commands_are_bounded_and_edge_triggered():
    keys = {"i", "u"}
    viewer = SimpleNamespace(is_key_down=lambda key: key in keys)
    previous = {}
    command = keyboard_command(viewer, [0.14, 0, 0.48], previous, [0.15, 0.1, 0.5])
    assert command == [0.15, 0, 0.5]
    assert keyboard_command(viewer, command, previous, [0.15, 0.1, 0.5]) == command
    keys.clear()
    keys.add("x")
    assert keyboard_command(viewer, command, previous, [0.15, 0.1, 0.5]) == [0, 0, 0]


def test_pretrained_rejects_wrong_file(tmp_path):
    path = tmp_path / "policy.onnx"
    path.write_bytes(b"not the published basketball model")
    with pytest.raises(ValueError, match="checksum"):
        pretrained_policy(path)


def test_trainable_b11_actor_matches_release_through_memory_resets():
    from rsl_rl.modules import ActorCriticRecurrent, EmpiricalNormalization

    from examples.microduck.basketball_policy import CACHE, initialize_actor_from_onnx

    if not CACHE.is_file():
        pytest.skip("Published b11 cache required for real ONNX parity")
    torch.manual_seed(42)
    policy = ActorCriticRecurrent(61, 64, 14, actor_hidden_dims=[512, 256, 128], rnn_hidden_dim=256)
    normalizer = EmpiricalNormalization(61)
    runner = SimpleNamespace(alg=SimpleNamespace(policy=policy), obs_normalizer=normalizer)
    critic_before = {k: v.clone() for k, v in policy.critic.state_dict().items()}
    initialize_actor_from_onnx(runner, CACHE)
    reference = BasketballOnnxPolicy(CACHE, count=2)
    frozen = {k: v.clone() for k, v in normalizer.state_dict().items()}
    normalizer.train()
    with torch.no_grad():
        for step in range(80):
            if step == 31:
                policy.reset(torch.tensor([True, False]))
                reference.reset([0])
            obs = torch.randn(2, 61) * normalizer.std + normalizer.mean
            actual = policy.act_inference(normalizer(obs))
            torch.testing.assert_close(actual, reference(obs), atol=2e-5, rtol=2e-5)
    for key, value in frozen.items():
        torch.testing.assert_close(normalizer.state_dict()[key], value, rtol=0, atol=0)
    for key, value in critic_before.items():
        torch.testing.assert_close(policy.critic.state_dict()[key], value, rtol=0, atol=0)
    policy.reset(torch.ones(2, dtype=torch.bool))
    policy.act_inference(normalizer(obs)).square().mean().backward()
    assert policy.memory_a.rnn.weight_ih_l0.grad.abs().sum() > 0
    assert policy.actor[0].weight.grad.abs().sum() > 0


def test_onnx_memory_is_per_world_and_reset_is_selective(monkeypatch):
    import onnxruntime

    class Session:
        def __init__(self, *args, **kwargs):
            pass

        def get_inputs(self):
            return [SimpleNamespace(name=k, shape=v) for k, v in
                    {"obs": [1, 61], "h_in": [1, 1, 256], "c_in": [1, 1, 256]}.items()]

        def get_outputs(self):
            return [SimpleNamespace(name=k) for k in ("actions", "h_out", "c_out")]

        def run(self, names, inputs):
            h = inputs["h_in"] + inputs["obs"][0, 0]
            c = inputs["c_in"] + 2
            return h[:, 0, :14].copy(), h.copy(), c.copy()

    monkeypatch.setattr(onnxruntime, "InferenceSession", Session)
    policy = BasketballOnnxPolicy("synthetic.onnx", count=2)
    obs = torch.zeros(2, 61)
    obs[:, 0] = torch.tensor([1, 3])
    policy(obs)
    actions = policy(obs)
    assert actions[:, 0].tolist() == [2, 6]
    policy.reset([0])
    assert np.count_nonzero(policy.h[0]) == np.count_nonzero(policy.c[0]) == 0
    assert policy.h[1, 0, 0, 0] == 6
    assert policy.c[1, 0, 0, 0] == 4
    actions = policy(obs)
    assert actions[:, 0].tolist() == [1, 9]
    policy.reset()
    assert not policy.h.any() and not policy.c.any()


@pytest.fixture(scope="module")
def cpu_env():
    env = BasketballEnv(num_envs=1, solver="mujoco", device="cpu", auto_reset=False)
    yield env
    env.close()


def test_blind_observations_exclude_ball_state(cpu_env):
    env = cpu_env
    obs, extras = env.reset()
    original = obs.clone()
    critic = extras["observations"]["critic"].clone()
    assert obs.shape == (1, 61) and critic.shape == (1, 64)
    assert not obs[:, 51:61].any()
    for state in env.states:
        env._views(state)[0][:, env.ball_q] += 0.1
        env.sync_selected_state(state, torch.tensor([0]))
    env.observation_cache = None
    obs, extras = env.get_observations()
    torch.testing.assert_close(obs, original)
    assert not torch.equal(extras["observations"]["critic"][:, -6:], critic[:, -6:])
    env.set_command([0.1, -0.05, 0.2])
    obs, _ = env.get_observations()
    torch.testing.assert_close(obs[:, 48:51], torch.tensor([[0.1, -0.05, 0.2]]))
    assert not obs[:, 51:].any()


def test_playback_never_auto_resets_and_ball_is_dynamic(cpu_env):
    env = cpu_env
    env.reset()
    assert wp.to_torch(env.model.body_mass)[env.ball_body].item() == pytest.approx(0.62)
    assert not env.hold.any()
    env.episode_length_buf[:] = env.max_episode_length
    for state in env.states:
        env._views(state)[0][:, 0] = 0.3
        env.sync_selected_state(state, torch.tensor([0]))
    _, reward, done, extras = env.step(torch.zeros(1, 14))
    assert torch.isfinite(reward).all()
    assert extras["basketball"]["failed"].all()
    assert not done.any() and env.total_resets == 0
    assert env.episode_length_buf.item() > env.max_episode_length
    env.reset()
    assert not env.failed_once.any()
    assert not env.policy_actions.any()


def test_training_metrics_preserve_terminal_ball_state():
    env = BasketballEnv(num_envs=1, solver="mujoco", device="cpu", auto_reset=True)
    try:
        for state in env.states:
            env._views(state)[0][:, env.ball_q] = 0.5
            env.sync_selected_state(state, torch.tensor([0]))
        _, _, done, extras = env.step(torch.zeros(1, 14))
        assert done.all() and env.total_resets == 1
        assert extras["basketball"]["ball_position"][0, 0] > 0.4
        assert env.ball_state()[0][0, 0] == 0
    finally:
        env.close()


@pytest.mark.skipif(os.getenv("MICRODUCK_GPU_TESTS") != "1", reason="Opt-in real CUDA physics")
@pytest.mark.parametrize("solver", ["synreal", "mujoco"])
def test_gpu_partial_reset_isolation(solver):
    if solver == "synreal":
        from examples.microduck.scenes import login

        login()
    env = BasketballEnv(num_envs=2, solver=solver, device="cuda:0", auto_reset=True)
    try:
        for _ in range(3):
            env.step(torch.zeros(2, 14, device="cuda:0"))
        pose = wp.to_torch(env.states[0].body_q).reshape(2, -1, 7)[1].clone()
        commands = env.commands[1].clone()
        env.level[:] = torch.tensor([2, 4], device=env.device)
        env._reset(torch.tensor([0], device=env.device))
        torch.testing.assert_close(wp.to_torch(env.states[0].body_q).reshape(2, -1, 7)[1], pose, rtol=0, atol=0)
        torch.testing.assert_close(env.commands[1], commands)
        assert env.level.tolist() == [2, 4]
        assert env.hold[0].item() == 0.25
        assert env.episode_length_buf.tolist() == [0, 3]
        obs, reward, _, _ = env.step(torch.zeros(2, 14, device=env.device))
        assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
    finally:
        env.close()
