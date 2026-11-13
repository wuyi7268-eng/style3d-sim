"""Cloth-only selection, causal load history, function-preserving migration and PPO round trips."""

import copy
from types import SimpleNamespace

import pytest
import torch
from rsl_rl.modules import ActorCritic, EmpiricalNormalization

from examples.microduck.checkpoint import checkpoint_policy_variant, initialize_transfer
from examples.microduck.cloth_policy import (
    LEGACY_SPEC,
    SPEC,
    ClothHistory,
    ClothHistoryActorCritic,
    ClothNormalization,
)
from examples.microduck.common import setup
from examples.microduck.observations import configure_policy_observations
from examples.microduck.policy import HistoryActorCritic, HistoryNormalization
from examples.microduck.ppo import create_runner, microduck_symmetry, ppo_config


@pytest.mark.parametrize("terrain", ["cloth", "hammock", "elastic_bridge"])
def test_cloth_scope_and_legacy_isolation(terrain):
    assert setup("train", ["--solver", "synreal", "--terrain", terrain])[0].policy_variant == "cloth_history"
    assert setup("train", ["--solver", "synreal", "--terrain", terrain, "--task", "velstand"])[0].policy_variant == "standard"
    with pytest.raises(SystemExit):
        setup("train", ["--solver", "synreal", "--terrain", terrain, "--policy-variant", "history"])


def test_noncloth_tasks_keep_existing_networks():
    for terrain in ("default", "mixed_soft_pad", "sand", "trampoline"):
        args = ["--solver", "synreal", "--terrain", terrain]
        assert setup("train", args)[0].policy_variant == "standard"
        assert setup("train", args + ["--policy-variant", "history"])[0].policy_variant == "history"
        with pytest.raises(SystemExit):
            setup("train", args + ["--policy-variant", "cloth_history"])


def test_load_feedback_timing_order_partial_reset_and_repeated_reads():
    history = ClothHistory(2, "cpu", version=2)
    current = torch.zeros(2, 74)
    history.read(current)
    for step in range(1, 24):
        history.advance()
        torque = torch.arange(14).float().repeat(2, 1) * 0.01
        history.record_torque(torque)
        history.record_torque(-torque / 2)
        expected = torque[:, [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]] / 4 / 0.96
        torch.testing.assert_close(history.feedback, expected)
        current[:, :48] = step
        current[:, 61:71] = history.feedback
        past = history.read(current).clone().reshape(2, 20, 61)
        times = torch.tensor([max(0, step - 1 - lag) for lag in range(20)], dtype=torch.float)
        torch.testing.assert_close(past[0, :, 0], times)
        torch.testing.assert_close(history.read(current), past.flatten(1))
    history.reset(torch.tensor([0]))
    current[0, 61:71] = history.feedback[0]
    after = history.read(current).reshape(2, 20, 61)
    assert not after[0, :, 48:].any()
    torch.testing.assert_close(after[1], past[1])
    torch.testing.assert_close(after[0, :, :48], current[0, :48].expand(20, -1))


@pytest.mark.parametrize("source_variant", ["standard", "history", "cloth_history_v1", "cloth_history"])
def test_transfer_preserves_actor_critic_and_statistics(tmp_path, source_variant):
    torch.manual_seed(12)
    dims = {"standard": (61, 70), "history": (541, 550), "cloth_history_v1": (1231, 1240), "cloth_history": (1294, 1303)}
    cls = {"standard": ActorCritic, "history": HistoryActorCritic,
           "cloth_history_v1": ClothHistoryActorCritic, "cloth_history": ClothHistoryActorCritic}[source_variant]
    norm = {"standard": EmpiricalNormalization, "history": HistoryNormalization,
            "cloth_history_v1": ClothNormalization, "cloth_history": ClothNormalization}[source_variant]
    options = {"actor_hidden_dims": [32, 16], "critic_hidden_dims": [32, 16]}
    source = cls(*dims[source_variant], 14, **options)
    a_norm = norm(74 if source_variant == "cloth_history" else 71 if source_variant == "cloth_history_v1" else 61)
    c_dim = 83 if source_variant == "cloth_history" else 80 if source_variant == "cloth_history_v1" else 70
    c_norm = norm(c_dim)
    if source_variant != "standard":
        c_norm = norm(c_dim, a_norm)
    a_norm(torch.randn(32, dims[source_variant][0]) * 2 + 1)
    c_norm(torch.randn(32, dims[source_variant][1]) * 3 - 1)
    path = tmp_path / "model_12.pt"
    torch.save({"model_state_dict": source.state_dict(), "obs_norm_state_dict": a_norm.state_dict(),
                    "privileged_obs_norm_state_dict": c_norm.state_dict(), "iter": 12}, path)
    target = ClothHistoryActorCritic(1294, 1303, 14, **options)
    target_norm = ClothNormalization(74)
    runner = SimpleNamespace(device="cpu", alg=SimpleNamespace(policy=target), empirical_normalization=True,
                             obs_normalizer=target_norm, privileged_obs_normalizer=ClothNormalization(83, target_norm))
    info = initialize_transfer(runner, path)
    assert info["network_expanded"] == (source_variant != "cloth_history")
    source.eval(), target.eval(), a_norm.eval(), c_norm.eval()
    runner.obs_normalizer.eval(), runner.privileged_obs_normalizer.eval()
    a, c, past = torch.randn(8, 74), torch.randn(8, 83), torch.randn(8, 20, 61)
    target_a, target_c = torch.cat((a, past.flatten(1)), -1), torch.cat((c, past.flatten(1)), -1)
    if source_variant == "standard":
        source_a, source_c = a[:, :61], c[:, :70]
    elif source_variant == "history":
        old_past = past[:, :10, :48].flatten(1)
        source_a, source_c = torch.cat((a[:, :61], old_past), -1), torch.cat((c[:, :70], old_past), -1)
    elif source_variant == "cloth_history_v1":
        old_past = past[:, :, :58].flatten(1)
        source_a, source_c = torch.cat((a[:, :71], old_past), -1), torch.cat((c[:, :80], old_past), -1)
    else:
        source_a, source_c = target_a, target_c
    torch.testing.assert_close(source.act_inference(a_norm(source_a)), target.act_inference(target_norm(target_a)))
    torch.testing.assert_close(source.evaluate(c_norm(source_c)), target.evaluate(runner.privileged_obs_normalizer(target_c)))
    # Expanded inputs must be learnable, even though their initial contribution is exactly zero.
    target.act_inference(target_norm(target_a)).square().mean().backward()
    assert target.actor[0].weight.grad[:, 61:71].abs().sum() > 0
    assert target.actor[0].weight.grad[:, 71:74].abs().sum() > 0
    if source_variant in ("history", "cloth_history_v1"):
        assert target.actor_history_encoder[0].weight.grad.reshape(128, 20, 61)[:, :, 58:].abs().sum() > 0
    if source_variant == "history":
        assert target.actor_history_encoder[0].weight.grad.reshape(128, 20, 61)[:, 10:].abs().sum() > 0
        assert target.actor_history_encoder[0].weight.grad.reshape(128, 20, 61)[:, :, 48:].abs().sum() > 0


class ClothEnv:
    num_envs, num_actions, num_privileged_obs, device, max_episode_length = 4, 14, 1303, "cpu", 100

    def __init__(self):
        self.cfg = {"task": "velocity", "terrain": {"kind": "cloth"}}
        self.episode_length_buf = torch.zeros(4, dtype=torch.long)
        self.history = ClothHistory(4, "cpu")
        self.actor = torch.randn(4, 74)

    def get_observations(self):
        history = self.history.read(self.actor)
        actor = torch.cat((self.actor, history), -1)
        critic = torch.cat((self.actor[:, :61], torch.ones(4, 9), self.actor[:, 61:], history), -1)
        return actor, {"observations": {"critic": critic}}

    def step(self, actions):
        self.history.advance()
        self.history.record_torque(actions.clamp(-0.96, 0.96))
        self.actor = torch.cat((torch.randn(4, 61), self.history.feedback, torch.randn(4, 3)), -1)
        actor, extras = self.get_observations()
        return actor, -actions.square().sum(-1), torch.zeros(4, dtype=torch.long), extras


def test_ppo_checkpoint_normalization_and_symmetry(tmp_path):
    cfg = ppo_config(4, "cloth_history")
    cfg["num_steps_per_env"] = 4
    cfg["policy"].update(actor_hidden_dims=[32], critic_hidden_dims=[32])
    cfg["algorithm"].update(num_learning_epochs=1, num_mini_batches=1, symmetry_cfg=None)
    env = ClothEnv()
    runner = create_runner(env, cfg, str(tmp_path), "cpu")
    before = copy.deepcopy(runner.alg.policy.state_dict())
    runner.learn(2)
    for name in ("actor_history_encoder.0.weight", "critic_history_encoder.0.weight"):
        assert not torch.equal(before[name], runner.alg.policy.state_dict()[name])
    obs, extras = env.get_observations()
    mirrored, _ = microduck_symmetry(env, obs)
    twice, _ = microduck_symmetry(env, mirrored[4:])
    torch.testing.assert_close(twice[4:], obs)
    expected = runner.get_inference_policy()(obs)
    restored = create_runner(ClothEnv(), cfg, device="cpu")
    restored.load(str(tmp_path / "model_1.pt"))
    torch.testing.assert_close(restored.get_inference_policy()(obs), expected)
    normalized = restored.obs_normalizer(obs)
    critic = restored.privileged_obs_normalizer(extras["observations"]["critic"])
    torch.testing.assert_close(normalized[:, 74:], critic[:, 83:])
    torch.testing.assert_close(normalized[:, 61:71], obs[:, 61:71])
    assert restored.alg.optimizer.state


def test_metadata_validates_feedback_semantics():
    metadata = {"policy_variant": "cloth_history", "history": copy.deepcopy(SPEC), "task": "velocity",
                    "terrain": {"kind": "elastic_bridge"}, "ppo": ppo_config(4, "cloth_history")}
    assert checkpoint_policy_variant(metadata) == "cloth_history"
    legacy = {**metadata, "history": LEGACY_SPEC}
    assert checkpoint_policy_variant(legacy) == "cloth_history"
    metadata["history"]["feedback_units"] = "amperes"
    with pytest.raises(ValueError, match="contract"):
        checkpoint_policy_variant(metadata)


def test_real_control_loop_feedback_and_playback_on_cpu():
    from examples.microduck.env import MicroDuckEnv
    from examples.microduck.play import PolicyPlayback

    # Exercise the real controller on CPU rigid ground, without claiming cloth dynamics validation.
    env = MicroDuckEnv(device="cpu", num_envs=1)
    try:
        env.cfg["terrain"] = {"kind": "cloth"}
        configure_policy_observations(env, "cloth_history")
        obs, extras = env.get_observations()
        assert obs.shape == (1, 1294)
        assert extras["observations"]["critic"].shape == (1, 1303)
        assert not obs[:, 61:71].any()
        env.step(torch.full((1, 14), 0.03), auto_reset=False)
        obs, _ = env.get_observations()
        assert obs[:, 61:71].abs().sum() > 0
        from examples.microduck.cloth_current import CURRENT_SPEC

        assert (obs[:, 61:71] >= 0).all()
        assert obs[:, 61:71].max() <= 0.96 / CURRENT_SPEC["kt_nm_per_a"] + 0.001
        assert obs[:, 71:74].abs().sum() > 0
        assert obs[:, 71:74].abs().max() <= 16
        previous = obs.clone()
        env.step(torch.zeros(1, 14), auto_reset=False)
        obs, _ = env.get_observations()
        torch.testing.assert_close(obs[:, 74:].reshape(1, 20, 61)[:, 0, 58:], previous[:, 71:74])
        torch.testing.assert_close(PolicyPlayback(env).observe(), obs)
        env._reset(torch.tensor([0]))
        obs, _ = env.get_observations()
        assert not obs[:, 61:71].any()
        assert not obs[:, 74:].reshape(1, 20, 61)[:, :, 48:].any()
    finally:
        env.close()


def test_flat_checkpoint_train_entry_migration_and_standalone_replay(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    from examples.microduck import train
    from examples.microduck.checkpoint import transfer_metadata
    from examples.microduck.scenes import load_policy_checkpoint

    source_path = Path("/home/ubuntu/models/velocity_24300/model_24300.pt")
    if not source_path.is_file():
        pytest.skip("Optional real flat-ground checkpoint is unavailable")
    source_metadata = transfer_metadata(source_path)

    class Env(ClothEnv):
        mixed_terrain = False

        def __init__(self):
            super().__init__()
            self.cfg.update({key: source_metadata[key] for key in (
                "solver", "task", "recipe", "observation_contract", "actuator", "self_collisions",
                "kp", "kd", "effort_limit", "policy_dt", "physics_dt", "official_alignment",
            )})
            self.cfg.update(policy_variant="cloth_history", history=copy.deepcopy(SPEC))
            self.official = SimpleNamespace(spec={}, common_step_counter=0)

        def close(self):
            pass

    env = Env()
    monkeypatch.setattr(train, "create_env", lambda args: env)
    from examples.microduck import ppo

    monkeypatch.setattr(ppo, "apply_official_ppo", lambda cfg, spec, count: cfg)
    config = tmp_path / "training.json"
    config.write_text(json.dumps({"ppo": {"num_steps_per_env": 4, "algorithm": {
        "num_learning_epochs": 1, "num_mini_batches": 1, "symmetry_cfg": None,
    }}}))
    assert train.main([
        "--solver", "synreal", "--terrain", "cloth", "--init-from", str(source_path),
        "--device", "cpu", "--num-envs", "4", "--iterations", "1",
        "--log-root", str(tmp_path / "runs"), "--training-config", str(config),
    ]) == 0
    run = next((tmp_path / "runs" / "microduck_synreal").iterdir())
    metadata = json.loads((run / "config.json").read_text())
    assert metadata["transfer_from"]["network_expanded"]
    assert metadata["history"] == SPEC
    assert checkpoint_policy_variant(metadata) == "cloth_history"
    loaded = load_policy_checkpoint(run, "cpu", allow_recipe_mismatch=True)
    obs, _ = env.get_observations()
    assert torch.isfinite(loaded.actor.act_inference(loaded.normalizer(obs))).all()
    args, _, _ = setup("play", ["--solver", "synreal", "--checkpoint", str(loaded.path), "--allow-recipe-mismatch"])
    assert args.policy_variant == "cloth_history"
    assert args.terrain == "cloth"
