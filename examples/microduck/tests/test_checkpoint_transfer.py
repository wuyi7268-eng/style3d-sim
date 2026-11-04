"""Transfer keeps the learned function while retaining a fresh optimizer."""

from types import SimpleNamespace

import pytest
import torch
from rsl_rl.modules import ActorCritic, EmpiricalNormalization

from examples.microduck.checkpoint import initialize_transfer, resolve_checkpoint


@pytest.mark.parametrize("inherit_stage", [False, True])
def test_transfer_restores_policy_and_both_normalizers_but_not_optimizer(tmp_path, inherit_stage):
    def policy():
        return ActorCritic(61, 70, 14, actor_hidden_dims=[16], critic_hidden_dims=[16])

    torch.manual_seed(7)
    source = policy()
    actor_norm, critic_norm = EmpiricalNormalization(61), EmpiricalNormalization(70)
    actor_norm(torch.randn(64, 61) * 2 + 3)
    critic_norm(torch.randn(64, 70) * 4 - 2)
    path = tmp_path / "model_123.pt"
    torch.save(
        {
            "model_state_dict": source.state_dict(),
            "obs_norm_state_dict": actor_norm.state_dict(),
            "privileged_obs_norm_state_dict": critic_norm.state_dict(),
            "iter": 123,
            "optimizer_state_dict": {"must_not_be_loaded": True},
        },
        path,
    )
    target = policy()
    optimizer = torch.optim.Adam(target.parameters(), lr=0.0001)
    runner = SimpleNamespace(
        device="cpu",
        alg=SimpleNamespace(policy=target, optimizer=optimizer),
        empirical_normalization=True,
        obs_normalizer=EmpiricalNormalization(61),
        privileged_obs_normalizer=EmpiricalNormalization(70),
        current_learning_iteration=999,
    )
    provenance = initialize_transfer(runner, path, inherit_training_stage=inherit_stage)
    source.eval(), target.eval(), actor_norm.eval(), runner.obs_normalizer.eval()
    observations = torch.randn(8, 61)
    torch.testing.assert_close(
        source.act_inference(actor_norm(observations)),
        target.act_inference(runner.obs_normalizer(observations)),
        atol=0,
        rtol=0,
    )
    for name, value in critic_norm.state_dict().items():
        torch.testing.assert_close(value, runner.privileged_obs_normalizer.state_dict()[name], atol=0, rtol=0)
    assert runner.current_learning_iteration == (124 if inherit_stage else 0)
    assert provenance["training_stage_inherited"] is inherit_stage
    assert provenance["start_iteration"] == runner.current_learning_iteration
    assert not optimizer.state
    assert optimizer.param_groups[0]["lr"] == 0.0001
    assert provenance["source_iteration"] == 123
    assert not provenance["optimizer_restored"]


@pytest.mark.parametrize("mode", ["train", "play"])
def test_inherit_stage_requires_transfer(mode):
    from examples.microduck.common import setup

    with pytest.raises(SystemExit):
        setup(mode, ["--inherit-training-stage"])


@pytest.mark.parametrize("variant", ["standard", "history"])
@pytest.mark.parametrize("inherit_stage", [False, True])
@pytest.mark.parametrize("task", ["velocity", "velstand"])
def test_transfer_training_stage_through_ppo_and_resume(tmp_path, monkeypatch, variant, inherit_stage, task):
    import copy
    import json

    from examples.microduck import train
    from examples.microduck.observations import CONTRACT, HISTORY_SPEC
    from examples.microduck.policy_spec import EFFORT, KD, KP
    from examples.microduck.ppo import create_runner, ppo_config
    from examples.microduck.recipe import RECIPE

    class Env:
        num_envs, num_actions, max_episode_length, device = 4, 14, 100, "cpu"
        num_privileged_obs = 550 if variant == "history" else 70
        mixed_terrain = False

        def __init__(self):
            self.cfg = {"observation_contract": CONTRACT}
            self.episode_length_buf = torch.zeros(4, dtype=torch.long)
            self.official = SimpleNamespace(spec={}, common_step_counter=0)
            self.reset_steps = []
            self.step_counts = []
            self.closed = False

        def get_observations(self):
            actor = torch.ones(4, 541 if variant == "history" else 61)
            return actor, {"observations": {"critic": torch.ones(4, self.num_privileged_obs)}}

        def reset(self):
            self.reset_steps.append(self.official.common_step_counter)
            return self.get_observations()

        def step(self, actions):
            assert actions.shape == (4, 14)
            assert torch.isfinite(actions).all()
            self.step_counts.append(self.official.common_step_counter)
            self.official.common_step_counter += 1
            obs, extras = self.get_observations()
            return obs, torch.arange(4).float(), torch.zeros(4, dtype=torch.long), extras

        def close(self):
            self.closed = True

    cfg = ppo_config(4, variant)
    cfg["policy"].update(actor_hidden_dims=[16], critic_hidden_dims=[16])
    cfg["algorithm"].update(num_learning_epochs=1, num_mini_batches=1)
    source = create_runner(Env(), cfg, str(tmp_path), "cpu")
    source.current_learning_iteration = 16200
    source.logger_type = "tensorboard"
    checkpoint = tmp_path / "model_16200.pt"
    source.save(str(checkpoint))
    metadata = dict(
        solver="synreal", task=task, recipe=RECIPE, observation_contract=CONTRACT,
        actuator="position_pd", self_collisions=False, kp=KP, kd=KD, effort_limit=EFFORT,
        policy_dt=0.02, physics_dt=0.005, policy_variant=variant, history=HISTORY_SPEC, ppo=cfg,
        official_alignment={"actor_dim": 61, "critic_dim": 70, "recipe_fingerprint": "old-rewards"},
    )
    if task == "velstand":
        from dataclasses import asdict, replace

        from examples.microduck.env import PhysicsConfig

        metadata["physics_dt"] = 0.0025
        metadata["physics"] = asdict(replace(PhysicsConfig().with_substeps(8), joint_constraint_primal_iterations=4))
        metadata["ppo"] = copy.deepcopy(cfg)
        metadata["ppo"]["teacher_bc"] = {"must_not_be_inherited_by_student_finetune": True}
    (tmp_path / "config.json").write_text(json.dumps(metadata))
    config_path = tmp_path / "training.json"
    config_path.write_text(json.dumps({"ppo": {"algorithm": cfg["algorithm"]}}))
    env = Env()
    monkeypatch.setattr(train, "create_env", lambda args: env)
    # Keep real CLI validation, transfer loading, PPO updates and checkpoint saving; only physics is replaced.
    from examples.microduck import ppo

    monkeypatch.setattr(ppo, "apply_official_ppo", lambda config, spec, num_envs: config)
    arguments = [
        "--solver", "synreal", "--task", task, "--init-from", str(checkpoint), "--device", "cpu", "--num-envs", "4",
        "--iterations", "2", "--log-root", str(tmp_path / "runs"), "--training-config", str(config_path),
    ]
    if inherit_stage:
        arguments.append("--inherit-training-stage")
    assert train.main(arguments) == 0
    start = 16201 if inherit_stage else 0
    assert env.reset_steps == ([start * 24] if inherit_stage else [])
    assert env.step_counts == list(range(start * 24, (start + 2) * 24))
    assert env.closed
    run = next((tmp_path / "runs" / "microduck_synreal").iterdir())
    saved = run / f"model_{start + 1}.pt"
    assert saved.is_file()
    saved_metadata = json.loads((run / "config.json").read_text())
    assert not saved_metadata["ppo"].get("teacher_bc")
    provenance = saved_metadata["transfer_from"]
    assert provenance["start_iteration"] == start
    assert provenance["training_stage_inherited"] is inherit_stage
    resumed = create_runner(Env(), cfg, str(tmp_path / "resumed"), "cpu")
    resumed.load(str(saved))
    assert resumed.current_learning_iteration == start + 1
    assert resumed.alg.optimizer.state
    resumed.current_learning_iteration += 1
    resumed.learn(1)
    assert resumed.current_learning_iteration == start + 2
    if inherit_stage:
        config_path.write_text(json.dumps({"ppo": {"num_steps_per_env": 12}}))
        with pytest.raises(ValueError, match="num_steps_per_env"):
            train.main(arguments)
        from examples.microduck.common import setup

        with pytest.raises(SystemExit):
            setup("train", ["--resume", str(checkpoint), "--inherit-training-stage"])


def test_directory_chooses_highest_numbered_checkpoint(tmp_path):
    for name in ("model_9.pt", "model_100.pt", "model_final.pt"):
        (tmp_path / name).touch()
    assert resolve_checkpoint(tmp_path).name == "model_100.pt"
    with pytest.raises(ValueError):
        resolve_checkpoint(tmp_path / "missing")


def test_first_ppo_action_uses_loaded_normalization_like_later_steps(tmp_path):
    from rsl_rl.runners import OnPolicyRunner

    from examples.microduck.ppo import InitialObservationEnv, ppo_config

    class Env:
        num_envs, num_actions, max_episode_length, device = 4, 14, 100, "cpu"
        episode_length_buf = torch.zeros(4, dtype=torch.long)

        def __init__(self):
            self.cfg = {}

        def get_observations(self):
            return torch.full((4, 61), 3.0), {"observations": {"critic": torch.full((4, 70), 5.0)}}

        def step(self, actions):
            obs, extras = self.get_observations()
            return obs, torch.arange(4).float(), torch.zeros(4, dtype=torch.long), extras

    env = Env()
    cfg = ppo_config(4)
    cfg["num_steps_per_env"] = 2
    cfg["policy"].update(actor_hidden_dims=[16], critic_hidden_dims=[16])
    cfg["algorithm"].update(num_learning_epochs=1, num_mini_batches=1)
    runner = OnPolicyRunner(env, cfg, str(tmp_path), "cpu")
    for normalizer, center in ((runner.obs_normalizer, 3.0), (runner.privileged_obs_normalizer, 5.0)):
        normalizer._mean.fill_(center)
        normalizer._std.fill_(2.0)
        normalizer.count.fill_(int(1.0e8))
    recorded = []
    act = runner.alg.act

    def record(obs, critic):
        recorded.append((obs.clone(), critic.clone()))
        return act(obs, critic)

    runner.alg.act = record
    runner.env = InitialObservationEnv(env, runner)
    runner.learn(1)
    assert len(recorded) == 2
    for obs, critic in recorded:
        assert not obs.any()
        assert not critic.any()


@pytest.mark.parametrize("variant", ["standard", "history"])
def test_runner_resume_normalizes_first_action_for_both_networks(tmp_path, variant):
    from examples.microduck.ppo import create_runner, ppo_config

    class Env:
        num_envs, num_actions, max_episode_length, device = 4, 14, 100, "cpu"
        num_privileged_obs = 550 if variant == "history" else 70
        episode_length_buf = torch.zeros(4, dtype=torch.long)
        cfg = {}

        def get_observations(self):
            actor, critic = torch.full((4, 61), 3.0), torch.full((4, 70), 5.0)
            if variant == "history":
                history = torch.full((4, 480), 3.0)
                actor, critic = torch.cat((actor, history), -1), torch.cat((critic, history), -1)
            return actor, {"observations": {"critic": critic}}

        def step(self, actions):
            obs, extras = self.get_observations()
            return obs, torch.arange(4).float(), torch.zeros(4, dtype=torch.long), extras

    cfg = ppo_config(4, variant)
    cfg["num_steps_per_env"] = 2
    cfg["policy"].update(actor_hidden_dims=[16], critic_hidden_dims=[16])
    cfg["algorithm"].update(num_learning_epochs=1, num_mini_batches=1)
    original = create_runner(Env(), cfg, str(tmp_path), "cpu")
    for normalizer, center in ((original.obs_normalizer, 3.0), (original.privileged_obs_normalizer, 5.0)):
        stats = getattr(normalizer, "current", normalizer)
        stats._mean.fill_(center)
        stats._std.fill_(2.0)
        stats.count.fill_(int(1.0e8))
    checkpoint = tmp_path / "source.pt"
    original.logger_type = "tensorboard"
    original.save(str(checkpoint))
    restored = create_runner(Env(), cfg, str(tmp_path), "cpu")
    restored.load(str(checkpoint))
    recorded = []
    act = restored.alg.act

    def record(obs, critic):
        recorded.append((obs.clone(), critic.clone()))
        return act(obs, critic)

    restored.alg.act = record
    restored.learn(1)
    assert len(recorded) == 2
    for obs, critic in recorded:
        torch.testing.assert_close(obs, torch.zeros_like(obs))
        torch.testing.assert_close(critic, torch.zeros_like(critic))
