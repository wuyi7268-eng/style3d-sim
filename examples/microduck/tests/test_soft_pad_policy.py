"""Checkpoint normalization, actor history, and servo mapping for soft-terrain replay."""

import json

import newton
import pytest
import torch
import warp as wp
from rsl_rl.modules import ActorCritic, EmpiricalNormalization

from examples.microduck.observations import CONTRACT, HISTORY_SPEC
from examples.microduck.play import HEAD_UP_COMMAND
from examples.microduck.policy import HistoryActorCritic, HistoryNormalization
from examples.microduck.policy_spec import EFFORT, KD, KP
from examples.microduck.recipe import RECIPE, recipe_fingerprint
from examples.microduck.scenes import SoftPadPlayback, load_policy_checkpoint


@pytest.fixture
def saved_policy(tmp_path, request):
    torch.manual_seed(42)
    task, critic_dim = getattr(request, "param", ("velocity", 70))
    cfg = {"actor_hidden_dims": [16, 8], "critic_hidden_dims": [16, 8], "activation": "elu"}
    actor = ActorCritic(61, critic_dim, 14, **cfg)
    normalizer = EmpiricalNormalization(61)
    normalizer.update(torch.randn(12, 61) * 2 + 3)
    metadata = {
        "solver": "synreal",
        "task": task,
        "recipe": RECIPE,
        "observation_contract": CONTRACT,
        "actuator": "position_pd",
        "self_collisions": False,
        "kp": KP,
        "kd": KD,
        "effort_limit": EFFORT,
        "policy_dt": 0.02,
        "physics_dt": 0.005,
        "official_alignment": {"actor_dim": 61, "critic_dim": critic_dim, "recipe_fingerprint": recipe_fingerprint()},
        "ppo": {"policy": {"class_name": "ActorCritic", **cfg}, "empirical_normalization": True},
    }
    (tmp_path / "config.json").write_text(json.dumps(metadata))
    for iteration in (9, 10):
        torch.save(
            {
                "model_state_dict": actor.state_dict(),
                "obs_norm_state_dict": normalizer.state_dict(),
                "iter": iteration,
            },
            tmp_path / f"model_{iteration}.pt",
        )
    return tmp_path, metadata, actor, normalizer


@pytest.mark.parametrize("saved_policy", [("velocity", 70), ("standup", 68), ("ballkick", 74)], indirect=True)
def test_latest_iteration_and_frozen_normalized_inference(saved_policy):
    path, _, actor, normalizer = saved_policy
    loaded = load_policy_checkpoint(path, "cpu")
    assert loaded.path.name == "model_10.pt"
    assert loaded.iteration == 10
    assert not loaded.normalizer.training
    assert not any(parameter.requires_grad for parameter in loaded.actor.parameters())
    observation = torch.randn(3, 61)
    expected = actor.act_inference((observation - normalizer.mean) / (normalizer.std + 0.01))
    before = {name: value.clone() for name, value in loaded.normalizer.state_dict().items()}
    torch.testing.assert_close(loaded.actor.act_inference(loaded.normalizer(observation)), expected)
    for name, value in before.items():
        torch.testing.assert_close(loaded.normalizer.state_dict()[name], value)


@pytest.mark.parametrize("task,critic_dim", [("velocity", 70), ("standup", 68)])
def test_history_checkpoint_loads_independent_actor_critic_encoders(tmp_path, task, critic_dim):
    cfg = {"actor_hidden_dims": [16, 8], "critic_hidden_dims": [16, 8], "activation": "elu"}
    policy = HistoryActorCritic(541, critic_dim + 480, 14, **cfg)
    normalizer = HistoryNormalization(61)
    normalizer(torch.randn(12, 541))
    normalizer.eval()
    metadata = {
        "solver": "synreal",
        "task": task,
        "recipe": RECIPE,
        "observation_contract": CONTRACT,
        "policy_variant": "history",
        "history": HISTORY_SPEC,
        "actuator": "position_pd",
        "self_collisions": False,
        "kp": KP,
        "kd": KD,
        "effort_limit": EFFORT,
        "policy_dt": 0.02,
        "physics_dt": 0.005,
        "official_alignment": {"actor_dim": 61, "critic_dim": critic_dim, "recipe_fingerprint": recipe_fingerprint()},
        "ppo": {"policy": {"class_name": "HistoryActorCritic", **cfg}, "empirical_normalization": True},
    }
    (tmp_path / "config.json").write_text(json.dumps(metadata))
    path = tmp_path / "model_0.pt"
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "obs_norm_state_dict": normalizer.state_dict(),
            "iter": 0,
        },
        path,
    )

    loaded = load_policy_checkpoint(path, "cpu")
    assert isinstance(loaded.actor, HistoryActorCritic)
    assert loaded.actor.actor[0].in_features == 125
    assert loaded.actor.critic[0].in_features == critic_dim + 64
    assert not ({id(p) for p in loaded.actor.actor_history_encoder.parameters()}
                & {id(p) for p in loaded.actor.critic_history_encoder.parameters()})
    observation = torch.randn(3, 541)
    actions = loaded.actor.act_inference(loaded.normalizer(observation))
    assert actions.shape == (3, 14)
    torch.testing.assert_close(actions, policy.act_inference(normalizer(observation)))


@pytest.mark.parametrize(
    "field,value",
    [
        ("policy_dt", 1 / 60),
        ("physics_dt", None),
        ("physics_dt", 0.0),
        ("physics_dt", float("nan")),
        ("physics_dt", 0.003),
        ("actuator", "bam"),
        ("task", "unknown_task"),
    ],
)
def test_incompatible_control_contract_is_rejected(saved_policy, field, value):
    path, metadata, _, _ = saved_policy
    metadata[field] = value
    (path / "config.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=field):
        load_policy_checkpoint(path, "cpu", allow_recipe_mismatch=True)


@pytest.mark.parametrize("override, expected_dt", [(None, None), (8, 0.0025)])
@pytest.mark.parametrize("recorded_linear", [None, 50])
@pytest.mark.parametrize("use_checkpoint", [False, True])
def test_scene_uses_central_physics_unless_checkpoint_restore_is_requested(
    saved_policy, monkeypatch, override, expected_dt, recorded_linear, use_checkpoint,
):
    from newton.viewer import ViewerNull

    import style3d_sim as s3d
    from examples.microduck.env import MicroDuckEnv, PhysicsConfig, environment_config
    from examples.microduck.scenes import SoftPadScene, create_parser

    path, metadata, _, _ = saved_policy
    if expected_dt is None:
        expected_dt = (PhysicsConfig.from_checkpoint(metadata).physics_dt if use_checkpoint
                       else environment_config("mixed_soft_pad").physics.physics_dt)
    if recorded_linear is not None:
        metadata["solver_settings"] = {"linear_iterations": recorded_linear}
        (path / "config.json").write_text(json.dumps(metadata))
    cli = ["--preview", "--checkpoint", str(path), "--device", "cpu", "--cells", "2", "2", "2"]
    if override is not None:
        cli.extend(["--substeps", str(override)])
    if use_checkpoint:
        cli.append("--use-checkpoint-environment")

    class BeforeFinalization(Exception):
        pass

    def stop_before_sdf(*args, **kwargs):
        # Timing resolves before the CUDA-only scene mesh finalization.
        raise BeforeFinalization

    monkeypatch.setattr(s3d.ModelBuilder, "finalize", stop_before_sdf)
    env = MicroDuckEnv.__new__(MicroDuckEnv)

    def construct_env(*args, **kwargs):
        env.__init__(*args, **kwargs)
        return env

    monkeypatch.setattr("examples.microduck.env.MicroDuckEnv", construct_env)
    viewer = ViewerNull(num_frames=1)
    scene = SoftPadScene.__new__(SoftPadScene)
    try:
        with pytest.raises(BeforeFinalization):
            scene.__init__(viewer, create_parser().parse_args(cli))
        assert env.step_dt == pytest.approx(0.02)
        assert env.physics_dt == pytest.approx(expected_dt)
        assert env.decimation * env.physics_dt == pytest.approx(env.step_dt)
        expected_linear = (
            recorded_linear if use_checkpoint and recorded_linear is not None else PhysicsConfig().linear_iterations
        )
        assert env.physics.linear_iterations == expected_linear
    finally:
        viewer.close()


def test_recipe_override_is_explicit_and_does_not_replace_normalizer(saved_policy):
    path, metadata, _, _ = saved_policy
    metadata["official_alignment"]["recipe_fingerprint"] = "old-recipe"
    (path / "config.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="allow-recipe-mismatch"):
        load_policy_checkpoint(path, "cpu")
    assert load_policy_checkpoint(path, "cpu", allow_recipe_mismatch=True).recipe_mismatch
    checkpoint = torch.load(path / "model_10.pt", weights_only=True)
    del checkpoint["obs_norm_state_dict"]
    torch.save(checkpoint, path / "model_10.pt")
    with pytest.raises(KeyError, match="obs_norm_state_dict"):
        load_policy_checkpoint(path, "cpu", allow_recipe_mismatch=True)


def test_train_play_and_scene_share_edited_environment_settings(saved_policy, monkeypatch):
    from dataclasses import replace

    from examples.microduck import env as environment
    from examples.microduck import scenes
    from examples.microduck.common import setup

    profile = replace(environment.environment_config("mixed_soft_pad"), spawn="ground", clearance=0.006,
                      physics=replace(environment.PhysicsConfig(), linear_iterations=21),
                      pad=replace(environment.PAD_CONFIG, density=90.0, cells=(4, 4, 2)))
    monkeypatch.setitem(environment.ENVIRONMENT_CONFIGS, "mixed_soft_pad", profile)
    path, _, _, _ = saved_policy
    cli = ["--solver", "synreal", "--environment-profile", "mixed_soft_pad"]
    training, _, _ = setup("train", cli + ["--init-from", str(path / "model_10.pt")])
    playback, _, _ = setup("play", cli + ["--checkpoint", str(path / "model_10.pt")])
    captured = {}

    class BeforeConstruction(Exception):
        pass

    def capture(**kwargs):
        captured.update(kwargs)
        raise BeforeConstruction

    monkeypatch.setattr("examples.microduck.env.MicroDuckEnv", capture)
    args = scenes.create_parser().parse_args(["--preview", "--device", "cpu", "--checkpoint", str(path)])
    with pytest.raises(BeforeConstruction):
        scenes.SoftPadScene(None, args)
    assert training.physics == playback.physics == captured["physics"] == profile.physics
    assert training.pad_config == playback.pad_config
    assert training.pad_config["density"] == captured["pad_config"].density == 90.0
    assert captured["pad_config"].cells == (4, 4, 2)
    assert captured["spawn"] == "ground" and captured["clearance"] == 0.006


def test_actor_history_reset_and_saturated_servo_targets(saved_policy):
    from examples.microduck.env import MicroDuckEnv

    path, _, _, _ = saved_policy
    checkpoint = load_policy_checkpoint(path, "cpu")
    env = MicroDuckEnv(solver="mujoco", num_envs=1, device="cpu")
    env.official.observations.spec["actor"]["enable_corruption"] = False
    env.commands.zero_()
    state = env.states[0]
    playback = SoftPadPlayback(env, checkpoint, [0.2, 0.0, 0.0])
    playback.begin_frame()
    observation = playback.last_observation
    torch.testing.assert_close(observation[:, 34:48], torch.zeros(1, 14))
    torch.testing.assert_close(observation[:, 48:51], torch.tensor([[0.2, 0.0, 0.0]]))
    torch.testing.assert_close(observation[:, 51:55], torch.tensor([HEAD_UP_COMMAND]))
    torch.testing.assert_close(observation[:, 55:], torch.zeros(1, 6))
    previous_actions = env.actions.clone()
    playback.begin_frame()
    torch.testing.assert_close(playback.last_observation[:, 34:48], previous_actions)

    env.actions[:] = torch.linspace(-10.0, 10.0, 14)
    untouched = wp.to_torch(env.control.joint_target_q).clone()
    env.apply_policy_targets()
    targets = wp.to_torch(env.control.joint_target_q)
    target_indices = env.qi if newton.use_coord_layout_targets else env.qdi
    applied = KP * (targets[target_indices] - wp.to_torch(state.joint_q)[env.qi]) - KD * wp.to_torch(state.joint_qd)[env.qdi]
    torch.testing.assert_close(applied, EFFORT * env.actions[0].sign(), atol=1.0e-6, rtol=0)
    passive = torch.ones(len(targets), dtype=torch.bool)
    passive[target_indices] = False
    torch.testing.assert_close(targets[passive], untouched[passive])

    playback.reset()
    assert playback.steps == 0
    assert playback.last_observation is None
    # RL reset returns its first observation; interactive reset itself leaves delays empty.
    env.official.observations.reset(torch.tensor([0]))
    assert all(delay.steps.item() == 0 for delay in env.official.observations.delays.values())
    playback.begin_frame()
    torch.testing.assert_close(playback.last_observation[:, 34:48], torch.zeros(1, 14))


@pytest.mark.parametrize("saved_policy", [("standup", 68), ("sitstand", 68), ("roulade", 68)], indirect=True)
def test_non_velocity_replay_preserves_task_command_slots(saved_policy):
    from examples.microduck.env import MicroDuckEnv

    path, metadata, _, _ = saved_policy
    loaded = load_policy_checkpoint(path, "cpu")
    env = MicroDuckEnv(task=metadata["task"], device="cpu", num_envs=1)
    try:
        env.official.observations.spec["actor"]["enable_corruption"] = False
        env.commands[:, :3] = torch.tensor([[0.6, 0.8, 0.0]])
        before = env.commands[:, :3].clone()
        playback = SoftPadPlayback(env, loaded, [0.3, 0.0, 0.0])
        playback.begin_frame()
        torch.testing.assert_close(env.commands[:, :3], before)
        term = env.official.observations.spec["actor"]["terms"]["command"]
        expected = torch.zeros_like(before) if term["func"].endswith(".zero_command_padding") else before
        torch.testing.assert_close(playback.last_observation[:, 48:51], expected)
    finally:
        env.close()


@pytest.mark.parametrize("saved_policy", [("standup", 68)], indirect=True)
def test_scene_constructs_checkpoint_task_and_collision_model(saved_policy, monkeypatch):
    from examples.microduck import scenes

    path, metadata, _, _ = saved_policy
    metadata["self_collisions"] = True
    (path / "config.json").write_text(json.dumps(metadata))
    captured = {}

    class BeforeConstruction(Exception):
        pass

    def capture(**kwargs):
        captured.update(kwargs)
        raise BeforeConstruction

    monkeypatch.setattr("examples.microduck.env.MicroDuckEnv", capture)
    args = scenes.create_parser().parse_args(["--preview", "--device", "cpu", "--checkpoint", str(path)])
    with pytest.raises(BeforeConstruction):
        scenes.SoftPadScene(None, args)
    assert captured["task"] == "standup" and captured["enable_self_collisions"] is True
