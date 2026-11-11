"""Environment unification must preserve saved dynamics and policy control across entry points."""

import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch
import warp as wp

from examples.microduck.common import setup
from examples.microduck.env import MicroDuckEnv, PhysicsConfig, environment_config
from examples.microduck.play import PolicyPlayback
from examples.microduck.recipe import recipe_fingerprint


@pytest.mark.parametrize("mode", ["train", "play"])
def test_saved_solver_profile_requires_explicit_restore(tmp_path, mode):
    physics = replace(PhysicsConfig(), linear_iterations=PhysicsConfig().linear_iterations + 7)
    metadata = {
        "solver": "synreal", "task": "velocity", "recipe": "official_adapted_v1",
        "observation_contract": "official_asymmetric_pd_v2", "physics": asdict(physics),
        "official_alignment": {"recipe_fingerprint": recipe_fingerprint()}, "ppo": {},
    }
    checkpoint = tmp_path / "model_100.pt"
    checkpoint.touch()
    (tmp_path / "config.json").write_text(json.dumps(metadata))
    flag = "--resume" if mode == "train" else "--checkpoint"
    cli = ["--solver", "synreal", flag, str(checkpoint)]
    if mode == "train":
        with pytest.raises(SystemExit):
            setup(mode, cli)
    else:
        args, _, _ = setup(mode, cli)
        assert args.physics == PhysicsConfig()
    args, _, _ = setup(mode, cli + ["--use-checkpoint-environment"])
    assert args.physics == physics
    assert args.physics.linear_iterations != PhysicsConfig().linear_iterations


def test_training_profiles_and_cli_use_central_environment(monkeypatch):
    from examples.microduck import env as environment

    profile = replace(environment.environment_config("mixed_soft_pad"), num_envs=17, pad_probability=0.3,
                      physics=replace(PhysicsConfig(), linear_iterations=23),
                      pad=replace(environment.PAD_CONFIG, density=95.0))
    monkeypatch.setitem(environment.ENVIRONMENT_CONFIGS, "mixed_soft_pad", profile)
    cli = ["--training-config", "examples/microduck/configs/rtx4090_mixed_soft_pad.json"]
    args, _, _ = setup("train", cli)
    assert (args.num_envs, args.terrain, args.pad_probability) == (17, "mixed_soft_pad", 0.3)
    assert args.physics.linear_iterations == 23
    assert args.pad_config["density"] == 95.0
    args, _, _ = setup("train", cli + ["--num-envs", "29", "--pad-probability", "0", "--no-randomize-foot-friction"])
    assert args.num_envs == 29 and args.pad_probability == 0 and not args.randomize_foot_friction
    args, _, _ = setup("train", cli + ["--environment-profile", "official"])
    assert args.terrain == "default"


def test_mixed_replay_uses_current_pad_but_can_restore_recorded_environment(tmp_path):
    from examples.microduck.env import PAD_CONFIG

    checkpoint = tmp_path / "model_100.pt"
    checkpoint.touch()
    metadata = {
        "solver": "synreal", "task": "velocity", "recipe": "official_adapted_v1",
        "observation_contract": "official_asymmetric_pd_v2", "solver_settings": {"linear_iterations": 50},
        "official_alignment": {"recipe_fingerprint": recipe_fingerprint()}, "ppo": {},
        "terrain": {"kind": "mixed_soft_pad", "pad_probability": 0.8,
                    "pad": asdict(replace(PAD_CONFIG, density=120.0))},
    }
    (tmp_path / "config.json").write_text(json.dumps(metadata))
    cli = ["--solver", "synreal", "--checkpoint", str(checkpoint)]
    args, _, _ = setup("play", cli)
    assert args.terrain == "mixed_soft_pad" and args.pad_probability == 0.5
    assert args.pad_sampling == "fixed_groups_center"
    assert args.pad_config["density"] == PAD_CONFIG.density
    assert args.physics == environment_config("mixed_soft_pad").physics
    assert args.physics.decimation == 8 and args.physics.physics_dt == 0.0025
    assert args.physics.joint_constraint_primal_iterations == 4
    training, _, _ = setup("train", ["--solver", "synreal", "--terrain", "mixed_soft_pad"])
    assert training.physics == args.physics
    args, _, _ = setup("play", cli + ["--use-checkpoint-environment", "--pad-probability", "0"])
    assert args.pad_config["density"] == 120.0 and args.physics.linear_iterations == 50
    assert args.pad_probability == 0
    assert args.pad_sampling == "independent_per_episode"
    assert (tmp_path / "config.json").read_text() == json.dumps(metadata)


def test_changed_contact_stiffness_requires_explicit_checkpoint_environment(tmp_path):
    from examples.microduck.env import PAD_CONFIG

    pad = asdict(PAD_CONFIG)
    pad.pop("contact_stiffness")
    metadata = {
        "solver": "synreal", "task": "velocity", "recipe": "official_adapted_v1",
        "observation_contract": "official_asymmetric_pd_v2",
        "physics": asdict(environment_config("mixed_soft_pad").physics),
        "official_alignment": {"recipe_fingerprint": recipe_fingerprint()}, "ppo": {},
        "terrain": {"kind": "mixed_soft_pad", "pad_probability": 0.5, "pad": pad,
                    "soft_contact": {"ke": 50000.0}},
    }
    checkpoint = tmp_path / "model_100.pt"
    checkpoint.touch()
    saved = json.dumps(metadata)
    checkpoint.with_name("config.json").write_text(saved)
    cli = ["--solver", "synreal", "--resume", str(checkpoint)]
    with pytest.raises(SystemExit):
        setup("train", cli)
    args, _, _ = setup("train", cli + ["--use-checkpoint-environment"])
    assert args.pad_config["contact_stiffness"] == 50000.0
    assert args.pad_sampling == "independent_per_episode"
    # A newly saved recipe must resume despite JSON converting tuples to lists.
    metadata["terrain"]["pad"]["contact_stiffness"] = PAD_CONFIG.contact_stiffness
    metadata["terrain"]["soft_contact"]["ke"] = PAD_CONFIG.contact_stiffness
    metadata["terrain"]["sampling"] = "fixed_groups_center"
    checkpoint.with_name("config.json").write_text(json.dumps(metadata))
    args, _, _ = setup("train", cli)
    assert args.pad_config["contact_stiffness"] == PAD_CONFIG.contact_stiffness


def test_legacy_solver_metadata_and_explicit_substeps():
    metadata = {"physics_dt": 0.005, "policy_dt": 0.02, "solver_settings": {"linear_iterations": 50}}
    physics = PhysicsConfig.from_checkpoint(metadata)
    override = physics.with_substeps(8)
    assert override.physics_dt == 0.0025 and override.linear_iterations == 50
    assert override.physics_dt * override.decimation == physics.physics_dt * physics.decimation
    del metadata["solver_settings"]
    assert PhysicsConfig.from_checkpoint(metadata) == PhysicsConfig()


@pytest.mark.parametrize("settings", [{"physics_dt": 0}, {"decimation": 0}, {"linear_iterations": -1}])
def test_invalid_physics_is_rejected_before_constructing_simulation(settings):
    with pytest.raises(ValueError):
        PhysicsConfig(**settings)


def test_training_and_fast_playback_produce_identical_physics_on_cpu():
    trained = MicroDuckEnv(solver="mujoco", num_envs=1, device="cpu", seed=42)
    replayed = MicroDuckEnv(solver="mujoco", num_envs=1, device="cpu", seed=42)
    for env in (trained, replayed):
        env.official.pushes.events = {}
        env.official.update_commands = lambda: None
        env.official.observations.spec["actor"]["enable_corruption"] = False
    playback = PolicyPlayback(replayed)
    try:
        for index in range(12):
            actions = torch.sin(torch.arange(14).float() + index * 0.1)[None] * 0.03
            trained.step(actions)
            playback.step(actions)
            for name in ("body_q", "body_qd", "joint_q", "joint_qd"):
                torch.testing.assert_close(
                    wp.to_torch(getattr(trained.states[0], name)), wp.to_torch(getattr(replayed.states[0], name)),
                    atol=0, rtol=0,
                )
        assert trained.sim_time == replayed.sim_time
    finally:
        trained.close()
        replayed.close()


def test_mixed_playback_disables_interval_events_but_training_keeps_them():
    env = MicroDuckEnv(solver="mujoco", num_envs=1, device="cpu", seed=42)
    calls = []
    env.official.pushes.apply = lambda: calls.append("push")
    actions = torch.zeros(1, 14)
    try:
        env.step(actions)
        assert calls == ["push"]
        calls.clear()
        # Exercise the mixed-playback dispatch using real CPU dynamics, without
        # claiming that MuJoCo supports deformable pads.
        from unittest.mock import patch

        real_step = env.step

        def dispatch(*args, **kwargs):
            env.mixed_terrain = False
            return real_step(*args, **kwargs)

        env.mixed_terrain = True
        env.max_episode_length = 1
        resets = env.total_resets
        with patch.object(env, "step", side_effect=dispatch):
            env.step_playback(actions)
        assert calls == []
        assert env.total_resets == resets
        env.step_playback(actions)
        assert calls == []
        assert env.total_resets == resets
        env.step(actions)
        assert calls == ["push"]
        assert env.total_resets == resets + 1
    finally:
        env.close()


@pytest.mark.parametrize("mixed_dispatch", [False, True])
def test_playback_passes_mouse_force_to_every_substep_and_releases_it(monkeypatch, mixed_dispatch):
    env = MicroDuckEnv(solver="mujoco", num_envs=1, device="cpu", seed=42)
    forces = []
    dragging = True
    solver_step = env.solver.step
    full_step = env.step

    def apply_forces(state):
        if dragging:
            wp.to_torch(state.body_f)[env.root, 0] += 1.0

    def record_step(state, *args):
        forces.append(wp.to_torch(state.body_f).clone())
        return solver_step(state, *args)

    def dispatch(*args, **kwargs):
        # Test mixed playback routing with real CPU steps; deformable dynamics
        # require SynReal and are not exercised by this force-delivery test.
        env.mixed_terrain = False
        return full_step(*args, **kwargs)

    monkeypatch.setattr(env.solver, "step", record_step)
    monkeypatch.setattr(env, "render", lambda: None)
    monkeypatch.setattr(env, "step", dispatch)
    env.viewer = SimpleNamespace(apply_forces=apply_forces, close=lambda: None)
    actions = torch.zeros(1, 14)

    def playback():
        env.mixed_terrain = mixed_dispatch
        env.step_playback(actions)

    try:
        playback()
        assert len(forces) == env.decimation
        expected = torch.zeros_like(forces[0])
        expected[env.root, 0] = 1.0
        for force in forces:
            torch.testing.assert_close(force, expected, atol=0, rtol=0)
        forces.clear()
        dragging = False
        playback()
        assert len(forces) == env.decimation
        assert all(torch.count_nonzero(force) == 0 for force in forces)
        forces.clear()
        dragging = True
        env.step(actions)
        assert len(forces) == env.decimation
        assert all(torch.count_nonzero(force) == 0 for force in forces)
    finally:
        env.close()
