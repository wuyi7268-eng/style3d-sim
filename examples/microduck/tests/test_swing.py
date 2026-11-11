"""Swing mechanics, reward history, observation isolation and real SynReal smoke checks."""

import json
import math
import os

import newton
import pytest
import torch
import warp as wp

from examples.microduck.swing import SwingEnv, checkpoint_metadata, load_config
from examples.microduck.swing_scene import apply_cords, build_swing_robot, cord_tension, mechanism_state
from examples.microduck.swing_task import SwingObjective


def state(count=2, angle=0.0):
    cfg = load_config()
    g = cfg["geometry"]
    q = torch.tensor([[0.0, math.sin(angle / 2), 0.0, math.cos(angle / 2)]]).repeat(count, 1)
    anchor = torch.tensor([0.0, 0.0, g["anchor_height"]])
    from examples.microduck.adapter import rotate

    offset = rotate(q, torch.tensor([[0.0, 0.0, g["attachment_z"] + g["hang_length"]]]).repeat(count, 1))
    position = anchor - offset
    return mechanism_state(position, q, torch.zeros(count, 3), torch.zeros(count, 3), g)


def test_asset_keeps_seat_mass_and_seated_free_joints():
    cfg = load_config()
    robot, qi, _ = build_swing_robot(newton.ModelBuilder, cfg["geometry"], cfg["joint_reference"])
    assert sum(robot.body_mass) == pytest.approx(0.895243181)
    assert [robot.joint_q[i] for i in qi] == cfg["joint_reference"]
    assert robot.joint_q[:7] == pytest.approx([0, 0, 0.2828, 0, 0, 0, 1])
    assert len(qi) == 14


def test_cords_never_push_and_apply_torque_about_com():
    g = load_config()["geometry"]
    length = torch.tensor([0.35, 0.38, 0.3822, 0.40])
    assert cord_tension(length, g).tolist() == pytest.approx([0, 0, 4.4, 140], abs=1e-3)
    wp.init()
    poses = wp.array([[0, 0, 0.2828, 0, 0, 0, 1], [0, 0, 0.40, 0, 0, 0, 1]], dtype=wp.transform, device="cpu")
    com = wp.array([[0.01, 0, 0], [0.01, 0, 0]], dtype=wp.vec3, device="cpu")
    forces = wp.zeros(2, dtype=wp.spatial_vector, device="cpu")
    wp.launch(apply_cords, 2, inputs=[poses, com, forces, 1, 0, g["attachment_y"], g["attachment_z"],
              g["anchor_height"], g["length"], g["stiffness"], g["limit"], g["limit_stiffness"]], device="cpu")
    result = forces.numpy()
    assert result[0] == pytest.approx([0, 0, 8.8, 0, 0.088, 0], abs=1e-3)
    assert result[1] == pytest.approx([0] * 6)


def test_plane_cue_ignores_pure_pitch_and_tracks_yaw():
    assert torch.allclose(state(angle=0.8)["cue"], torch.zeros(2, 3), atol=1e-7)
    cfg = load_config()["geometry"]
    q = torch.tensor([[0, 0, math.sin(0.2), math.cos(0.2)]])
    s = mechanism_state(torch.tensor([[0, 0, 0.2828]]), q, torch.zeros(1, 3), torch.zeros(1, 3), cfg)
    assert s["cue"][0, 1].item() == pytest.approx(-math.sin(0.4))


def test_frontiers_pay_once_on_each_side():
    objective = SwingObjective(2, "cpu", load_config())
    # Rotate the complete suspension about its anchors, preserving both cord lengths.
    s = state(angle=0.2)
    first = objective.update(s, torch.zeros(2), 0.02)["swing_peak_progress"]
    assert (first > 0).all()
    assert (objective.update(s, torch.zeros(2), 0.02)["swing_peak_progress"] == 0).all()
    assert (objective.update(state(angle=-0.2), torch.zeros(2), 0.02)["swing_peak_progress"] > 0).all()
    assert objective.peaks.sum(-1).tolist() == pytest.approx([0.4, 0.4])


def test_invalid_latch_and_partial_reset_preserve_other_world():
    objective = SwingObjective(2, "cpu", load_config())
    s = state(angle=0.2)
    s["lateral"][0] = 0.03
    for _ in range(2):
        terms = objective.update(s, torch.ones(2), 0.02)
    assert objective.invalid.tolist() == [True, False]
    assert terms["swing_height"][0] == 0 and terms["invalid_episode_debt"][0] > 0
    recovered = objective.update(state(angle=0.3), torch.ones(2), 0.02)
    assert recovered["swing_height"][0] == 0 and recovered["invalid_episode_debt"][0] == 1
    before = objective.peaks[1].clone()
    objective.reset(torch.tensor([0]))
    assert not objective.invalid.any()
    assert torch.equal(before, objective.peaks[1])
    assert not objective.peaks[0].any()


def test_penalty_signs_and_weight_curriculum():
    cfg = load_config()
    objective = SwingObjective(2, "cpu", cfg)
    terms = {name: torch.ones(2) for name in cfg["rewards"]}
    initial = objective.weighted(terms, 0)
    later = objective.weighted(terms, 4000 * 48 + 1)
    assert initial["swing_peak_progress"][0] == 224
    assert later["swing_peak_progress"][0] == 160
    for name in ("invalid_episode_debt", "string_slack", "action_rate_l2", "dof_pos_limits"):
        assert (initial[name] < 0).all() and (later[name] < 0).all()


def test_walking_checkpoint_is_rejected(tmp_path):
    metadata = {"task": "velocity", "observation_contract": "official_asymmetric_pd_v2"}
    (tmp_path / "config.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="incompatible"):
        checkpoint_metadata(tmp_path / "model_0.pt")


def test_pretrained_adapter_matches_rsl_normalization_and_freezes_statistics(tmp_path):
    from rsl_rl.modules import EmpiricalNormalization

    from examples.microduck.swing_policy import PretrainedSwingPolicy

    normalizer = EmpiricalNormalization(61)
    normalizer(torch.randn(8, 61) * 3 + 2)
    normalizer.eval()
    network = torch.nn.Sequential(torch.nn.Linear(61, 512), torch.nn.ELU(), torch.nn.Linear(512, 256),
                                  torch.nn.ELU(), torch.nn.Linear(256, 128), torch.nn.ELU(), torch.nn.Linear(128, 14))
    state = {f"mlp.{key}": value for key, value in network.state_dict().items()}
    state.update({f"obs_normalizer.{key}": value for key, value in normalizer.state_dict().items()})
    path = tmp_path / "upstream.pt"
    torch.save({"actor_state_dict": state}, path)
    policy = PretrainedSwingPolicy(path)
    before = {key: value.clone() for key, value in policy.state_dict().items()}
    observations = torch.randn(5, 61) * 10
    with torch.no_grad():
        assert torch.equal(policy(observations), network(normalizer(observations)).clamp(-1, 1))
        assert torch.equal(policy(observations), policy(observations))
    assert not any(p.requires_grad for p in policy.parameters())
    assert all(torch.equal(value, policy.state_dict()[key]) for key, value in before.items())


@pytest.mark.skipif(os.environ.get("MICRODUCK_GPU_TESTS") != "1", reason="Opt-in SynReal physics")
def test_synreal_observations_actions_manual_reset_and_world_isolation():
    cfg = load_config()
    cfg["episode_s"] = 0.04
    env = SwingEnv(cfg, num_envs=2, auto_reset=False)
    try:
        obs, extras = env.get_observations()
        assert obs.shape == (2, 61) and extras["observations"]["critic"].shape == (2, 68)
        assert not obs[:, 6:34].any() and not obs[:, 48:].any()
        for _ in range(3):
            obs, reward, done, extras = env.step(torch.full((2, 14), 2.0, device=env.device))
        assert done.all() and env.total_resets == 0 and env.episode_length_buf.tolist() == [3, 3]
        assert torch.isfinite(reward).all() and (obs[:, 34:48] == 1).all()
        assert torch.allclose(env.actions, torch.full_like(env.actions, 0.7))
        assert not obs[:, 51:].any()
        # Privileged frontier/validity cannot leak into the actor's command slots.
        actor = obs.clone()
        env.objective.peaks += 1
        env.objective.invalid[:] = True
        env.observation_cache = None
        assert torch.equal(env.get_observations()[0], actor)
        before = [wp.to_torch(s.body_q).reshape(2, -1, 7)[1].clone() for s in env.states]
        env._reset(torch.tensor([0], device=env.device))
        for s, expected in zip(env.states, before, strict=True):
            assert torch.equal(wp.to_torch(s.body_q).reshape(2, -1, 7)[1], expected)
        assert env.episode_length_buf.tolist() == [0, 3]
        assert env.objective.invalid.tolist() == [False, True]
    finally:
        env.close()


@pytest.mark.parametrize("solver", ["synreal", "mujoco"])
def test_unified_cli_supports_training_resume_and_playback(solver, tmp_path):
    from examples.microduck.swing import parse_args

    checkpoint = tmp_path / "model_1.pt"
    train = parse_args(["train", "--solver", solver, "--checkpoint", str(checkpoint), "--iterations", "3"])
    assert train.solver == solver and train.iterations == 3 and train.viewer == "null"
    assert train.checkpoint == checkpoint and train.num_envs == 8
    play = parse_args(["play", "--solver", solver, "--checkpoint", str(checkpoint)])
    assert play.viewer == "gl" and play.num_envs == 1
    assert play.device == ("cpu" if solver == "mujoco" else "cuda:0")
    assert parse_args(["play", "--solver", solver, "--pretrained"]).pretrained


@pytest.mark.parametrize("arguments", [
    ["play", "--solver", "mujoco"],
    ["train", "--solver", "mujoco", "--pretrained"],
    ["train", "--solver", "mujoco", "--config", "swing.json"],
    ["train", "--solver", "mujoco", "--viewer", "gl"],
    ["train", "--solver", "mujoco", "--report", "frames.json"],
    ["train", "--solver", "synreal", "--allow-source-mismatch"],
    ["play", "--pretrained", "--checkpoint", "model.pt"],
])
def test_unified_cli_rejects_unsupported_or_conflicting_options(arguments):
    from examples.microduck.swing import parse_args

    with pytest.raises(SystemExit) as error:
        parse_args(arguments)
    assert error.value.code == 2


def test_checkpoints_cannot_cross_backends(tmp_path):
    from examples.microduck.swing import mujoco_checkpoint_metadata

    path = tmp_path / "model_0.pt"
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"task": "swing", "solver": "mujoco"}))
    with pytest.raises(ValueError, match="incompatible"):
        checkpoint_metadata(path)
    config.write_text(json.dumps({"task": "swing", "solver": "synreal"}))
    with pytest.raises(ValueError, match="incompatible"):
        mujoco_checkpoint_metadata(path)


def test_synreal_cpu_has_a_clear_device_error():
    from examples.microduck.swing import parse_args, run_synreal

    args = parse_args(["preview", "--device", "cpu", "--viewer", "null"])
    with pytest.raises(ValueError, match="SynReal requires a CUDA device"):
        run_synreal(args)
