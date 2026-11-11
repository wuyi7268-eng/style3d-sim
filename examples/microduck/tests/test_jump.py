"""Jump contracts: event ordering, reward exploits, history and checkpoint isolation."""

from types import SimpleNamespace

import pytest
import torch

from examples.microduck.jump import (
    CRITIC_DIM,
    CURRENT_DIM,
    FRAME_DIM,
    JumpActorCritic,
    JumpCycle,
    JumpEnv,
    JumpHistory,
    checkpoint_metadata,
)
from examples.microduck.jump import (
    load_config as load_recipe,
)


def load_current_config():
    from pathlib import Path

    return load_recipe(Path(__file__).parents[1] / "configs/trampoline_jump_legacy_v4.json")


def load_config():
    """Exercise saved height-command semantics separately from the new default."""
    config = load_current_config()
    config["contract"] = "trampoline_jump_v1"
    config["task"].update(
        shaping_version=3, height_min=0.02, height_initial_max=0.05, height_max=0.15,
        height_increment=0.025, zero_probability=0.2, height_tolerance=0.015,
        height_relative_tolerance=0.15, curriculum_window=100, curriculum_success=0.8,
    )
    config["task"].pop("height_reward_scale")
    config["rewards"]["apex"] = 2.0
    return config


def sample(count=1, **updates):
    state = SimpleNamespace(
        contact=torch.ones(count, 2, dtype=torch.bool),
        com_z=torch.full((count,), 0.55),
        com_vz=torch.zeros(count),
        support_relative_speed=torch.zeros(count),
        leg_height=torch.full((count,), 0.12),
        leg_extension_speed=torch.zeros(count),
        tilt=torch.zeros(count),
        speed=torch.zeros(count),
        omega=torch.zeros(count),
        root_clearance=torch.full((count,), 0.12),
        drift=torch.zeros(count),
        bed_z=torch.full((count,), 0.4),
        bad_contact=torch.zeros(count, dtype=torch.bool),
        outside=torch.zeros(count, dtype=torch.bool),
        foot_clearance=torch.full((count, 2), 0.004),
    )
    for key, value in updates.items():
        setattr(state, key, torch.as_tensor(value).expand_as(getattr(state, key)).clone())
    return state


def cycle(target=0.05):
    c = JumpCycle(1, "cpu", load_config()["task"])
    c.reset(torch.tensor([0]), torch.tensor([target], dtype=torch.float32), torch.tensor([0.55]))
    for _ in range(11):
        c.advance(sample(), 0.02)
    assert c.phase.item() == 1
    return c


def flight(c):
    for _ in range(2):
        events, _, _ = c.advance(sample(contact=False, com_z=0.57, com_vz=0.8), 0.02)
    assert events["launch"].item() > 0
    assert c.phase.item() == 2
    events, _, _ = c.advance(sample(contact=False, com_z=0.60, com_vz=-0.01), 0.02)
    assert events["apex"].item() == pytest.approx(1)
    events, _, _ = c.advance(sample(contact=False, com_z=0.58, com_vz=-0.6), 0.02)
    assert events["apex"].item() == 0


def test_complete_jump_requires_stable_two_foot_landing():
    c = cycle()
    flight(c)
    events, success, failure = c.advance(sample(), 0.02)
    assert events["landing"].item() == pytest.approx(1)
    assert not success.item() and not failure.item()
    for _ in range(21):
        _, success, failure = c.advance(sample(), 0.02)
        if success.item():
            break
    assert success.item() and not failure.item()


def test_reference_locked_and_compression_cannot_replace_airborne_height():
    c = cycle()
    total = 0
    for _ in range(5):
        events, success, _ = c.advance(sample(bed_z=0.25, com_z=0.48), 0.02)
        total += events["compression"].item()
        assert events["apex"].item() == 0 and not success.item()
    assert total == pytest.approx(1)
    assert c.reference.item() == pytest.approx(0.55)
    assert c.peak.item() == pytest.approx(0.55)


def test_positive_command_cannot_succeed_by_standing():
    c = cycle()
    for _ in range(150):
        _, success, failure = c.advance(sample(), 0.02)
        assert not success.item()
        if failure.item():
            break
    assert failure.item()


def test_zero_command_can_stand_without_flight():
    c = cycle(0)
    for _ in range(21):
        _, success, failure = c.advance(sample(), 0.02)
        if success.item():
            break
    assert success.item() and not failure.item()


def test_single_foot_landing_and_body_contact_fail():
    c = cycle()
    flight(c)
    for _ in range(6):
        _, success, failure = c.advance(sample(contact=[[True, False]]), 0.02)
        assert not success.item()
    assert failure.item()
    c = cycle()
    _, success, failure = c.advance(sample(bad_contact=True), 0.02)
    assert failure.item() and not success.item()


def test_wrong_height_cannot_succeed_even_with_good_landing():
    c = cycle(0.15)
    for _ in range(2):
        c.advance(sample(contact=False, com_z=0.57, com_vz=0.8), 0.02)
    c.advance(sample(contact=False, com_z=0.60, com_vz=-0.01), 0.02)
    for _ in range(30):
        _, success, _ = c.advance(sample(), 0.02)
        assert not success.item()


def test_preparation_tracks_bed_motion_but_landing_stays_strict():
    c = JumpCycle(1, "cpu", load_config()["task"])
    c.reset(torch.tensor([0]), torch.tensor([0.05]), torch.tensor([0.55]))
    total = 0.0
    for _ in range(11):
        events, _, _ = c.advance(sample(speed=0.3, support_relative_speed=0.01), 0.02)
        total += events["prepare"].item()
    assert c.phase.item() == 1
    assert total == pytest.approx(1)
    flight(c)
    for _ in range(25):
        _, success, _ = c.advance(sample(speed=0.3, support_relative_speed=0.01), 0.02)
        assert not success.item()
    assert c.stable_time.item() == 0


def test_preparation_progress_cannot_be_farmed_by_repeated_balance_loss():
    c = JumpCycle(1, "cpu", load_config()["task"])
    c.reset(torch.tensor([0]), torch.tensor([0.05]), torch.tensor([0.55]))
    total = 0.0
    for _ in range(10):
        for _ in range(4):
            events, _, _ = c.advance(sample(), 0.02)
            total += events["prepare"].item()
        c.advance(sample(contact=False), 0.02)
    assert total == pytest.approx(0.4)
    assert c.phase.item() == 0


def test_preparation_reference_averages_support_window_and_times_out_without_support():
    c = JumpCycle(1, "cpu", load_config()["task"])
    c.reset(torch.tensor([0]), torch.tensor([0.05]), torch.tensor([0.55]))
    heights = []
    for i in range(12):
        z = 0.55 + i * 0.001
        heights.append(z)
        c.advance(sample(com_z=z), 0.02)
        if c.phase.item() == 1:
            break
    assert c.reference.item() == pytest.approx(sum(heights) / len(heights))
    c.reset(torch.tensor([0]), torch.tensor([0.05]), torch.tensor([0.55]))
    for _ in range(152):
        events, success, failure = c.advance(sample(contact=False), 0.02)
        if failure.item():
            break
    assert failure.item() and not success.item()
    assert events["prepare"].item() == 0


def test_push_requires_preload_and_active_extension_and_is_bounded():
    c = cycle()
    events, _, _ = c.advance(sample(com_vz=0.5, leg_extension_speed=0.2), 0.02)
    assert events["push"].item() == 0  # No preload yet.
    events, _, _ = c.advance(sample(bed_z=0.39, leg_height=0.11, com_vz=-0.2), 0.02)
    assert events["crouch"].item() > 0 and events["compression"].item() > 0
    events, _, _ = c.advance(sample(com_vz=0.5, leg_extension_speed=0), 0.02)
    assert events["push"].item() == 0  # Passive upward transport by the bed.
    total = 0.0
    for velocity in (0.2, 0.5, 0.1, 0.5, 0.8, 3.0, 0.5, 3.0):
        events, success, _ = c.advance(sample(com_vz=velocity, leg_extension_speed=0.2), 0.02)
        total += events["push"].item()
        assert not success.item()
    assert total == pytest.approx(1.0)
    assert c.phase.item() == 1  # Upward speed alone is not a flight.


def test_platform_jump_style_launch_shaping_precedes_strict_takeoff():
    """Launch velocity is dense; flight still requires genuine two-foot release."""
    c = cycle()
    c.advance(sample(bed_z=0.39, leg_height=0.11), 0.02)
    events, _, _ = c.advance(
        sample(
            com_z=0.565,
            com_vz=0.35,
            leg_extension_speed=0.12,
            foot_clearance=[[0.02, 0.021]],
            contact=True,
        ),
        0.02,
    )
    assert events["launch_velocity"].item() > 0
    assert events["apex_progress"].item() == 0
    assert c.phase.item() == 1
    events, _, _ = c.advance(
        sample(
            com_z=0.57,
            com_vz=0.35,
            leg_extension_speed=0.12,
            foot_clearance=[[0.02, 0.021]],
            contact=False,
        ),
        0.02,
    )
    assert events["apex_progress"].item() == 0
    events, _, _ = c.advance(sample(contact=False, com_z=0.58, com_vz=0.35), 0.02)
    assert events["apex_progress"].item() > 0
    assert c.phase.item() == 2


def test_v3_launch_is_bounded_without_preload_and_rejects_passive_rebound():
    c = cycle()
    events, _, _ = c.advance(sample(com_vz=0.5, leg_extension_speed=0), 0.02)
    assert events["launch_velocity"].item() == 0
    total = 0.0
    for speed in (0.1, 0.3, 0.1, 0.3, 0.5, 2.0, 0.5):
        events, _, _ = c.advance(sample(com_vz=speed, leg_extension_speed=speed), 0.02)
        total += events["launch_velocity"].item()
        assert events["push"].item() == 0  # v2 hard preload gate is still shut.
    assert total == pytest.approx(1)
    assert c.phase.item() == 1


@pytest.mark.parametrize("updates", [
    {"bad_contact": True}, {"outside": True}, {"contact": [[True, False]]},
    {"tilt": 0.7}, {"omega": 4.0}, {"drift": 0.2},
])
def test_v3_unsafe_motion_cannot_earn_launch(updates):
    c = cycle()
    events, _, _ = c.advance(sample(com_vz=0.5, leg_extension_speed=0.5, **updates), 0.02)
    assert events["launch_velocity"].item() == 0


def test_v3_flight_progress_is_bounded_and_refunded_once_on_failure():
    c = cycle()
    total = 0.0
    for z in (0.56, 0.57, 0.60, 0.58, 0.60, 0.65):
        events, _, _ = c.advance(sample(contact=False, com_z=z, com_vz=0.5), 0.02)
        total += events["apex_progress"].item()
    assert total == pytest.approx(1)
    events, _, failure = c.advance(sample(bad_contact=True), 0.02)
    assert failure.item()
    assert events["apex_progress"].item() == pytest.approx(-total)
    events, _, _ = c.advance(sample(bad_contact=True), 0.02)
    assert events["apex_progress"].item() == 0


def test_v3_landing_signal_requires_flight_and_paired_contact():
    c = cycle()
    events, _, _ = c.advance(sample(), 0.02)
    assert events["landing_stability"].item() == 0
    flight(c)
    events, _, _ = c.advance(sample(contact=[[True, False]]), 0.02)
    assert events["landing_stability"].item() == 0
    events, success, _ = c.advance(sample(), 0.02)
    assert events["landing_stability"].item() > 0
    assert not success.item()


def test_v3_partial_reset_clears_only_selected_reward_budgets():
    c = JumpCycle(2, "cpu", load_config()["task"])
    c.reset(torch.arange(2), torch.full((2,), 0.05), torch.full((2,), 0.55))
    for _ in range(11):
        c.advance(sample(2), 0.02)
    c.advance(sample(2, com_vz=0.5, leg_extension_speed=0.5), 0.02)
    for _ in range(2):
        c.advance(sample(2, contact=False, com_z=0.58, com_vz=0.5), 0.02)
    c.reset(torch.tensor([0]), torch.tensor([0.05]), torch.tensor([0.55]))
    for name in ("launch_progress", "height_progress", "height_paid"):
        assert getattr(c, name)[0] == 0
        assert getattr(c, name)[1] > 0


@pytest.mark.parametrize("version", [1, 2])
def test_saved_legacy_config_runs_without_v3_keys(version):
    config = load_config()
    config["task"]["shaping_version"] = version
    for name in ("launch_velocity_cap", "launch_tilt_deg", "launch_angular_speed"):
        config["task"].pop(name)
    for name in ("launch_velocity", "apex_progress", "landing_stability"):
        config["rewards"].pop(name)
    if version == 1:
        for name in ("prepare", "crouch", "push"):
            config["rewards"].pop(name)
        for name in ("prepare_relative_speed", "prepare_angular_speed", "prepare_tilt_deg", "prepare_timeout_s",
                     "crouch_target", "push_min_compression", "push_min_crouch", "push_extension_speed"):
            config["task"].pop(name)
    c = JumpCycle(1, "cpu", config["task"])
    c.reset(torch.tensor([0]), torch.tensor([0.05]), torch.tensor([0.55]))
    for _ in range(12):
        events, _, _ = c.advance(sample(), 0.02)
        assert not {"launch_velocity", "apex_progress", "landing_stability"}.intersection(events)
        assert all(name in config["rewards"] for name in events)


@pytest.mark.parametrize("updates", [
    {"bad_contact": True}, {"outside": True}, {"contact": [[True, False]]}, {"tilt": 0.5},
])
def test_unsafe_or_single_foot_motion_does_not_earn_preload(updates):
    c = cycle()
    events, success, _ = c.advance(sample(bed_z=0.37, leg_height=0.09, **updates), 0.02)
    assert events["compression"].item() == 0
    assert events["crouch"].item() == 0
    assert events["push"].item() == 0
    assert not success.item()


def test_zero_command_does_not_reward_crouching_or_pushing():
    c = cycle(0)
    c.advance(sample(bed_z=0.37, leg_height=0.09), 0.02)
    events, _, _ = c.advance(sample(com_vz=1, leg_extension_speed=0.3), 0.02)
    assert all(events[name].item() == 0 for name in ("compression", "crouch", "push", "launch",
                                                                "launch_velocity", "apex_progress", "landing_stability"))


def test_launch_speed_accounts_for_height_below_reference():
    c = cycle()
    velocity = (2 * 9.81 * 0.08) ** 0.5  # 3 cm below reference, with a 5 cm target.
    for _ in range(2):
        events, _, _ = c.advance(sample(contact=False, com_z=0.52, com_vz=velocity), 0.02)
    assert events["launch"].item() == pytest.approx(1)


def test_shaping_budgets_reset_only_selected_worlds():
    c = JumpCycle(2, "cpu", load_config()["task"])
    ids = torch.arange(2)
    c.reset(ids, torch.full((2,), 0.05), torch.full((2,), 0.55))
    for _ in range(11):
        c.advance(sample(2), 0.02)
    c.advance(sample(2, bed_z=0.37, leg_height=0.09), 0.02)
    c.advance(sample(2, com_vz=1, leg_extension_speed=0.3), 0.02)
    names = ("prepare_progress", "crouch", "push_progress", "reference_leg")
    before = {name: getattr(c, name).clone() for name in names}
    c.reset(torch.tensor([0]), torch.tensor([0.05]), torch.tensor([0.55]))
    for name in names:
        assert getattr(c, name)[0] == 0
        assert getattr(c, name)[1] == before[name][1]


def test_legacy_checkpoint_preserves_preparation_and_reward_semantics():
    cfg = load_config()["task"].copy()
    cfg.pop("shaping_version")
    c = JumpCycle(1, "cpu", cfg)
    c.reset(torch.tensor([0]), torch.tensor([0.05]), torch.tensor([0.55]))
    for _ in range(20):
        events, _, _ = c.advance(sample(speed=0.3, support_relative_speed=0.01), 0.02)
    assert c.phase.item() == 0
    assert not {"prepare", "crouch", "push"}.intersection(events)


def test_history_reads_do_not_advance_and_partial_reset_is_isolated():
    h = JumpHistory(2, 20, "cpu")
    x = torch.zeros(2, FRAME_DIM)
    h.read(x)
    h.advance()
    x += 1
    before = h.read(x).clone()
    torch.testing.assert_close(h.read(x), before)
    h.reset(torch.tensor([0]))
    x[0] = 7
    after = h.read(x)
    assert (after[0] == 7).all()
    torch.testing.assert_close(after[1], before[1])


def test_network_encoders_are_independent_and_phase_changes_actions():
    torch.manual_seed(42)
    h = 20 * FRAME_DIM
    policy = JumpActorCritic(CURRENT_DIM + h, CRITIC_DIM + h, 14, actor_hidden_dims=[32], critic_hidden_dims=[32])
    obs = torch.zeros(2, CURRENT_DIM + h)
    obs[1, FRAME_DIM] = 0.5
    actions = policy.act_inference(obs)
    assert not torch.allclose(actions[0], actions[1])
    actions.square().mean().backward()
    assert all(p.grad is not None for p in policy.actor_history.parameters())
    assert all(p.grad is None for p in policy.critic_history.parameters())
    assert policy.evaluate(torch.zeros(2, CRITIC_DIM + h)).shape == (2, 1)


def test_curriculum_requires_hard_height_success_and_request_is_latched():
    env = JumpEnv.__new__(JumpEnv)
    env.task_cfg = load_config()["task"]
    env.max_height = False
    env.fixed_height = None
    env.height_limit = 0.05
    from collections import deque

    env.outcomes = deque(maxlen=100)
    env.curriculum(torch.zeros(100), torch.ones(100, dtype=torch.bool))
    assert env.height_limit == 0.05
    env.curriculum(torch.full((100,), 0.05), torch.ones(100, dtype=torch.bool))
    assert env.height_limit == pytest.approx(0.075)
    env.cycle = cycle()
    env.set_height(0.1)
    assert env.cycle.target.item() == pytest.approx(0.05)
    with pytest.raises(ValueError):
        env.set_height(float("nan"))


def test_old_checkpoint_rejected_before_physics(tmp_path):
    import json

    path = tmp_path / "model.pt"
    path.touch()
    (tmp_path / "config.json").write_text(json.dumps({"task": "velocity", "observation_contract": "official_asymmetric_pd_v2"}))
    with pytest.raises(ValueError, match="Only trampoline_jump"):
        checkpoint_metadata(path)


def test_diagnostic_replay_allows_source_change_but_checks_materials(tmp_path, capsys):
    import json

    from examples.microduck.jump import CONTRACT, SCENE_CONFIG

    path = tmp_path / "model.pt"
    path.touch()
    metadata = {
        "task": "trampoline_jump",
        "observation_contract": CONTRACT,
        "source_fingerprint": "old-source",
        "terrain": {"config": vars(SCENE_CONFIG).copy()},
    }
    config = tmp_path / "config.json"
    config.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="source changed"):
        checkpoint_metadata(path)
    assert checkpoint_metadata(path, allow_source_mismatch=True) == json.loads(config.read_text())
    assert "diagnostic playback" in capsys.readouterr().out
    metadata["terrain"]["config"]["trampoline_height"] += 0.1
    config.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="materials/geometry"):
        checkpoint_metadata(path, allow_source_mismatch=True)


def test_source_override_rejected_for_training_before_physics(capsys):
    from examples.microduck.jump import main

    with pytest.raises(SystemExit) as error:
        main(["train", "--allow-source-mismatch"])
    assert error.value.code == 2
    assert "only supported for play" in capsys.readouterr().err


def test_gpu_cloth_contacts_reset_isolation_and_no_assistance():
    import os

    import numpy as np
    import warp as wp

    from examples.microduck.jump import login

    if os.getenv("MICRODUCK_GPU_TESTS") != "1" or not torch.cuda.is_available():
        pytest.skip("Opt-in SynReal CUDA test")
    login()
    env = JumpEnv(load_config(), num_envs=2, height=0)
    try:
        import newton

        from examples.microduck.policy_spec import ASSET, build_robot

        reference, _, _ = build_robot(task="velocity")
        collision_flags = int(newton.ShapeFlags.COLLIDE_SHAPES | newton.ShapeFlags.COLLIDE_PARTICLES)
        expected = [label.rsplit("/", 1)[-1] for label, flags in zip(reference.shape_label, reference.shape_flags)
                    if flags & collision_flags]
        actual = [label.rsplit("/", 1)[-1] for label, flags in zip(env.model.shape_label, env.model.shape_flags.numpy())
                  if flags & collision_flags and label.rsplit("/", 1)[-1] in expected]
        robot_colliders = (env.model.shape_body.numpy() >= 0) & ((env.model.shape_flags.numpy() & collision_flags) != 0)
        assert env.cfg["robot_asset"] == ASSET.name == "robot_walk.xml"
        assert len(expected) == 5
        assert sorted(actual) == sorted(expected * env.num_envs)
        assert int(robot_colliders.sum()) == len(expected) * env.num_envs
        seen = False
        for _ in range(10):
            obs, reward, done, _ = env.step(torch.zeros(2, 14, device=env.device))
            seen |= bool((env.sensors.feet.data.soft_found > 0).all())
            assert not done.any(), "Normal initial compression is not non-foot body contact"
            assert torch.isfinite(reward).all()
            assert obs.shape == (2, CURRENT_DIM + 1 + 20 * FRAME_DIM)
            np.testing.assert_array_equal(env.base.states[0].body_f.numpy(), 0)
        assert seen
        before_obs = env.get_observations()[0].clone()
        before = [
            {name: wp.to_torch(getattr(state, name)).clone() for name in ("body_q", "body_qd", "particle_q", "particle_qd")}
            for state in env.base.states
        ]
        history = env.history.frames.clone()
        env._reset(torch.tensor([0], device=env.device))
        for state, saved in zip(env.base.states, before, strict=True):
            for name, original in saved.items():
                worlds = wp.to_torch(env.model.body_world if name.startswith("body") else env.model.particle_world)
                torch.testing.assert_close(
                    wp.to_torch(getattr(state, name))[worlds == 1], original[worlds == 1], atol=0, rtol=0
                )
        torch.testing.assert_close(env.history.frames[1], history[1])
        torch.testing.assert_close(env.get_observations()[0][1], before_obs[1])
        env.set_height(0.08)
        assert env.cycle.target[1] == 0
        env._reset(torch.tensor([0], device=env.device))
        assert env.cycle.target[0].item() == pytest.approx(0.08)
        assert env.cycle.target[1] == 0
    finally:
        env.close()


def test_gpu_playback_keeps_terminal_worlds_until_manual_reset():
    import os

    import warp as wp

    from examples.microduck.jump import login

    if os.getenv("MICRODUCK_GPU_TESTS") != "1" or not torch.cuda.is_available():
        pytest.skip("Opt-in SynReal CUDA test")
    login()
    env = JumpEnv(load_config(), num_envs=2, height=0, auto_reset=False)
    actions = torch.zeros(2, 14, device=env.device)
    try:
        env.cycle.age[:] = env.task_cfg["episode_s"]
        obs, _, done, extras = env.step(actions)
        assert done.all() and extras["jump"]["failure"].all()
        assert env.episode_length_buf.tolist() == [1, 1]
        _, _, done, _ = env.step(actions)
        assert not done.any(), "A finished attempt must not be reported again every frame"
        assert env.episode_length_buf.tolist() == [2, 2]
        assert env.completed.all()
        env.set_height(0.08)
        obs, _ = env.reset()
        assert not env.completed.any()
        assert env.episode_length_buf.tolist() == [0, 0]
        assert not env.previous.any() and not env.base.actions.any()
        torch.testing.assert_close(env.cycle.target, torch.full((2,), 0.08, device=env.device))
        torch.testing.assert_close(
            obs[:, CURRENT_DIM + 1:].reshape(2, 20, FRAME_DIM), obs[:, None, :FRAME_DIM].expand(-1, 20, -1)
        )
        for state in env.base.states:
            particle_ids = env.terrain.cloth_indices
            torch.testing.assert_close(
                wp.to_torch(state.particle_q)[particle_ids], wp.to_torch(env.model.particle_q)[particle_ids]
            )
        # The same terminal signal must still reset both worlds in training mode.
        env.auto_reset = True
        env.cycle.age[:] = env.task_cfg["episode_s"]
        _, _, done, _ = env.step(actions)
        assert done.all()
        assert env.episode_length_buf.tolist() == [0, 0]
    finally:
        env.close()


def test_gpu_jump_mouse_drag_reaches_solver_and_release_clears_force(monkeypatch):
    import os

    import numpy as np
    import warp as wp
    from newton._src.viewer.picking import Picking
    from newton.viewer import ViewerGL

    from examples.microduck.jump import login

    if os.getenv("MICRODUCK_GPU_TESTS") != "1" or not torch.cuda.is_available():
        pytest.skip("Opt-in SynReal CUDA test")
    login()
    env = JumpEnv(load_config(), num_envs=2, height=0, auto_reset=False)
    offsets = wp.array([[0, 0, 0], [2, 0, 0]], dtype=wp.vec3, device=env.model.device)
    picking = Picking(env.model, world_offsets=offsets)
    viewer = SimpleNamespace(picking_enabled=True, picking=picking, wind=None, close=lambda: None)
    viewer.apply_forces = lambda state: ViewerGL.apply_forces(viewer, state)
    env.base.viewer = viewer
    monkeypatch.setattr(env.base, "render", lambda: None)
    solver_step = env.base.solver.step
    forces = []

    def record_step(state, *args):
        forces.append(state.body_f.numpy().copy())
        return solver_step(state, *args)

    monkeypatch.setattr(env.base.solver, "step", record_step)
    actions = torch.zeros(2, 14, device=env.device)
    try:
        # Use the actual viewer ray-picking and spring force, including display offsets for world 1.
        picking.pick(env.base.states[0], wp.vec3(2, 0, 1), wp.vec3(0, 0, -1))
        body = int(picking.pick_body.numpy()[0])
        assert body >= 0 and env.model.body_world.numpy()[body] == 1
        picking.update(wp.vec3(2.1, 0, 1), wp.vec3(0, 0, -1))
        env.step(actions)
        assert len(forces) == env.base.decimation
        assert forces[0][body, 0] > 0
        # Damping may reverse the spring force as the picked link accelerates.
        assert all(np.linalg.norm(force[body, :3]) > 0 for force in forces)
        assert all(np.count_nonzero(force[env.model.body_world.numpy() == 0]) == 0 for force in forces)
        dragged_velocity = env.base.states[0].body_qd.numpy().copy()
        forces.clear()
        picking.release()
        env.step(actions)
        assert len(forces) == env.base.decimation
        assert all(np.count_nonzero(force) == 0 for force in forces)
        env.reset()
        env.step(actions)
        assert np.linalg.norm(dragged_velocity - env.base.states[0].body_qd.numpy()) > 1e-3
        forces.clear()
        picking.pick(env.base.states[0], wp.vec3(2, 0, 1), wp.vec3(0, 0, -1))
        assert picking.is_picking()
        picking.update(wp.vec3(2.1, 0, 1), wp.vec3(0, 0, -1))
        env.auto_reset = True
        env.step(actions)
        assert len(forces) == env.base.decimation
        assert all(np.count_nonzero(force) == 0 for force in forces), "Training must not receive viewer forces"
    finally:
        env.close()


def maximum_cycle():
    c = JumpCycle(1, "cpu", load_current_config()["task"])
    c.reset(torch.tensor([0]), None, torch.tensor([0.55]))
    for _ in range(11):
        c.advance(sample(), 0.02)
    assert c.phase.item() == 1
    return c


def test_maximum_height_has_no_commands_or_curriculum():
    from examples.microduck.jump import MAX_HEIGHT_CONTRACT, main, runner_config

    config = load_current_config()
    assert config["contract"] == MAX_HEIGHT_CONTRACT
    removed = {"height_min", "height_max", "height_initial_max", "zero_probability", "curriculum_window"}
    assert not removed & config["task"].keys()
    policy = runner_config(config, 4)["policy"]
    assert policy["current_dim"] == 82 and policy["critic_dim"] == 87
    with pytest.raises(SystemExit):
        main(["train", "--height", "0.1"])
    env = JumpEnv.__new__(JumpEnv)
    env.max_height = True
    with pytest.raises(ValueError, match="no height command"):
        env.validate_height(0)
    env.validate_height(None)


def test_maximum_height_cycle_rearms_after_each_stable_landing():
    c = maximum_cycle()
    for _ in range(2):
        c.advance(sample(contact=False, com_z=0.57, com_vz=0.8), 0.02)
    c.advance(sample(contact=False, com_z=0.60, com_vz=-0.01), 0.02)
    success = False
    for _ in range(24):
        _, success_tensor, failure = c.advance(sample(), 0.02)
        success = bool(success_tensor.item())
        assert not failure.item()
        if success:
            break
    assert success
    c.rearm(torch.tensor([0]), torch.tensor([0.56]))
    assert c.phase.item() == 0
    assert c.age.item() == 0
    assert c.reference.item() == pytest.approx(0.56)
    assert c.target.item() == 0


def test_maximum_height_reward_keeps_increasing_beyond_old_cap_and_refunds_failure():
    c = maximum_cycle()
    c.advance(sample(contact=False, com_z=0.57, com_vz=0.8), 0.02)
    total = 0
    for z in (0.58, 0.60, 0.70, 0.85):
        events, _, failure = c.advance(sample(contact=False, com_z=z, com_vz=0.8), 0.02)
        assert not failure.item()
        assert events["apex_progress"].item() > 0
        total += events["apex_progress"].item()
    assert total == pytest.approx(0.30 / 0.05, abs=1e-5)
    for _ in range(3):
        events, _, _ = c.advance(sample(contact=False, com_z=0.80, com_vz=0.8), 0.02)
        assert events["apex_progress"].item() == 0
    events, _, failure = c.advance(sample(bad_contact=True), 0.02)
    assert failure.item() and events["apex_progress"].item() == pytest.approx(-total)
    events, _, _ = c.advance(sample(bad_contact=True), 0.02)
    assert events["apex_progress"].item() == 0


@pytest.mark.parametrize("rise", [0.02, 0.10, 0.25])
def test_maximum_height_any_genuine_jump_can_land_without_target_error(rise):
    c = maximum_cycle()
    for _ in range(2):
        events, _, _ = c.advance(sample(contact=False, com_z=0.55 + rise, com_vz=0.8), 0.02)
    assert events["launch"].item() == 1
    c.advance(sample(contact=False, com_z=0.55 + rise, com_vz=-0.01), 0.02)
    for _ in range(23):
        _, success, failure = c.advance(sample(), 0.02)
        if success.item():
            break
    assert success.item() and not failure.item()


def test_maximum_height_standing_and_bed_compression_are_not_flight():
    c = maximum_cycle()
    events, _, _ = c.advance(sample(com_vz=0.4, leg_extension_speed=0.4), 0.02)
    assert events["launch_velocity"].item() > 0
    assert events["launch"].item() == events["apex_progress"].item() == 0
    for _ in range(140):
        events, success, failure = c.advance(sample(bed_z=0.3, com_z=0.5), 0.02)
        assert not success.item() and events["apex_progress"].item() == 0
        if failure.item():
            break
    assert failure.item()


def test_gpu_maximum_height_observations_reset_and_checkpoint(tmp_path):
    import os

    from examples.microduck.jump import create_runner, login, runner_config

    if os.getenv("MICRODUCK_GPU_TESTS") != "1" or not torch.cuda.is_available():
        pytest.skip("Opt-in SynReal CUDA test")
    login()
    config = load_current_config()
    env = JumpEnv(config, num_envs=2, auto_reset=False)
    try:
        assert not hasattr(env, "height_limit") and not hasattr(env, "outcomes")
        actor, extras = env.get_observations()
        assert actor.shape == (2, CURRENT_DIM + 20 * FRAME_DIM)
        assert extras["observations"]["critic"].shape == (2, CRITIC_DIM + 20 * FRAME_DIM)
        before = actor.clone()
        env.cycle.target.fill_(999)
        env._cache = None
        torch.testing.assert_close(env.get_observations()[0], before)
        env.cycle.age[:] = env.task_cfg["episode_s"]
        _, reward, done, extras = env.step(torch.zeros(2, 14, device=env.device))
        assert done.all() and torch.isfinite(reward).all()
        assert "target" not in extras["jump"] and "Jump/height_limit" not in extras["log"]
        env._reset(torch.tensor([0], device=env.device))
        assert env.completed.tolist() == [False, True]
        runner = create_runner(env, runner_config(config, 2))
        runner.logger_type = "tensorboard"
        path = tmp_path / "model.pt"
        runner.save(str(path))
        infos = torch.load(path, map_location="cpu", weights_only=False)["infos"]
        assert infos == {"contract": config["contract"]}
        runner.load(str(path))
    finally:
        env.close()
