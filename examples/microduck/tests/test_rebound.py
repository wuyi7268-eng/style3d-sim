"""Continuous reward accounting and sensor-only actor regression tests."""

import pytest
import torch

from examples.microduck.jump import (
    CONTRACT,
    CURRENT_FRAME_CONTRACT,
    JumpActorCritic,
    JumpEnv,
    load_config,
    runner_config,
)
from examples.microduck.rebound import ReboundCycle, sensor_frame, specific_force
from examples.microduck.tests.test_jump import sample


def make_cycle(count=1):
    c = ReboundCycle(count, 'cpu', load_config()['task'])
    c.reset(torch.arange(count), None, torch.full((count,), .55))
    for _ in range(11):
        c.advance(sample(count), .02)
    assert (c.phase == 1).all()
    return c


def jump(c, height=.12):
    launches = 0
    for _ in range(2):
        terms, _, failure = c.advance(sample(contact=False, com_z=.58, com_vz=1., foot_clearance=.025), .02)
        launches += terms['launch'].item()
        assert not failure.any()
    assert launches == 1
    for _ in range(5):
        c.advance(sample(contact=False, com_z=.55 + height, com_vz=.2, foot_clearance=.08), .02)
    c.advance(sample(contact=False, com_z=.55 + height, com_vz=-.1, foot_clearance=.08), .02)
    for _ in range(2):
        terms, success, failure = c.advance(sample(speed=.8, com_vz=-.5, leg_height=.10), .02)
    assert success.all() and not failure.any()
    return terms


def test_repeated_dynamic_landings_keep_reference_and_clock():
    c = make_cycle()
    first = jump(c)
    assert first['continuity'].item() == 0
    assert c.phase.item() == 3
    before_age, reference = c.age.clone(), c.reference.clone()
    second = jump(c, .10)
    assert second['continuity'].item() == pytest.approx(2.)
    assert c.bounce_count.item() == 2
    assert (c.age > before_age).all()
    torch.testing.assert_close(c.reference, reference)
    assert c.contact_to_takeoff.item() < .2


def test_failure_refunds_only_current_flight_once():
    c = make_cycle()
    jump(c)
    # A landed flight remains credited even if the next support period fails.
    terms, _, failure = c.advance(sample(bad_contact=True), .02)
    assert failure.item() and terms['apex_progress'].item() == 0
    terms, _, failure = c.advance(sample(bad_contact=True), .02)
    assert not failure.item() and all(v.item() == 0 for v in terms.values())
    c = make_cycle()
    for _ in range(4):
        c.advance(sample(contact=False, com_z=.65, com_vz=.5, foot_clearance=.1), .02)
    terms, _, failure = c.advance(sample(bad_contact=True), .02)
    assert failure.item() and terms['apex_progress'].item() == pytest.approx(-2.)


def test_grazing_contact_does_not_create_second_takeoff_or_bonus():
    c = make_cycle()
    for _ in range(5):
        c.advance(sample(contact=False, com_z=.65, com_vz=.5, foot_clearance=.1), .02)
    c.advance(sample(contact=[[True, False]]), .02)
    for _ in range(3):
        terms, success, failure = c.advance(sample(contact=False, com_z=.65, com_vz=.3, foot_clearance=.1), .02)
        assert not success.item() and not failure.item()
        assert terms['launch'].item() == terms['continuity'].item() == 0
    assert c.phase.item() == 2


def test_stationary_crouching_cannot_earn_flight_or_repeat_shaping():
    c = make_cycle()
    total = 0
    for i in range(140):
        terms, success, failure = c.advance(sample(leg_height=.08 if i % 2 else .12, bed_z=.37), .02)
        total += terms['crouch'].item()
        assert not success.item()
        assert terms['continuity'].item() == terms['apex_progress'].item() == 0
        if failure.item():
            break
    assert failure.item() and total <= 1.0001


def test_selected_reset_preserves_other_world_accounting():
    c = make_cycle(2)
    c.last_height[:] = .12
    c.reset(torch.tensor([0]), None, torch.tensor([.54]))
    assert c.last_height.tolist() == pytest.approx([0, .12])
    assert c.phase.tolist() == [0, 1]


def test_specific_force_at_rest_freefall_and_rotated_sensor():
    zero = torch.zeros(1, 3)
    quat = torch.tensor([[0., 0., 0., 1.]])
    torch.testing.assert_close(specific_force(zero, zero, quat, .02), torch.tensor([[0., 0., 9.81]]))
    torch.testing.assert_close(specific_force(zero, torch.tensor([[0., 0., -.1962]]), quat, .02), zero)
    quat = torch.tensor([[1., 0., 0., 0.]])
    torch.testing.assert_close(specific_force(zero, zero, quat, .02), torch.tensor([[0., 0., -9.81]]))


def test_sensor_frame_never_reads_privileged_fields():
    from types import SimpleNamespace

    s = SimpleNamespace(angular=torch.zeros(1, 3), gravity=torch.tensor([[0., 0., -1.]]),
                        q=torch.zeros(1, 14), qd=torch.zeros(1, 14))
    frame = sensor_frame(s, torch.tensor([[0., 0., 9.81]]), torch.zeros(1, 14))
    assert frame.shape == (1, 51)
    assert frame[0, 8] == 1


@pytest.mark.parametrize("history_length", [0, 40])
def test_gpu_sensor_observation_isolation_timeout_and_checkpoint(tmp_path, monkeypatch, history_length):
    import os

    from examples.microduck.jump import create_runner, login, runner_config

    if os.getenv('MICRODUCK_GPU_TESTS') != '1' or not torch.cuda.is_available():
        pytest.skip('Opt-in CUDA test')
    login()
    cfg = load_config()
    cfg['task']['history_length'] = history_length
    cfg['contract'] = CONTRACT if history_length else CURRENT_FRAME_CONTRACT
    env = JumpEnv(cfg, 2, auto_reset=False)
    try:
        obs, extras = env.get_observations()
        assert obs.shape == (2, 51 * (1 + history_length))
        assert extras['observations']['critic'].shape == (2, 87 + 51 * history_length)
        assert (env.history is None) == (history_length == 0)
        for _ in range(15):
            previous_obs, snapshot = obs, obs.clone()
            obs, reward, _, _ = env.step(torch.zeros(2, 14, device=env.device))
            torch.testing.assert_close(previous_obs, snapshot)
            assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
        before = obs.clone()
        env.cycle.phase[:] = 3
        env.cycle.reference += 10
        env._cache = None
        torch.testing.assert_close(env.get_observations()[0], before)
        history = env.history.frames.clone() if env.history is not None else None
        env._reset(torch.tensor([0], device=env.device))
        torch.testing.assert_close(env.get_observations()[0][1], before[1])
        if history is not None:
            torch.testing.assert_close(env.history.frames[1], history[1])
        assert env.acceleration[0].norm().item() == pytest.approx(9.81)
        env.reset()
        env.auto_reset = True
        env.episode_length_buf[:] = env.max_episode_length
        _, _, done, extras = env.step(torch.zeros(2, 14, device=env.device))
        assert done.all() and extras['time_outs'].all()
        assert not extras['jump']['failure'].any()
        assert env.episode_length_buf.tolist() == [0, 0]
        advance = env.cycle.advance

        def successful_landing(state, dt):
            terms, _, failure = advance(state, dt)
            return terms, torch.ones_like(failure), failure

        with monkeypatch.context() as patch:
            patch.setattr(env.cycle, 'advance', successful_landing)
            patch.setattr(env, '_reset', lambda ids: pytest.fail('A landing must not reset dynamics or history'))
            _, _, done, extras = env.step(torch.zeros(2, 14, device=env.device))
            assert extras['jump']['success'].all() and not done.any()
            assert env.episode_length_buf.tolist() == [1, 1]
        runner = create_runner(env, runner_config(cfg, 2))
        runner.logger_type = 'tensorboard'
        runner.save(str(tmp_path / 'model.pt'))
        runner.load(str(tmp_path / 'model.pt'))
    finally:
        env.close()


def test_current_frame_policy_has_no_encoders_and_supports_ppo_and_checkpoint(tmp_path):
    cfg = runner_config(load_config(), 2)['policy'].copy()
    cfg.pop('class_name')
    policy = JumpActorCritic(51, 87, 14, **cfg)
    assert not any('history' in name for name in policy.state_dict())
    actor = torch.randn(2, 51)
    critic = torch.randn(2, 87)
    assert policy.act(actor).shape == (2, 14)
    loss = -policy.get_actions_log_prob(policy.action_mean.detach()).mean() + policy.evaluate(critic).square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in policy.parameters())
    before = policy.act_inference(actor).detach().clone()
    # A plain MLP must not retain state from unrelated observations.
    policy.act_inference(torch.randn_like(actor))
    torch.testing.assert_close(policy.act_inference(actor), before)
    path = tmp_path / 'policy.pt'
    torch.save(policy.state_dict(), path)
    restored = JumpActorCritic(51, 87, 14, **cfg)
    restored.load_state_dict(torch.load(path, weights_only=True))
    torch.testing.assert_close(restored.act_inference(actor), before)
    legacy = JumpActorCritic(2091, 2127, 14, history_length=40, current_dim=51, critic_dim=87, frame_dim=51)
    with pytest.raises(RuntimeError, match='size mismatch'):
        restored.load_state_dict(legacy.state_dict())


@pytest.mark.parametrize('contract,length', [(CURRENT_FRAME_CONTRACT, 40), (CONTRACT, 0),
                                           (CURRENT_FRAME_CONTRACT, -1), (CURRENT_FRAME_CONTRACT, True)])
def test_observation_contract_rejects_wrong_history_length(tmp_path, contract, length):
    import json

    cfg = load_config()
    cfg['contract'] = contract
    cfg['task']['history_length'] = length
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='history_length'):
        load_config(path)


def test_history_warm_start_rejected_before_physics(monkeypatch, capsys):
    from examples.microduck import jump as module

    old = load_config()
    old['contract'] = CONTRACT
    old['task']['history_length'] = 40
    monkeypatch.setattr(module, 'checkpoint_metadata', lambda *a, **k: {'observation_contract': CONTRACT, 'jump': old})
    monkeypatch.setattr(module, 'login', lambda: pytest.fail('Reject incompatible policies before initializing physics'))
    with pytest.raises(SystemExit) as error:
        module.main(['train', '--init-from', 'old-history.pt'])
    assert error.value.code == 2
    assert 'different jump observation contract' in capsys.readouterr().err


def test_airborne_bonus_once_until_confirmed_landing():
    c = make_cycle()
    paid = 0
    for _ in range(12):
        terms, _, _ = c.advance(sample(contact=False, com_z=.60, com_vz=.4, foot_clearance=.03), .02)
        paid += terms['airborne'].item()
        assert terms['stall'].item() == 0
    assert paid == 1
    c.advance(sample(contact=[[True, False]]), .02)
    for _ in range(6):
        terms, _, _ = c.advance(sample(contact=False, com_z=.60, com_vz=.3, foot_clearance=.03), .02)
        assert terms['airborne'].item() == 0
    for _ in range(2):
        c.advance(sample(), .02)
    assert c.confirmed.item()
    paid = 0
    for _ in range(6):
        terms, _, _ = c.advance(sample(contact=False, com_z=.60, com_vz=.4, foot_clearance=.03), .02)
        paid += terms['airborne'].item()
    assert paid == 1


def test_airborne_bonus_rejects_short_low_and_unsafe_flights():
    for updates in ({'foot_clearance': .005}, {'tilt': .8}):
        c = make_cycle()
        for _ in range(12):
            state = {'contact': False, 'com_z': .60, 'com_vz': .4, 'foot_clearance': .03}
            state.update(updates)
            terms, _, _ = c.advance(sample(**state), .02)
            assert terms['airborne'].item() == 0
    c = make_cycle()
    for _ in range(3):
        terms, _, _ = c.advance(sample(contact=False, com_z=.60, com_vz=.4, foot_clearance=.03), .02)
        assert terms['airborne'].item() == 0


def test_stall_grace_ramp_and_contact_chatter_does_not_restart_clock():
    c = make_cycle()
    for _ in range(20):
        terms, _, _ = c.advance(sample(), .02)
        assert terms['stall'].item() == 0
    for _ in range(30):
        terms, _, _ = c.advance(sample(), .02)
    assert terms['stall'].item() > 0
    before = c.stall_age.clone()
    terms, _, _ = c.advance(sample(contact=False), .02)
    assert terms['stall'].item() == 0
    terms, _, _ = c.advance(sample(), .02)
    assert terms['stall'].item() > 0 and (c.stall_age >= before).all()
    for _ in range(6):
        terms, _, _ = c.advance(sample(contact=False, com_z=.6, com_vz=.4, foot_clearance=.03), .02)
    assert c.stall_age.item() == 0
    for _ in range(2):
        terms, _, _ = c.advance(sample(), .02)
        assert terms['stall'].item() == 0


def test_legacy_reward_recipe_does_not_emit_new_terms():
    cfg = load_config()['task'].copy()
    cfg.pop('reward_revision', None)
    c = ReboundCycle(1, 'cpu', cfg)
    c.reset(torch.tensor([0]), None, torch.tensor([.55]))
    terms, _, _ = c.advance(sample(), .02)
    assert 'airborne' not in terms and 'stall' not in terms


def test_new_reward_reset_isolation_and_failure_suppression():
    c = make_cycle(2)
    c.airborne_paid[:] = True
    c.stall_age[:] = 1.4
    c.reset(torch.tensor([0]), None, torch.tensor([.55]))
    assert c.airborne_paid.tolist() == [False, True]
    assert c.stall_age.tolist() == pytest.approx([0, 1.4])
    c = make_cycle()
    for _ in range(3):
        c.advance(sample(contact=False, com_z=.6, com_vz=.4, foot_clearance=.03), .02)
    terms, _, failure = c.advance(
        sample(contact=False, com_z=.6, com_vz=.4, foot_clearance=.03, bad_contact=True), .02)
    assert failure.item() and terms['airborne'].item() == terms['stall'].item() == 0
