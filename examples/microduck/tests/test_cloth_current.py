"""Current units, causal sampling, historical contracts and safe feedback migration."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from examples.microduck.checkpoint import initialize_transfer
from examples.microduck.cloth_current import (
    CURRENT_SPEC,
    hardware_leg_current,
    simulated_leg_current,
)
from examples.microduck.cloth_policy import (
    LEGACY_SPEC,
    SPEC,
    V2_SPEC,
    ClothHistory,
    ClothHistoryActorCritic,
    layout,
)
from examples.microduck.ppo import microduck_symmetry
from examples.microduck.task_profiles import load_task_spec


def test_current_matches_official_bus_units_and_joint_order():
    # Distinct values expose a mouth/neck inclusion or right-leg off-by-one error.
    amperes = torch.arange(14).float().reshape(1, 14) / 100
    torque = -amperes * CURRENT_SPEC["kt_nm_per_a"]
    bus = torch.cat((amperes[:, :9], torch.full((1, 1), 99.), amperes[:, 9:]), -1) * 1000
    expected = amperes[:, [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]]
    torch.testing.assert_close(simulated_leg_current(torque), expected)
    torch.testing.assert_close(hardware_leg_current(-bus), expected)
    torch.testing.assert_close(simulated_leg_current(torque), simulated_leg_current(-torque))
    assert not simulated_leg_current(torch.zeros(2, 14)).any()
    # Quantization is applied after taking magnitude, as on the bus.
    small = torch.full((1, 14), 0.0016 * CURRENT_SPEC["kt_nm_per_a"])
    torch.testing.assert_close(simulated_leg_current(small), torch.full((1, 10), .002))


@pytest.mark.parametrize("function,width", [(simulated_leg_current, 14), (hardware_leg_current, 15)])
def test_invalid_sensor_input(function, width):
    with pytest.raises(ValueError):
        function(torch.zeros(1, width - 1))
    with pytest.raises(FloatingPointError):
        function(torch.full((1, width), float("nan")))


def test_current_uses_latest_sample_and_preserves_other_environment_on_reset():
    history = ClothHistory(2, "cpu")
    actor = torch.zeros(2, 74)
    history.read(actor)
    history.advance()
    history.record_torque(torch.full((2, 14), .2))
    history.record_torque(torch.full((2, 14), -.1))
    expected = simulated_leg_current(torch.full((2, 14), -.1))
    torch.testing.assert_close(history.feedback, expected)
    actor[:, 61:71] = history.feedback
    history.read(actor)
    history.advance()
    history.record_torque(torch.full((2, 14), .4))
    actor[:, 61:71] = history.feedback
    past = history.read(actor).reshape(2, 20, 61).clone()
    torch.testing.assert_close(past[:, 0, 48:58], expected)
    assert not past[:, 1:, 48:58].any()
    before = history.feedback.clone()
    history.reset(torch.tensor([0]))
    actor[0, 61:74] = 0
    after = history.read(actor).reshape(2, 20, 61)
    assert not history.feedback[0].any()
    torch.testing.assert_close(history.feedback[1], before[1])
    torch.testing.assert_close(after[1], past[1])
    assert not after[0, :, 48:].any()


@pytest.mark.parametrize("spec", [LEGACY_SPEC, V2_SPEC, SPEC])
def test_feedback_symmetry_respects_saved_semantics(spec):
    actor_dim, _, frame, size = layout(spec["version"])
    obs = torch.ones(1, actor_dim + size)
    env = SimpleNamespace(cfg={"history": spec})
    reflected, _ = microduck_symmetry(env, obs)
    sign = 1 if spec["version"] == 3 else -1
    assert (reflected[1, 61:71] == sign).all()
    assert (reflected[1, actor_dim:].reshape(20, frame)[:, 48:58] == sign).all()


@pytest.mark.parametrize("spec", [LEGACY_SPEC, V2_SPEC, SPEC])
def test_transfer_drops_only_obsolete_signed_feedback(tmp_path, spec):
    actor, critic, _, size = layout(spec["version"])
    source = ClothHistoryActorCritic(actor + size, critic + size, 14, actor_hidden_dims=[16], critic_hidden_dims=[16])
    target = ClothHistoryActorCritic(1294, 1303, 14, actor_hidden_dims=[16], critic_hidden_dims=[16])
    state = copy.deepcopy(source.state_dict())
    path = tmp_path / "model_0.pt"
    torch.save({"model_state_dict": state, "iter": 0}, path)
    path.with_name("config.json").write_text(json.dumps({"history": spec}))
    runner = SimpleNamespace(device="cpu", alg=SimpleNamespace(policy=target), empirical_normalization=False)
    info = initialize_transfer(runner, path)
    assert info["feedback_reinitialized"] == (spec["version"] < 3)
    for branch, base in (("actor", 61), ("critic", 70)):
        torch.testing.assert_close(target.state_dict()[f"{branch}.0.weight"][:, :base], state[f"{branch}.0.weight"][:, :base])
        if spec["version"] < 3:
            assert not target.state_dict()[f"{branch}.0.weight"][:, base:base + 10].any()
            assert not target.state_dict()[f"{branch}_history_encoder.0.weight"].view(128, 20, 61)[:, :, 48:58].any()
        else:
            torch.testing.assert_close(target.state_dict()[f"{branch}.0.weight"], state[f"{branch}.0.weight"])


def test_bridge_retains_contact_quality_without_prescribed_gait():
    spec = load_task_spec("velocity", "elastic_bridge")[0]
    for name in ("air_time", "foot_clearance", "foot_swing_height", "gait_cadence", "gait_overdue",
                 "foot_swing_progress", "commanded_foot_placement"):
        assert spec["rewards"][name]["weight"] == 0
        assert not any(c["params"].get("reward_name") == name for c in spec["curricula"].values())
    for name in ("cloth_contact_slip", "cloth_nonfoot_contact", "action_rate_l2", "dof_pos_limits"):
        assert spec["rewards"][name]["weight"] < 0
