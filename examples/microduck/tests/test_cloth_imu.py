"""Physical accelerometer semantics, versioned playback and reset isolation."""

import copy
import json
import math
import xml.etree.ElementTree as ET

import pytest
import torch

from examples.microduck.cloth_imu import ClothAccelerometer, site_velocity
from examples.microduck.cloth_policy import (
    LEGACY_SPEC,
    SPEC,
    V2_SPEC,
    ClothHistoryActorCritic,
    ClothNormalization,
)
from examples.microduck.observations import configure_policy_observations
from examples.microduck.policy_spec import ASSET, EFFORT, KD, KP


def sensor(count=1):
    return ClothAccelerometer(count, "cpu", SPEC["imu_acceleration"])


def test_imu_site_matches_robot_asset():
    site = ET.parse(ASSET).find(".//site[@name='imu']")
    assert [float(x) for x in site.attrib["pos"].split()] == SPEC["imu_acceleration"]["site_position_body_m"]
    quat = [float(x) for x in site.attrib["quat"].split()]
    assert quat[1:] + quat[:1] == SPEC["imu_acceleration"]["site_quaternion_xyzw"]


def test_static_freefall_translation_and_rotated_sensor():
    imu = sensor()
    zero = torch.zeros(1, 3)
    gravity = torch.tensor([[0., 0., -9.81]])
    identity = torch.tensor([[0., 0., 0., 1.]])
    imu.sample(zero, zero, identity, gravity, 0.005)
    torch.testing.assert_close(imu.acceleration, torch.tensor([[0., 0., 1.]]))
    imu.advance()
    imu.sample(zero, gravity * 0.005, identity, gravity, 0.005)
    torch.testing.assert_close(imu.acceleration, zero)
    imu.advance()
    imu.sample(zero, torch.tensor([[9.81, 0., 0.]]) * 0.005, identity, gravity, 0.005)
    torch.testing.assert_close(imu.acceleration, torch.tensor([[1., 0., 1.]]))
    imu.advance()
    pitch90 = torch.tensor([[0., math.sqrt(0.5), 0., math.sqrt(0.5)]])
    imu.sample(zero, zero, pitch90, gravity, 0.005)
    torch.testing.assert_close(imu.acceleration, torch.tensor([[-1., 0., 0.]]), atol=1e-6, rtol=0)


def test_rotating_offset_site_detects_centripetal_acceleration():
    dt, omega = 0.0001, 2.0
    before_q = torch.tensor([[0., 0., 0., 1.]])
    after_q = torch.tensor([[0., 0., math.sin(omega * dt / 2), math.cos(omega * dt / 2)]])
    com_velocity, com = torch.zeros(1, 3), torch.zeros(1, 3)
    angular = torch.tensor([[0., 0., omega]])
    site = torch.tensor([1., 0., 0.])
    before = site_velocity(com_velocity, angular, before_q, com, site)
    after = site_velocity(com_velocity, angular, after_q, com, site)
    imu = sensor()
    imu.sample(before, after, after_q, torch.zeros(1, 3), dt)
    torch.testing.assert_close(imu.acceleration * 9.81, torch.tensor([[-4., 0., 0.]]), atol=0.002, rtol=0)


def test_filter_reset_and_clipping_are_per_environment():
    imu = sensor(2)
    zero = torch.zeros(2, 3)
    quat = torch.tensor([[0., 0., 0., 1.]]).repeat(2, 1)
    imu.sample(zero, torch.ones(2, 3) * 9.81 * 0.01, quat, zero, 0.01)
    imu.sample(zero, torch.ones(2, 3) * 3 * 9.81 * 0.01, quat, zero, 0.01)
    torch.testing.assert_close(imu.acceleration, torch.full((2, 3), 2.))
    imu.reset(torch.tensor([0]))
    assert not imu.acceleration[0].any()
    torch.testing.assert_close(imu.acceleration[1], torch.full((3,), 2.))
    imu.advance()
    imu.sample(zero, torch.ones(2, 3) * 1e6, quat, zero, 0.01)
    assert (imu.acceleration == 16).all()


@pytest.mark.parametrize("version", [1, 2, 3])
def test_versioned_checkpoint_playback_configuration(tmp_path, version):
    from examples.microduck.common import setup
    from examples.microduck.env import MicroDuckEnv
    from examples.microduck.observations import CONTRACT
    from examples.microduck.ppo import ppo_config
    from examples.microduck.recipe import RECIPE, recipe_fingerprint
    from examples.microduck.scenes import load_policy_checkpoint

    spec = copy.deepcopy({1: LEGACY_SPEC, 2: V2_SPEC, 3: SPEC}[version])
    actor_dim, _critic_dim, a_size, c_size = (71, 80, 1231, 1240) if version == 1 else (74, 83, 1294, 1303)
    cfg = ppo_config(1, "cloth_history")
    cfg["policy"].update(actor_hidden_dims=[16], critic_hidden_dims=[16])
    options = {key: value for key, value in cfg["policy"].items() if key != "class_name"}
    policy = ClothHistoryActorCritic(a_size, c_size, 14, **options)
    norm = ClothNormalization(actor_dim).eval()
    metadata = {
        "solver": "synreal", "task": "velocity", "recipe": RECIPE, "observation_contract": CONTRACT,
        "actuator": "position_pd", "self_collisions": False, "kp": KP, "kd": KD, "effort_limit": EFFORT,
        "policy_dt": 0.02, "physics_dt": 0.005, "policy_variant": "cloth_history", "history": spec,
        "terrain": {"kind": "cloth"}, "ppo": cfg,
        "official_alignment": {"actor_dim": 61, "critic_dim": 70,
                               "recipe_fingerprint": recipe_fingerprint("velocity", "cloth")},
    }
    (tmp_path / "config.json").write_text(json.dumps(metadata))
    path = tmp_path / "model_0.pt"
    torch.save({"model_state_dict": policy.state_dict(), "obs_norm_state_dict": norm.state_dict(), "iter": 0}, path)
    loaded = load_policy_checkpoint(path, "cpu")
    obs = torch.randn(2, a_size)
    torch.testing.assert_close(loaded.actor.act_inference(loaded.normalizer(obs)), policy.act_inference(norm(obs)))
    args, _, _ = setup("play", ["--solver", "synreal", "--checkpoint", str(path)])
    assert args.checkpoint_history == spec
    # Use real CPU controller/observations while avoiding any claim of soft-body physics validation.
    env = MicroDuckEnv(device="cpu", num_envs=1)
    try:
        env.cfg["terrain"] = {"kind": "cloth"}
        configure_policy_observations(env, args.policy_variant, args.checkpoint_history)
        assert env.num_obs == a_size
        assert env.num_privileged_obs == c_size
        if version == 1:
            assert env.official.observations.history.imu is None
        env.step(torch.zeros(1, 14), auto_reset=False)
        actual, _ = env.get_observations()
        assert torch.isfinite(loaded.actor.act_inference(loaded.normalizer(actual))).all()
    finally:
        env.close()


def test_acceleration_symmetry_is_polar_and_preserves_time_order():
    from examples.microduck.ppo import microduck_symmetry

    obs = torch.zeros(1, 1294)
    obs[:, 71:74] = torch.tensor([1., 2., 3.])
    past = obs[:, 74:].reshape(1, 20, 61)
    past[:, :, 58:61] = torch.arange(60).reshape(1, 20, 3)
    reflected, _ = microduck_symmetry(None, obs)
    torch.testing.assert_close(reflected[1:, 71:74], torch.tensor([[1., -2., 3.]]))
    torch.testing.assert_close(reflected[1:, 74:].reshape(1, 20, 61)[:, :, 58:61],
                               past[:, :, 58:61] * torch.tensor([1., -1., 1.]))


def test_nonfinite_sensor_input_is_rejected_before_clipping():
    imu = sensor()
    with pytest.raises(FloatingPointError, match="specific force"):
        imu.sample(torch.zeros(1, 3), torch.full((1, 3), float("inf")),
                   torch.tensor([[0., 0., 0., 1.]]), torch.zeros(1, 3), 0.005)
