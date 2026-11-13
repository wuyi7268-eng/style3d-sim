"""Velocity-only cloth policy, versioned independently from the legacy 10-frame network."""

import weakref

import torch
from rsl_rl.modules import ActorCritic, EmpiricalNormalization
from torch import nn

from .cloth_current import CURRENT_SPEC, LEG_INDICES, simulated_leg_current

VARIANT = "cloth_history"
POLICY_CLASS = "ClothHistoryActorCritic"
LENGTH, FRAME_DIM, LATENT_DIM = 20, 61, 64
SIZE = LENGTH * FRAME_DIM
ACTOR_DIM, CRITIC_DIM = 74, 83
LEGACY_SPEC = {
    "version": 1,
    "length": LENGTH,
    "frame_dim": 58,
    "source": "actor_proprioception_48_then_leg_feedback_10",
    "order": "newest_first_excluding_current",
    "reset": "repeat_first_frame_zero_feedback",
    "encoder_dims": [128, LATENT_DIM],
    "shared_encoder": False,
    "normalization": "legacy_proprioception_stats_and_fixed_feedback_scale",
    "leg_indices": list(LEG_INDICES),
    "feedback": "mean_substep_saturated_pd_torque_div_effort_limit",
    "feedback_units": "dimensionless_load_proxy_not_measured_amperes",
}


V2_SPEC = {
    **LEGACY_SPEC,
    "version": 2,
    "frame_dim": FRAME_DIM,
    "source": "actor_proprioception_48_then_leg_feedback_10_then_imu_acceleration_3",
    "imu_acceleration": {
        "quantity": "specific_force",
        "site_position_body_m": [-0.021, 0.0000664098, -0.0146984],
        "site_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        "sampling": "site_world_velocity_difference_each_physics_substep",
        "filter": "control_interval_mean_in_sensor_axes",
        "scale_m_s2": 9.81,
        "clip_g": 16.0,
        "alignment": "same_fixed_imu_rotation_as_actor_gyro",
        "reset": "zero_until_first_complete_control_interval",
        "noise": "none_uncalibrated",
    },
}

SPEC = {
    **V2_SPEC,
    "version": 3,
    "feedback": "xl330_simulated_absolute_current",
    "feedback_units": "absolute_amperes_div_scale_a",
    "current_sensor": CURRENT_SPEC,
}
SUPPORTED_SPECS = (LEGACY_SPEC, V2_SPEC, SPEC)


def layout(version=3):
    if version not in (1, 2, 3):
        raise ValueError(f"Unsupported cloth history version: {version}")
    frame = 58 if version == 1 else FRAME_DIM
    extra = frame - 48
    return 61 + extra, 70 + extra, frame, LENGTH * frame


def supports_cloth_policy(task, terrain):
    return task == "velocity" and terrain in ("cloth", "hammock", "elastic_bridge")


class ClothHistory:
    """Store past frames and versioned leg telemetry from the completed control interval."""

    def __init__(self, count, device, version=3):
        self.version = version
        self.actor_dim, _, frame_dim, _ = layout(version)
        self.frames = torch.zeros(count, LENGTH, frame_dim, device=device)
        self.current = torch.zeros(count, frame_dim, device=device)
        self.fresh = torch.ones(count, dtype=torch.bool, device=device)
        self.feedback = torch.zeros(count, len(LEG_INDICES), device=device)
        self._torque_sum = torch.zeros_like(self.feedback)
        self._samples = 0
        self.imu = None
        if version >= 2:
            from .cloth_imu import ClothAccelerometer

            self.imu = ClothAccelerometer(count, device, SPEC["imu_acceleration"])

    def reset(self, ids):
        self.fresh[ids] = True
        self.frames[ids] = 0
        self.current[ids] = 0
        self.feedback[ids] = 0
        self._torque_sum[ids] = 0
        if self.imu is not None:
            self.imu.reset(ids)

    def advance(self):
        self.frames[:, 1:] = self.frames[:, :-1].clone()
        self.frames[:, 0] = self.current
        self._torque_sum.zero_()
        self._samples = 0
        self.feedback.zero_()
        if self.imu is not None:
            self.imu.advance()

    def record_torque(self, torque):
        from .policy_spec import EFFORT

        if self.version >= 3:
            # The bus reports a sample, not a signed average that cancels on torque reversal.
            self.feedback.copy_(simulated_leg_current(torque))
            return
        # Saved v1/v2 policies retain their original signed load proxy.
        self._torque_sum.add_(torque.detach()[:, LEG_INDICES] / EFFORT)
        self._samples += 1
        self.feedback.copy_(self._torque_sum / self._samples)

    def read(self, actor):
        frame = torch.cat((actor[:, :48], actor[:, 61:self.actor_dim]), -1)
        self.frames[self.fresh] = frame[self.fresh, None, :]
        self.fresh.zero_()
        self.current.copy_(frame)
        return self.frames.flatten(1)


class ClothHistoryActorCritic(ActorCritic):
    def __init__(self, num_actor_obs, num_critic_obs, num_actions, **kwargs):
        version = 1 if num_actor_obs == 1231 else 2
        self.actor_dim, self.critic_dim, self.frame_dim, self.history_size = layout(version)
        if (num_actor_obs, num_critic_obs, num_actions) != (
            self.actor_dim + self.history_size, self.critic_dim + self.history_size, 14
        ):
            raise ValueError("Invalid versioned cloth Velocity observation dimensions")
        super().__init__(self.actor_dim + LATENT_DIM, self.critic_dim + LATENT_DIM, num_actions, **kwargs)
        self.actor_history_encoder = nn.Sequential(nn.Linear(self.history_size, 128), nn.ELU(), nn.Linear(128, LATENT_DIM))
        self.critic_history_encoder = nn.Sequential(nn.Linear(self.history_size, 128), nn.ELU(), nn.Linear(128, LATENT_DIM))

    def actor_input(self, observations):
        current, history = observations.split((self.actor_dim, self.history_size), dim=-1)
        return torch.cat((current, self.actor_history_encoder(history)), -1)

    def update_distribution(self, observations):
        super().update_distribution(self.actor_input(observations))

    def act_inference(self, observations):
        return self.actor(self.actor_input(observations))

    def evaluate(self, critic_observations, **kwargs):
        current, history = critic_observations.split((self.critic_dim, self.history_size), dim=-1)
        return self.critic(torch.cat((current, self.critic_history_encoder(history)), -1))


class ClothNormalization(nn.Module):
    """Keep legacy statistics intact; feedback already has a fixed physical scale."""

    def __init__(self, current_dim, actor_normalizer=None):
        super().__init__()
        if current_dim not in (71, 80, ACTOR_DIM, CRITIC_DIM):
            raise ValueError("Unexpected cloth current observation dimension")
        version = 1 if current_dim in (71, 80) else 2
        _, _, self.frame_dim, self.history_size = layout(version)
        self.current_dim = current_dim
        self.base_dim = current_dim - (self.frame_dim - 48)
        self.current = EmpiricalNormalization(self.base_dim, until=1.0e8)
        self._actor_stats = weakref.ref(actor_normalizer.current) if actor_normalizer is not None else None

    def forward(self, observations):
        current, history = observations.split((self.current_dim, self.history_size), -1)
        normalized = self.current(current[:, :self.base_dim])
        stats = self.current if self._actor_stats is None else self._actor_stats()
        history = history.reshape(-1, LENGTH, self.frame_dim)
        proprioception = (history[:, :, :48] - stats._mean[:, None, :48]) / (stats._std[:, None, :48] + stats.eps)
        past = torch.cat((proprioception, history[:, :, 48:]), -1).flatten(1)
        return torch.cat((normalized, current[:, self.base_dim:], past), -1)


def migrate_weights(policy, source, *, source_history=None):
    """Expand old layouts; discard signed-load connections when migrating to absolute current."""
    if source_history is not None and source_history not in SUPPORTED_SPECS:
        raise ValueError("Unsupported source feedback contract")
    if source_history is not None and source_history["version"] < 3:
        # Absolute current cannot reconstruct a signed load. Preserve the other
        # learned features, but relearn this channel instead of silently reusing it.
        source = {name: value.clone() for name, value in source.items()}
        for branch, base in (("actor", 61), ("critic", 70)):
            source[f"{branch}.0.weight"][:, base:base + 10] = 0
            frame = source_history["frame_dim"]
            source[f"{branch}_history_encoder.0.weight"].view(128, LENGTH, frame)[:, :, 48:58] = 0
    target = policy.state_dict()
    if source.keys() == target.keys() and all(source[k].shape == target[k].shape for k in target):
        policy.load_state_dict(source, strict=True)
        return False
    if policy.frame_dim != FRAME_DIM:
        raise ValueError("Network expansion targets the latest cloth policy only")
    has_history = "actor_history_encoder.0.weight" in source
    old_cloth = has_history and source["actor_history_encoder.0.weight"].shape[1] == 1160
    for name, value in source.items():
        if name not in target:
            raise ValueError(f"Unsupported source policy tensor: {name}")
        if name in ("actor.0.weight", "critic.0.weight"):
            base = 61 if name.startswith("actor") else 70
            source_current = base + (10 if old_cloth else 0)
            expected = source_current + (LATENT_DIM if has_history else 0)
            if value.shape != (target[name].shape[0], expected):
                raise ValueError(f"Unsupported source input shape: {name} {value.shape}")
            target[name].zero_()
            target[name][:, :source_current] = value[:, :source_current]
            if has_history:
                target[name][:, base + 13:] = value[:, source_current:]
        elif name in ("actor_history_encoder.0.weight", "critic_history_encoder.0.weight"):
            old_length, old_frame = (20, 58) if old_cloth else (10, 48)
            if value.shape != (128, old_length * old_frame):
                raise ValueError("Unsupported source history encoder")
            target[name].zero_()
            # New channels are interleaved per frame, so map time and feature axes separately.
            target[name].view(128, LENGTH, FRAME_DIM)[:, :old_length, :old_frame] = value.reshape(128, old_length, old_frame)
        else:
            if value.shape != target[name].shape:
                raise ValueError(f"Incompatible source tensor: {name}")
            target[name] = value
    expected_keys = set(target)
    if not has_history:
        expected_keys = {k for k in expected_keys if "history_encoder" not in k}
    if set(source) != expected_keys:
        raise ValueError("Source policy is missing required tensors")
    policy.load_state_dict(target, strict=True)
    return True


def migrate_normalizer(normalizer, source):
    # Standard checkpoints store EmpiricalNormalization directly; history checkpoints wrap it as current.
    mapped = {key if key.startswith("current.") else f"current.{key}": value for key, value in source.items()}
    normalizer.load_state_dict(mapped, strict=True)
