"""Asymmetric observations with the processing configured in the upstream snapshot."""

import math

import torch
import warp as wp

from examples.locomotion.math_utils import quat_rotate_inverse

from .adapter import rotate

CONTRACT = "official_asymmetric_pd_v2"
HISTORY_LENGTH = 10
HISTORY_DIM = 48
HISTORY_SIZE = HISTORY_LENGTH * HISTORY_DIM
HISTORY_SPEC = {
    "version": 1,
    "length": HISTORY_LENGTH,
    "frame_dim": HISTORY_DIM,
    "source": "actor_prefix",
    "order": "newest_first_excluding_current",
    "reset": "repeat_first_frame",
    "encoder_dims": [128, 64],
    "shared_encoder": False,
    "normalization": "current_actor_prefix",
}


class ObservationHistory:
    """Past actor-visible frames, newest first; reads never advance the window."""

    def __init__(self, count, device):
        self.frames = torch.zeros(count, HISTORY_LENGTH, HISTORY_DIM, device=device)
        self.current = torch.zeros(count, HISTORY_DIM, device=device)
        self.fresh = torch.ones(count, dtype=torch.bool, device=device)

    def reset(self, ids):
        self.fresh[ids] = True
        self.frames[ids] = 0
        self.current[ids] = 0

    def advance(self):
        # Called once at the control-step boundary, before the next observation exists.
        self.frames[:, 1:] = self.frames[:, :-1].clone()
        self.frames[:, 0] = self.current

    def read(self, actor):
        frame = actor[:, :HISTORY_DIM]
        self.frames[self.fresh] = frame[self.fresh, None, :]
        self.fresh.zero_()
        self.current.copy_(frame)
        return self.frames.flatten(1)


def configure_policy_observations(env, variant, history_spec=None):
    """Keep base observation dimensions separate from the policy's transport layout."""
    if variant == "cloth_history":
        from .cloth_policy import (
            SPEC,
            SUPPORTED_SPECS,
            ClothHistory,
            layout,
            supports_cloth_policy,
        )

        if not supports_cloth_policy(env.task, env.cfg.get("terrain", {}).get("kind", "default")):
            raise ValueError("cloth_history is restricted to Velocity on cloth, hammock or elastic_bridge")
        history_spec = SPEC if history_spec is None else history_spec
        if history_spec not in SUPPORTED_SPECS:
            raise ValueError("Unsupported cloth history contract")
        version = history_spec["version"]
        actor_dim, critic_dim, _, size = layout(version)
        processor = env.official.observations
        if processor.critic_dim != 70:
            raise ValueError("Cloth history requires the base Velocity critic contract")
        if processor.history is None:
            processor.history = ClothHistory(env.num_envs, env.device, version)
        elif not isinstance(processor.history, ClothHistory) or processor.history.version != version:
            raise ValueError("Cannot switch the history contract of a live environment")
        env.num_obs, env.num_privileged_obs = actor_dim + size, critic_dim + size
        env.cfg.update(policy_variant=variant, history=dict(history_spec))
        return
    if variant not in ("standard", "history"):
        raise ValueError(f"Unknown policy variant: {variant}")
    processor = env.official.observations
    if variant == "history" and processor.history is None:
        processor.history = ObservationHistory(env.num_envs, env.device)
    elif variant == "history" and not isinstance(processor.history, ObservationHistory):
        raise ValueError("Cannot switch the history contract of a live environment")
    elif variant == "standard" and processor.history is not None:
        raise ValueError("Cannot switch a live history environment to a standard policy")
    extra = HISTORY_SIZE if variant == "history" else 0
    env.num_obs = 61 + extra
    env.num_privileged_obs = processor.critic_dim + extra
    env.cfg["policy_variant"] = variant
    if variant == "history":
        env.cfg["history"] = dict(HISTORY_SPEC)


class ObservationDelay:
    """Snapshot delay semantics, including staggered lag updates and partial resets."""

    def __init__(self, term, count, device):
        self.minimum = term["delay_min_lag"]
        self.maximum = term["delay_max_lag"]
        self.period = term["delay_update_period"]
        if not term["delay_per_env"] or term["delay_hold_prob"]:
            raise NotImplementedError("Only the per-environment, zero-hold upstream configuration is supported")
        self.count, self.device = count, device
        self.steps = torch.zeros(count, dtype=torch.long, device=device)
        self.lags = torch.zeros_like(self.steps)
        self.phase = torch.zeros_like(self.steps)
        self.history = None
        self.staggered = term["delay_per_env_phase"]
        self.reset(torch.arange(count, device=device))

    def reset(self, ids):
        self.steps[ids] = 0
        self.lags[ids] = 0
        if self.period and self.staggered:
            self.phase[ids] = torch.randint(self.period, (len(ids),), device=self.device)

    def __call__(self, value):
        if self.history is None:
            self.history = value.unsqueeze(0).repeat(self.maximum + 1, 1, 1)
        else:
            self.history = torch.roll(self.history, 1, 0)
            self.history[0] = value
            fresh = self.steps == 0
            self.history[:, fresh] = value[fresh]
        update = (self.steps + self.phase) % self.period == 0 if self.period else torch.ones_like(self.steps).bool()
        candidate = torch.randint(self.minimum, self.maximum + 1, (self.count,), device=self.device)
        self.lags = torch.where(update, candidate, self.lags)
        lag = torch.minimum(self.lags, self.steps.clamp(max=self.maximum))
        self.steps += 1
        return self.history[lag, torch.arange(self.count, device=self.device)]


class OfficialObservations:
    def __init__(self, recipe):
        self.recipe = recipe
        env = recipe.env
        self.spec = recipe.spec["observations"]
        self.cache = None
        self.history = None
        self.delays = {}
        for group, cfg in self.spec.items():
            for name, term in cfg["terms"].items():
                if term["history_length"] or term.get("modifiers"):
                    raise NotImplementedError("Upstream observation history/modifiers need an explicit adapter")
                if term["delay_max_lag"]:
                    self.delays[group, name] = ObservationDelay(term, env.num_envs, env.device)
        bias_term = recipe.spec["events"]["encoder_bias"]["params"]
        bounds = bias_term["bias_range"]
        self.encoder_bias = torch.empty_like(env.actions).uniform_(*bounds)
        axis = torch.randn(env.num_envs, 3, device=env.device)
        axis /= axis.norm(dim=-1, keepdim=True) + 1.0e-8
        max_angle = self.spec["actor"]["terms"]["base_ang_vel"]["params"]["max_angle_deg"]
        angle = torch.rand(env.num_envs, 1, device=env.device) * math.radians(max_angle)
        self.imu_rotation = torch.cat((axis * torch.sin(angle / 2), torch.cos(angle / 2)), -1)
        # Force channels are omitted explicitly on both backends to retain cross-backend compatibility.
        self.omitted = {"critic": ["foot_contact_forces"]}
        dims = {
            "base_lin_vel": 3,
            "base_ang_vel": 3,
            "projected_gravity": 3,
            "joint_pos": 14,
            "joint_vel": 14,
            "actions": 14,
            "command": 3,
            "head_command": 4,
            "body_command": 6,
            "foot_height": 2,
            "foot_air_time": 2,
            "foot_contact": 2,
            "ball_position": 3,
            "ball_velocity": 3,
            "wheel_vel": 4,
        }
        self.critic_dim = sum(dims[name] for name in self.spec["critic"]["terms"] if name != "foot_contact_forces")

    def reset(self, ids):
        self.cache = None
        if self.history is not None:
            self.history.reset(ids)
        for delay in self.delays.values():
            delay.reset(ids)

    def advance(self):
        if self.history is not None:
            self.history.advance()
        self.cache = None

    def with_history(self, actor, critic=None):
        if self.history is None:
            return actor, critic
        if hasattr(self.history, "feedback"):
            feedback = self.history.feedback
            if self.history.imu is not None:
                feedback = torch.cat((feedback, self.history.imu.acceleration), -1)
            actor = torch.cat((actor, feedback), -1)
            critic = torch.cat((critic, feedback), -1) if critic is not None else None
        history = self.history.read(actor)
        actor = torch.cat((actor, history), -1)
        critic = torch.cat((critic, history), -1) if critic is not None else None
        return actor, critic

    def compute(self):
        env, recipe = self.recipe.env, self.recipe
        if self.cache is not None:
            # Playback can change commands between physics steps without advancing delay buffers.
            self.cache[0][:, 48:] = env.commands
            return self.with_history(*self.cache)
        q, qd, pose, linear, angular, gravity = env._kinematics()
        zero_torque = torch.zeros_like(env.actions)
        recipe.refresh(q, qd, pose, linear, angular, gravity, zero_torque, env.actions)
        foot = recipe.sensors.feet.data
        if hasattr(env, "encoder_state"):
            position, velocity_j = env.encoder_state(q, qd)
        else:
            position, velocity_j = q[:, env.qi], qd[:, env.qdi]
        values = {
            "base_lin_vel": linear,
            "base_ang_vel": angular,
            "projected_gravity": gravity,
            "joint_pos": position - env.home,
            "joint_vel": velocity_j,
            "actions": env.actions,
            "command": env.commands[:, :3],
            "head_command": env.commands[:, 3:7],
            "body_command": env.commands[:, 7:13],
            "foot_height": torch.nan_to_num(recipe.sensors.height.data.heights),
            "foot_air_time": torch.nan_to_num(foot.current_air_time),
            "foot_contact": (foot.found > 0).float(),
        }
        if "wheel_vel" in self.spec["critic"]["terms"]:
            ids = recipe.robot.find_joints(r"^passive_.*wheel")[0]
            values["wheel_vel"] = qd[:, env.all_qdi[ids]]
        if getattr(env, "family", env.task) == "ballkick":
            bodies = wp.to_torch(env.states[0].body_q).reshape(env.num_envs, -1, 7)
            velocity = wp.to_torch(env.states[0].body_qd).reshape(env.num_envs, -1, 6)
            values["ball_position"] = quat_rotate_inverse(pose[:, 3:], bodies[:, env.task_logic.ball_body, :3] - pose[:, :3])
            values["ball_velocity"] = quat_rotate_inverse(pose[:, 3:], velocity[:, env.task_logic.ball_body, :3])
        groups = []
        for group, cfg in self.spec.items():
            parts = []
            for name, term in cfg["terms"].items():
                if name in self.omitted.get(group, []):
                    continue
                value = values[name].clone()
                func = term["func"].rsplit(".", 1)[-1]
                if func == "zero_command_padding":
                    value.zero_()
                if func.endswith("imu_misaligned"):
                    value = rotate(self.imu_rotation, value)
                if name == "joint_pos" and term["params"].get("biased", False):
                    value += self.encoder_bias
                noise = term["noise"] if cfg["enable_corruption"] else None
                if noise:
                    if noise["operation"] != "add":
                        raise NotImplementedError("Non-additive observation noise")
                    value += torch.empty_like(value).uniform_(noise["n_min"], noise["n_max"])
                if term["clip"] is not None:
                    value.clamp_(*term["clip"])
                if term["scale"] is not None:
                    value *= torch.as_tensor(term["scale"], device=env.device)
                if (group, name) in self.delays:
                    value = self.delays[group, name](value)
                parts.append(value)
            groups.append(torch.cat(parts, -1))
        actor, critic = groups
        if actor.shape[1] != 61 or critic.shape[1] != self.critic_dim:
            raise RuntimeError(f"Unexpected observation contract: {actor.shape}, {critic.shape}")
        if not torch.isfinite(actor).all() or not torch.isfinite(critic).all():
            raise FloatingPointError("Non-finite official observations")
        self.cache = actor, critic
        return self.with_history(*self.cache)
