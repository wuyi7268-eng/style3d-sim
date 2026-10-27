"""Asymmetric observations with the processing configured in the upstream snapshot."""

import math

import torch
import warp as wp

from examples.locomotion.math_utils import quat_rotate_inverse

from .tasks import rotate

CONTRACT = "official_asymmetric_pd_v2"


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
        for delay in self.delays.values():
            delay.reset(ids)

    def compute(self):
        env, recipe = self.recipe.env, self.recipe
        if self.cache is not None:
            # Playback can change commands between physics steps without advancing delay buffers.
            self.cache[0][:, 48:] = env.commands
            return self.cache
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
        return self.cache
