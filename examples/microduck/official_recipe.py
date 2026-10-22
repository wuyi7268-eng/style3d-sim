"""Versioned, partial upstream recipe for Newton backends.

Unsupported terms are reported and omitted, never synthesized as zero-force sensors.
See OFFICIAL_PORT_STATUS.md for the training differences that remain.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import torch
import warp as wp

from . import official_reward_functions as upstream
from . import official_velocity_rewards as velocity_rewards
from .official_adapter import SceneEntityCfg, link_origin_velocity, quat_apply_inverse
from .official_sensors import OfficialSensors, whole_body_angular_momentum
from .policy_spec import JOINT_NAMES
from .tasks import rotate

RECIPE = "official_adapted_v1"
SNAPSHOT = Path(__file__).with_name("configs") / "official_snapshot.json"


def recipe_fingerprint():
    # Include executable formulas and adapters so resume cannot silently change its objective.
    digest = hashlib.sha256()
    for name in (
        "configs/official_snapshot.json",
        "official_reward_functions.py",
        "official_velocity_rewards.py",
        "official_sensors.py",
        "official_observations.py",
        "official_events.py",
        "official_randomization.py",
        "official_reset_functions.py",
        "official_adapter.py",
        "official_recipe.py",
        "env.py",
        "stability.py",
        "tasks.py",
        "policy_spec.py",
    ):
        digest.update((Path(__file__).parent / name).read_bytes())
    return digest.hexdigest()


class RobotView:
    joint_names = JOINT_NAMES

    def find_joints(self, patterns):
        patterns = [patterns] if isinstance(patterns, str) else patterns
        ids = [i for i, name in enumerate(self.joint_names) if any(re.fullmatch(p, name) for p in patterns)]
        return ids, [self.joint_names[i] for i in ids]

    find_joints_by_actuator_names = find_joints


def multiply_xyzw(a, b):
    xyz = a[..., 3:] * b[..., :3] + b[..., 3:] * a[..., :3] + torch.cross(a[..., :3], b[..., :3], dim=-1)
    w = a[..., 3:] * b[..., 3:] - (a[..., :3] * b[..., :3]).sum(-1, keepdim=True)
    return torch.cat((xyz, w), -1)


class OfficialRecipe:
    def __init__(self, env, robot):
        snapshot = json.loads(SNAPSHOT.read_text())
        self.spec = snapshot["tasks"][env.task]
        self.env = env
        self.robot = RobotView()
        self.num_envs, self.device, self.step_dt = env.num_envs, env.device, env.step_dt
        self.common_step_counter = 0
        self.episode_length_buf = env.episode_length_buf
        origin = torch.zeros(env.num_envs, 3, device=env.device)
        self.scene = {"robot": self.robot, "ball": SimpleNamespace()}

        # Reward functions use both scene[...] and scene.terrain.
        class Scene(dict):
            pass

        self.scene = Scene(self.scene)
        self.scene.terrain = SimpleNamespace(env_origins=origin)
        self.scene.env_origins = origin
        self.sensors = OfficialSensors(env, robot)
        self.scene.sensors = {
            "feet_ground_contact": self.sensors.feet,
            "support_foot_ground_contact": self.sensors.support,
            "self_collision": self.sensors.self_contact,
            "foot_height_scan": self.sensors.height,
        }
        self.scene.update(self.scene.sensors)
        self.scene["robot/root_angmom"] = SimpleNamespace(data=None)
        self.extras = {"log": {}}
        self.command_manager = SimpleNamespace(get_command=self.get_command, get_term=self.get_term)
        self.action_manager = SimpleNamespace(action=env.actions, prev_action=torch.zeros_like(env.actions))
        self.standing = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.phase_offset = torch.zeros(env.num_envs, device=env.device)
        self.timers = {name: torch.zeros(env.num_envs, device=env.device) for name in self.spec["commands"]}
        self.sites = []
        for name in ("mouth_tip", "left_foot", "right_foot"):
            index = next(i for i, label in enumerate(robot.shape_label) if label.endswith("/" + name))
            self.sites.append((robot.shape_body[index], torch.tensor(list(robot.shape_transform[index]), device=env.device)))
        self.limit = torch.tensor(
            [(robot.joint_limit_lower[i], robot.joint_limit_upper[i]) for i in env.qdi.tolist()], device=env.device
        )
        self.params = {name: self.decode(term["params"]) for name, term in self.spec["rewards"].items()}
        self.supported = {}
        self.omitted = {}
        base = {
            "feet_flat_penalty",
            "upright",
            "variable_posture",
            "variable_posture_body_relaxed",
            "body_angular_velocity_penalty",
            "action_rate_l2",
            "joint_pos_limits",
            "track_linear_velocity",
            "track_angular_velocity",
            "feet_air_time_upright",
        }
        for name, term in self.spec["rewards"].items():
            func = term["func"].rsplit(".", 1)[-1]
            if func in base or hasattr(upstream, func) or hasattr(velocity_rewards, func):
                self.supported[name] = func
            else:
                self.omitted[name] = func
        self.swing_height = None
        for name, func in self.supported.items():
            if func == "feet_swing_height":
                self.swing_height = velocity_rewards.feet_swing_height(SimpleNamespace(params=self.params[name]), self)
        env.max_episode_length = round(self.spec["episode_length_s"] / self.step_dt)
        from .official_events import PushEvents
        from .official_observations import CONTRACT, OfficialObservations

        self.observations = OfficialObservations(self)
        self.pushes = PushEvents(self)
        env.num_privileged_obs = self.observations.critic_dim
        env.cfg["observation_contract"] = CONTRACT
        self.alignment = {
            "actor_dim": 61,
            "critic_dim": self.observations.critic_dim,
            "omitted_observations": self.observations.omitted,
            "observation_processing": "upstream noise, encoder/IMU bias, clipping, scaling and delays; clean critic",
            "reset_events": "root pose, servo offsets, recovery poses, BallKick root-relative ball",
            "interval_events": list(self.pushes.events),
            "source_sha256": snapshot["source_sha256"],
            "recipe_fingerprint": recipe_fingerprint(),
            "supported_rewards": self.supported,
            "omitted_rewards": self.omitted,
            "approximate_rewards": [
                n for n in self.supported if n in ("joint_torques_l2", "joint_torque_rate_l2", "servo_stall")
            ],
            "differences": [
                "position-PD instead of BAM",
                "backend solver settings and collision capacity differ from official mjlab",
                "commanded PD torque instead of BAM actuator force",
                "force/impact sensor terms omitted; geometric contact flags differ between solvers",
                "terrain height specialized to the current infinite flat plane",
                "critic foot-contact-force channels omitted; contact flags use geometric contacts",
                "no mass/inertia, COM, BAM friction or armature randomization; no expert BC",
                "non-finite physics fails fast instead of per-environment NaN termination",
                "command sampling simplified; ranges and reward-weight stages preserved",
                (
                    "convex collision proxies; robot self-collisions enabled; self_collisions reward uses geometric contacts"
                    if env.cfg["self_collisions"]
                    else "convex collision proxies; robot self-collisions disabled; self_collisions reward has no valid signal"
                ),
                "foot friction fixed at 1.0 across environments instead of upstream randomization",
            ],
        }

    def decode(self, value):
        if isinstance(value, dict) and "name" in value and "joint_ids" in value:
            cfg = SceneEntityCfg(value["name"])
            if value.get("joint_names"):
                cfg.joint_ids = self.robot.find_joints(value["joint_names"])[0]
            if value.get("site_names"):
                cfg.site_ids = [("mouth_tip", "left_foot", "right_foot").index(n) for n in value["site_names"]]
            return cfg
        if isinstance(value, dict):
            return {k: self.decode(v) for k, v in value.items()}
        return value

    def get_command(self, name):
        return self.env.commands[:, {"twist": slice(0, 3), "head_pose": slice(3, 7), "body_pose": slice(7, 13)}[name]]

    def get_term(self, name):
        return SimpleNamespace(is_standing_env=self.standing)

    def stage(self, stages):
        selected = stages[0]
        for stage in stages:
            if self.common_step_counter > stage["step"]:
                selected = stage
        return selected

    def reset(self, ids):
        self.observations.reset(ids)
        self.sensors.reset(ids)
        if self.swing_height is not None:
            self.swing_height.peak_heights[ids] = 0.0
        self.phase_offset[ids] = torch.rand(len(ids), device=self.device)
        self.standing[ids] = False
        # Stateful reward caches must not couple two episodes after a partial reset.
        for name in ("_prev_neck_actions", "_prev_actuator_forces"):
            if hasattr(self, name):
                getattr(self, name)[ids] = 0.0
        self.update_commands(ids)

    def update_commands(self, ids=None):
        env = self.env
        if ids is None:
            for timer in self.timers.values():
                timer.sub_(self.step_dt)
        for name, config in self.spec["commands"].items():
            selected = ids if ids is not None else (self.timers[name] <= 0).nonzero().flatten()
            if not len(selected):
                continue
            if env.task == "groundpick":
                continue
            ranges = config["ranges"]
            if name == "twist":
                ranges = [ranges[n] for n in ("lin_vel_x", "lin_vel_y", "ang_vel_z")]
                probability = config.get("rel_standing_envs", 0.0)
                for term in self.spec["curricula"].values():
                    if term["func"].endswith(".standing_envs_curriculum"):
                        probability = self.stage(term["params"]["standing_stages"])["rel_standing_envs"]
                self.standing[selected] = torch.rand(len(selected), device=self.device) < probability
            else:
                for term in self.spec["curricula"].values():
                    params = term["params"]
                    if term["func"].endswith(".pose_command_range_curriculum") and params["command_name"] == name:
                        ranges = self.stage(params["range_stages"])["ranges"]
            bounds = torch.tensor(ranges, device=self.device)
            values = bounds[:, 0] + torch.rand(len(selected), len(bounds), device=self.device) * (bounds[:, 1] - bounds[:, 0])
            if name == "twist":
                values[self.standing[selected]] = 0.0
            else:
                zero = torch.rand(len(selected), device=self.device) < config.get("zero_command_prob", 0.0)
                values[zero] = 0.0
                if "axis_zero_prob" in config:
                    values *= torch.rand_like(values) >= torch.tensor(config["axis_zero_prob"], device=self.device)
                if "twist_command_name" in config:
                    values[~self.standing[selected]] = 0.0
            self.get_command(name)[selected] = values
            lo, hi = config["resampling_time_range"]
            self.timers[name][selected] = lo + torch.rand(len(selected), device=self.device) * (hi - lo)
        if env.task == "groundpick":
            period = self.spec["commands"]["twist"]["period"]
            phase = self.phase_offset + env.episode_length_buf * self.step_dt / period
            env.commands[:, 0] = torch.cos(2 * torch.pi * phase)
            env.commands[:, 1] = torch.sin(2 * torch.pi * phase)

    def refresh(self, q, qd, pose, linear, angular, gravity, torque, previous):
        env = self.env
        bodies = wp.to_torch(env.states[0].body_q).reshape(env.num_envs, -1, 7)
        velocity = wp.to_torch(env.states[0].body_qd).reshape(env.num_envs, -1, 6)
        positions, rotations, site_velocities = [], [], []
        com = wp.to_torch(env.model.body_com).reshape(env.num_envs, -1, 3)
        for body, local in self.sites:
            p = bodies[:, body]
            positions.append(p[:, :3] + rotate(p[:, 3:], local[:3].expand(env.num_envs, -1)))
            quat = multiply_xyzw(p[:, 3:], local[3:].expand(env.num_envs, -1))
            rotations.append(quat[:, [3, 0, 1, 2]])
            offset = rotate(p[:, 3:], local[:3].expand(env.num_envs, -1) - com[:, body])
            site_velocities.append(velocity[:, body, :3] + torch.cross(velocity[:, body, 3:], offset, dim=-1))
        data = SimpleNamespace(
            root_link_pos_w=pose[:, :3],
            root_link_quat_w=pose[:, [6, 3, 4, 5]],
            root_link_lin_vel_b=linear,
            root_link_ang_vel_b=angular,
            root_link_lin_vel_w=link_origin_velocity(
                velocity[:, 0, :3], velocity[:, 0, 3:], pose[:, 3:], wp.to_torch(env.model.body_com)[0]
            ),
            body_link_quat_w=bodies[:, :, [6, 3, 4, 5]],
            body_link_ang_vel_w=velocity[:, :, 3:],
            gravity_vec_w=torch.tensor([0.0, 0.0, -1.0], device=env.device),
            joint_pos=q[:, env.qi],
            joint_vel=qd[:, env.qdi],
            default_joint_pos=env.home.expand(env.num_envs, -1),
            actuator_force=torque,
            site_pos_w=torch.stack(positions, 1),
            site_quat_w=torch.stack(rotations, 1),
            site_lin_vel_w=torch.stack(site_velocities, 1),
        )
        self.robot.data = data
        count = self.sensors.robot_body_count
        mass = wp.to_torch(env.model.body_mass).reshape(env.num_envs, -1)[:, :count]
        inertia = wp.to_torch(env.model.body_inertia).reshape(env.num_envs, -1, 3, 3)[:, :count]
        self.scene["robot/root_angmom"].data = whole_body_angular_momentum(
            bodies[:, :count], velocity[:, :count], com[:, :count], mass, inertia
        )
        # On an infinite plane, both official height rays hit z=0; below-plane misses use the same clamp.
        self.sensors.height.data.heights = data.site_pos_w[:, 1:, 2].clamp(0.0, 1.0)
        self.gravity = gravity
        self.action_manager.prev_action = previous
        if env.task == "ballkick":
            self.scene["ball"].data = SimpleNamespace(root_link_lin_vel_w=velocity[:, env.task_logic.ball_body, :3])

    def base_reward(self, func, params):
        data = self.robot.data
        env = self.env
        if func == "feet_swing_height":
            return self.swing_height(self, **params)
        if func == "feet_air_time_upright":
            params = params.copy()
            gate = params.pop("gate_tilt_above_deg", 40.0)
            asset = self.scene[params.pop("asset_cfg", SceneEntityCfg("robot")).name]
            return velocity_rewards.feet_air_time(self, **params) * (1 - upstream._fallen_mask(self, asset, 0.0, gate))
        if hasattr(velocity_rewards, func):
            return getattr(velocity_rewards, func)(self, **params)
        if func == "feet_flat_penalty":
            if params.get("sensor_name") is not None:
                raise NotImplementedError("Contact-gated foot tilt requires a contact sensor")
            quats = data.site_quat_w[:, params["asset_cfg"].site_ids]
            gravity = torch.nn.functional.normalize(data.gravity_vec_w, dim=-1)
            return sum(quat_apply_inverse(quats[:, i], gravity)[:, :2].square().sum(-1) for i in range(quats.shape[1]))
        if func == "upright":
            return torch.exp(-self.gravity[:, :2].square().sum(-1) / params["std"] ** 2)
        if func == "track_linear_velocity":
            error = (env.commands[:, :2] - data.root_link_lin_vel_b[:, :2]).square().sum(-1)
            return torch.exp(-(error + data.root_link_lin_vel_b[:, 2].square()) / params["std"] ** 2)
        if func == "track_angular_velocity":
            error = (env.commands[:, 2] - data.root_link_ang_vel_b[:, 2]).square()
            return torch.exp(-(error + data.root_link_ang_vel_b[:, :2].square().sum(-1)) / params["std"] ** 2)
        if func == "action_rate_l2":
            return (env.actions - self.action_manager.prev_action).square().sum(-1)
        if func == "body_angular_velocity_penalty":
            return data.body_link_ang_vel_w[:, 0, :2].square().sum(-1)
        if func == "joint_pos_limits":
            # mjlab uses the asset's 0.9 soft-limit factor, not the hard stops.
            mid = self.limit.mean(-1)
            half = (self.limit[:, 1] - self.limit[:, 0]) * 0.45
            return ((mid - half - data.joint_pos).clamp(min=0) + (data.joint_pos - mid - half).clamp(min=0)).sum(-1)
        if func in ("variable_posture", "variable_posture_body_relaxed"):
            ids = params["asset_cfg"].joint_ids
            speed = env.commands[:, :2].norm(dim=-1) + env.commands[:, 2].abs()
            stds = {}
            for key in ("std_standing", "std_walking", "std_running", "std_body"):
                if key in params:
                    stds[key] = torch.tensor(
                        [next(v for pattern, v in params[key].items() if re.fullmatch(pattern, JOINT_NAMES[i])) for i in ids],
                        device=env.device,
                    )
            standing = speed < params["walking_threshold"]
            std = torch.where(standing[:, None], stds["std_standing"], stds["std_walking"])
            std = torch.where((speed >= params["running_threshold"])[:, None], stds["std_running"], std)
            if "std_body" in stds:
                active = standing & (env.commands[:, 9:12].abs().amax(-1) > 0)
                std = torch.where(active[:, None], stds["std_body"], std)
            return torch.exp(-((data.joint_pos[:, ids] - env.home[ids]) / std).square().mean(-1))
        return getattr(upstream, func)(self, **params)

    def reward(self, q, qd, pose, linear, angular, gravity, torque, previous):
        self.extras["log"].clear()
        self.refresh(q, qd, pose, linear, angular, gravity, torque, previous)
        total = torch.zeros(self.num_envs, device=self.device)
        terms = {}
        for name, func in self.supported.items():
            weight = self.spec["rewards"][name]["weight"]
            for curriculum in self.spec["curricula"].values():
                p = curriculum["params"]
                if curriculum["func"].endswith(".reward_weight") and p["reward_name"] == name:
                    # The upstream stage comparison is strictly greater than, including step zero.
                    for stage in p["weight_stages"]:
                        if self.common_step_counter > stage["step"]:
                            weight = stage["weight"]
            if weight == 0:
                continue
            value = self.base_reward(func, self.params[name])
            if value.shape != total.shape or not torch.isfinite(value).all():
                raise FloatingPointError(f"Non-finite or invalid official reward: {name}")
            terms[name] = value * weight
            total += terms[name] * self.step_dt
        self.common_step_counter += 1
        return total, terms
