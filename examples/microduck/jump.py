"""Isolated maximum-height trampoline training/playback; no external demo assistance."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import deque
from copy import copy, deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import warp as wp
from rsl_rl.modules import ActorCritic
from torch import nn

from .adapter import rotate
from .env import MicroDuckEnv, PhysicsConfig, environment_config
from .policy_spec import ASSET, build_robot
from .ppo import ppo_config
from .rebound import SENSOR_DIM, ReboundCycle, sensor_frame, specific_force
from .scenes import SCENE_CONFIG, login
from .sensors import OfficialSensors

CONFIG_PATH = Path(__file__).with_name("configs") / "trampoline_jump.json"
LEGACY_CONTRACT = "trampoline_jump_v1"
MAX_HEIGHT_CONTRACT = "trampoline_max_height_v1"
CONTRACT = "trampoline_rebound_sensors_v1"
CURRENT_FRAME_CONTRACT = "trampoline_rebound_current_frame_v1"
REBOUND_CONTRACTS = (CONTRACT, CURRENT_FRAME_CONTRACT)
SUPPORTED_CONTRACTS = (*REBOUND_CONTRACTS, MAX_HEIGHT_CONTRACT, LEGACY_CONTRACT)
FRAME_DIM, CURRENT_DIM, CRITIC_DIM = 75, 82, 87


def load_config(path=CONFIG_PATH):
    config = json.loads(Path(path).read_text())
    if config["contract"] not in SUPPORTED_CONTRACTS:
        raise ValueError("Unsupported jump contract")
    task = config["task"]
    for key, value in task.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid jump setting: {key}")
    if task.get("shaping_version", 1) < 4 and not 0 < task["height_min"] <= task["height_initial_max"] <= task["height_max"]:
        raise ValueError("Expected 0 < minimum <= initial maximum <= maximum height")
    maximum = task.get("shaping_version", 1) >= 4
    if maximum != (config["contract"] in (*REBOUND_CONTRACTS, MAX_HEIGHT_CONTRACT)):
        raise ValueError("Jump shaping and observation contracts must agree")
    if (task.get("shaping_version") == 5) != (config["contract"] in REBOUND_CONTRACTS):
        raise ValueError("Jump shaping and observation contracts must agree")
    current_frame = config["contract"] == CURRENT_FRAME_CONTRACT
    for key in (("history_length",) if maximum else ("history_length", "curriculum_window")):
        minimum = 0 if key == "history_length" and current_frame else 1
        if isinstance(task[key], bool) or not isinstance(task[key], int) or task[key] < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    if current_frame and task["history_length"] != 0:
        raise ValueError("Current-frame jump observations require history_length=0")
    for key in (() if maximum else ("zero_probability", "curriculum_success")):
        if not 0 <= task[key] <= 1:
            raise ValueError(f"{key} must be in [0, 1]")
    for key in (
        "hold_s",
        "settle_s",
        "episode_s",
        "takeoff_timeout_s",
        "minimum_air_s",
        "action_scale",
        "compression_target",
    ):
        if task[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if not maximum and any(task[k] <= 0 for k in ("height_tolerance", "height_increment")):
        raise ValueError("Height tolerance and increment must be positive")
    if not 0 < task["stable_tilt_deg"] < task["failure_tilt_deg"] < 90:
        raise ValueError("Expected stable tilt < failure tilt < 90 degrees")
    if not all(math.isfinite(v) for v in config["rewards"].values()):
        raise ValueError("Reward weights must be finite")
    if task.get("shaping_version", 1) not in (1, 2, 3, 4, 5):
        raise ValueError("Unsupported jump shaping_version")
    if task.get("shaping_version", 1) >= 2:
        for key in (
            "prepare_relative_speed", "prepare_angular_speed", "prepare_tilt_deg", "prepare_timeout_s",
            "crouch_target", "push_min_compression", "push_min_crouch", "push_extension_speed",
        ):
            if task.get(key, 0) <= 0:
                raise ValueError(f"{key} must be positive for jump shaping v2")
        if not task["stable_tilt_deg"] <= task["prepare_tilt_deg"] < task["failure_tilt_deg"]:
            raise ValueError("Expected stable tilt <= preparation tilt < failure tilt")
        if task["push_min_compression"] > task["compression_target"] or task["push_min_crouch"] > task["crouch_target"]:
            raise ValueError("Push thresholds must not exceed compression/crouch targets")
        if task["prepare_timeout_s"] <= task["settle_s"]:
            raise ValueError("Preparation timeout must exceed settling time")
        for name in ("prepare", "crouch", "push"):
            if name not in config["rewards"] or config["rewards"][name] < 0:
                raise ValueError(f"Missing or negative shaping reward: {name}")
    if task.get("shaping_version", 1) >= 3:
        for key in ("launch_velocity_cap", "launch_tilt_deg", "launch_angular_speed"):
            if task.get(key, 0) <= 0:
                raise ValueError(f"{key} must be positive for jump shaping v3")
        if task["launch_tilt_deg"] >= task["failure_tilt_deg"]:
            raise ValueError("Launch tilt must be below failure tilt")
        for name in ("launch_velocity", "apex_progress", "landing_stability"):
            if config["rewards"].get(name, -1) < 0:
                raise ValueError(f"Missing or negative shaping reward: {name}")
    if maximum and task.get("height_reward_scale", 0) <= 0:
        raise ValueError("height_reward_scale must be positive for maximum-height shaping")
    if task.get("shaping_version") == 5:
        for key in ("valid_flight_s", "valid_clearance_m", "valid_height_m", "contact_confirm_s", "flight_timeout_s"):
            if task.get(key, 0) <= 0:
                raise ValueError(f"{key} must be positive")
        if not task.get("continuous_bounce"):
            raise ValueError("Rebound training requires continuous_bounce")
        if not 0 < task.get("accel_filter_alpha", 0) <= 1:
            raise ValueError("accel_filter_alpha must be in (0, 1]")
        if task["valid_flight_s"] < task["minimum_air_s"]:
            raise ValueError("Flight validation must include the takeoff debounce interval")
        if task.get("reward_revision", 1) not in (1, 2):
            raise ValueError("Unsupported rebound reward revision")
        if task.get("reward_revision", 1) == 2:
            if not 0 < task.get("stall_grace_s", 0) < task["takeoff_timeout_s"]:
                raise ValueError("stall_grace_s must be positive and below takeoff timeout")
            if task.get("stall_ramp_s", 0) <= 0:
                raise ValueError("stall_ramp_s must be positive")
            if config["rewards"].get("airborne", 0) <= 0 or config["rewards"].get("stall", 0) >= 0:
                raise ValueError("Expected positive airborne reward and negative stall penalty")
    return config


class JumpCycle:
    """One attempt with a settled reference and one-shot rewards; old commands are replay-only."""

    def __init__(self, count, device, settings):
        self.cfg = settings
        # Preserve height-command v1/v2/v3 semantics during diagnostic playback.
        self.shaped = settings.get("shaping_version", 1) >= 2
        self.dense = settings.get("shaping_version", 1) >= 3
        self.max_height = settings.get("shaping_version", 1) == 4
        self.phase = torch.zeros(count, dtype=torch.long, device=device)
        for name in (
            "target",
            "reference",
            "peak",
            "age",
            "phase_age",
            "stable_time",
            "air_time",
            "landing_age",
            "compression",
            "reference_bed",
            "score",
            "prepare_progress",
            "crouch",
            "push_progress",
            "height_progress",
            "launch_progress",
            "height_paid",
            "reference_leg",
            "settle_com_sum",
            "settle_bed_sum",
            "settle_leg_sum",
        ):
            setattr(self, name, torch.zeros(count, device=device))
        self.touch_time = torch.full((count, 2), -1.0, device=device)
        self.apex_seen = torch.zeros(count, dtype=torch.bool, device=device)

    def reset(self, ids, targets, com_z):
        self.phase[ids] = 0
        for name in ("age", "phase_age", "stable_time", "air_time", "landing_age",
                     "compression", "score", "height_progress", "launch_progress", "height_paid"):
            getattr(self, name)[ids] = 0
        for name in ("prepare_progress", "crouch", "push_progress", "reference_leg",
                     "settle_com_sum", "settle_bed_sum", "settle_leg_sum"):
            getattr(self, name)[ids] = 0
        self.target[ids] = 0 if self.max_height else targets
        self.reference[ids] = com_z
        self.peak[ids] = com_z
        self.reference_bed[ids] = 0
        self.touch_time[ids] = -1
        self.apex_seen[ids] = False

    def rearm(self, ids, com_z):
        """Start the next bounce in-place after a stable landing, preserving cloth and robot state."""
        if len(ids) == 0:
            return
        self.reset(ids, None, com_z)

    def advance(self, state, dt):
        cfg = self.cfg
        phase = self.phase.clone()
        contact, z, vz = state.contact, state.com_z, state.com_vz
        both, none = contact.all(-1), ~contact.any(-1)
        jumping = torch.ones_like(self.target, dtype=torch.bool) if self.max_height else self.target > 0
        stable = (
            both
            & (state.tilt < math.radians(cfg["stable_tilt_deg"]))
            & (state.speed < cfg["stable_speed"])
            & (state.omega < cfg["stable_angular_speed"])
            & (state.root_clearance > cfg["stable_root_height"])
            & (state.drift < cfg["maximum_drift"])
        )
        self.age += dt
        self.phase_age += dt
        ready = stable
        shaping = {}
        if self.shaped:
            safe = (
                both & ~state.bad_contact & ~state.outside
                & (state.tilt < math.radians(cfg["prepare_tilt_deg"]))
                & (state.omega < cfg["prepare_angular_speed"])
                & (state.root_clearance > cfg["stable_root_height"])
                & (state.drift < cfg["maximum_drift"])
            )
            # A supported robot can move with the bed without losing its relative balance.
            ready = torch.where(jumping, safe & (state.support_relative_speed < cfg["prepare_relative_speed"]), stable)
            for name, value in (("settle_com_sum", z), ("settle_bed_sum", state.bed_z),
                                ("settle_leg_sum", state.leg_height)):
                total = getattr(self, name)
                setattr(self, name, torch.where((phase == 0) & ready, total + value * dt, 0.0))
        self.stable_time = torch.where(torch.where(phase == 0, ready, stable), self.stable_time + dt, 0.0)
        self.air_time = torch.where(none, self.air_time + dt, 0.0)
        armed = (phase == 0) & (self.stable_time >= cfg["settle_s"])
        self.reference[armed] = z[armed]
        self.reference_bed[armed] = state.bed_z[armed]
        if self.shaped:
            progress = torch.where(phase == 0, (self.stable_time / cfg["settle_s"]).clamp(max=1), 0.0)
            best = torch.maximum(self.prepare_progress, progress)
            shaping["prepare"] = best - self.prepare_progress
            self.prepare_progress = best
            # Average the supported preparation window rather than latching an oscillation extreme.
            duration = self.stable_time[armed]
            self.reference[armed] = self.settle_com_sum[armed] / duration
            self.reference_bed[armed] = self.settle_bed_sum[armed] / duration
            self.reference_leg[armed] = self.settle_leg_sum[armed] / duration
        self.peak[armed] = self.reference[armed]
        self.phase[armed] = 1
        self.phase_age[armed] = 0
        self.stable_time[armed] = 0
        loading = phase == 1
        old_compression = self.compression.clone()
        depth = (self.reference_bed - state.bed_z).clamp(0, cfg["compression_target"])
        supported = loading & (safe if self.shaped else both)
        self.compression = torch.where(supported, torch.maximum(self.compression, depth), self.compression)
        if self.shaped:
            old_crouch = self.crouch.clone()
            crouch = (self.reference_leg - state.leg_height).clamp(0, cfg["crouch_target"])
            self.crouch = torch.where(supported, torch.maximum(self.crouch, crouch), self.crouch)
            shaping["crouch"] = (self.crouch - old_crouch) / cfg["crouch_target"] * jumping
            # The new task uses a velocity scale; legacy playback matches the requested ballistic rise.
            if self.max_height:
                required_v = torch.full_like(vz, cfg["launch_velocity_cap"])
            else:
                remaining = (self.reference + self.target - z).clamp_min(0.005)
                required_v = torch.sqrt(2 * 9.81 * remaining)
            extending = (
                supported & jumping & (vz > 0)
                & (self.compression >= cfg["push_min_compression"])
                & (self.crouch >= cfg["push_min_crouch"])
                & (state.leg_extension_speed > cfg["push_extension_speed"])
            )
            progress = torch.where(extending, (vz / required_v).clamp(0, 1), 0.0)
            best = torch.maximum(self.push_progress, progress)
            shaping["push"] = best - self.push_progress
            self.push_progress = best
        takeoff = loading & jumping & (self.air_time >= cfg["minimum_air_s"]) & (vz > 0)
        self.phase[takeoff] = 2
        self.phase_age[takeoff] = 0
        flying = (phase == 2) | takeoff
        self.peak = torch.where(flying, torch.maximum(self.peak, z), self.peak)
        apex = flying & (vz <= 0) & ~self.apex_seen
        self.apex_seen |= apex
        if self.max_height:
            # There is no requested height to match; height progress carries the objective.
            height_ok = torch.ones_like(jumping)
            self.score = torch.ones_like(self.target)
        else:
            error = (self.peak - self.reference - self.target).abs()
            tolerance = torch.maximum(
                self.target * cfg["height_relative_tolerance"], torch.full_like(self.target, cfg["height_tolerance"])
            )
            height_ok = error <= tolerance
            self.score = torch.exp(-(error / tolerance).square())
        touchdown = (phase == 2) & contact.any(-1)
        # A short flight can cross apex and touch down within one control step.
        apex |= touchdown & ~self.apex_seen
        self.apex_seen |= touchdown
        self.phase[touchdown] = 3
        self.phase_age[touchdown] = 0
        self.stable_time[touchdown] = 0
        landing = (phase == 3) | touchdown
        self.landing_age = torch.where(touchdown, 0.0, self.landing_age + dt)
        new_touch = landing[:, None] & contact & (self.touch_time < 0)
        self.touch_time = torch.where(new_touch, self.landing_age[:, None], self.touch_time)
        landed_both = (self.touch_time >= 0).all(-1)
        paired = landed_both & ((self.touch_time[:, 0] - self.touch_time[:, 1]).abs() <= cfg["touchdown_window_s"])
        second_touch = new_touch.any(-1) & landed_both
        zero_success = loading & (~jumping) & (self.stable_time >= cfg["hold_s"])
        jump_success = (phase == 3) & paired & height_ok & (self.stable_time >= cfg["hold_s"])
        failure = (
            state.bad_contact
            | (state.tilt > math.radians(cfg["failure_tilt_deg"]))
            | state.outside
            | (state.root_clearance < 0.035)
            | (loading & jumping & (self.phase_age > cfg["takeoff_timeout_s"]))
            | ((phase == 3) & ~landed_both & (self.landing_age > cfg["touchdown_window_s"]))
            | ((phase == 3) & landed_both & ~paired)
        )
        # Incomplete finite-horizon attempts terminate without time-limit bootstrapping.
        failure |= self.age >= cfg["episode_s"]
        if self.shaped:
            failure |= (phase == 0) & (self.phase_age > cfg["prepare_timeout_s"]) & ~armed
        success = (zero_success | jump_success) & ~failure
        if not self.max_height:
            desired_v = torch.sqrt(2 * 9.81 * self.target)
            if self.shaped:
                desired_v = torch.sqrt(2 * 9.81 * (self.reference + self.target - z).clamp_min(0))
        if self.dense:
            # Adapt PlatformJump pj_launch's record increments to a moving bed.
            # World upward speed alone rewards passive bed rebound; require COM
            # extension relative to the feet, without a hard crouch/preload gate.
            controlled = (
                loading & both & jumping & ~failure
                & (state.tilt < math.radians(cfg["launch_tilt_deg"]))
                & (state.omega < cfg["launch_angular_speed"])
                & (state.root_clearance > cfg["stable_root_height"])
                & (state.drift < cfg["maximum_drift"])
            )
            speed = torch.minimum(vz, state.leg_extension_speed).clamp(0, cfg["launch_velocity_cap"])
            progress = torch.where(controlled, speed / cfg["launch_velocity_cap"], 0.0)
            best = torch.maximum(self.launch_progress, progress)
            shaping["launch_velocity"] = best - self.launch_progress
            self.launch_progress = best

            # In maximum-height mode, the scale sets units, not a target or cap.
            # Reclaim flight progress on failure. Retain the high-water mark so
            # a failed playback attempt cannot repeatedly earn the same height.
            if self.max_height:
                height = (self.peak - self.reference).clamp_min(0) / cfg["height_reward_scale"]
            else:
                height = ((self.peak - self.reference) / self.target.clamp_min(1e-6)).clamp(0, 1)
            progress = torch.where(flying & jumping & ~failure, height, 0.0)
            best = torch.maximum(self.height_progress, progress)
            increment = best - self.height_progress
            self.height_progress = best
            self.height_paid += increment
            shaping["apex_progress"] = increment - torch.where(failure, self.height_paid, 0.0)
            self.height_paid = torch.where(failure, 0.0, self.height_paid)

            # A continuous landing signal precedes the strict 0.4 s success
            # hold; high bounce speed or tilt suppress it rather than paying
            # only after the policy has already mastered the final balance.
            balance = torch.exp(
                -(state.tilt / math.radians(cfg["stable_tilt_deg"])).square()
                - (state.speed / cfg["stable_speed"]).square()
                - (state.omega / cfg["stable_angular_speed"]).square()
            )
            shaping["landing_stability"] = (
                balance * self.score * landing * paired * both * ~failure * jumping * dt
            )
        events = {
            **shaping,
            "compression": (self.compression - old_compression) / cfg["compression_target"] * jumping,
            "launch": takeoff.float() * (1.0 if self.max_height else torch.exp(-((vz - desired_v) / 0.5).square())),
            "apex": apex.float() * self.score,
            "landing": second_touch.float() * paired * self.score,
            "success": success.float(),
            "failure": failure.float(),
            "upright": torch.cos(state.tilt).clamp_min(0) * dt,
            "stable": stable.float() * ((~jumping) | (phase == 3)) * dt,
        }
        if self.shaped:
            for name in ("prepare", "crouch", "compression", "push", "launch", "apex", "landing"):
                events[name] = torch.where(failure, 0.0, events[name])
        return events, success, failure


class JumpHistory:
    def __init__(self, count, length, device, frame_dim=FRAME_DIM):
        self.frames = torch.zeros(count, length, frame_dim, device=device)
        self.current = torch.zeros(count, frame_dim, device=device)
        self.fresh = torch.ones(count, dtype=torch.bool, device=device)

    def advance(self):
        self.frames[:, 1:] = self.frames[:, :-1].clone()
        self.frames[:, 0] = self.current

    def read(self, frame):
        self.frames[self.fresh] = frame[self.fresh, None, :]
        self.fresh.zero_()
        self.current.copy_(frame)
        return self.frames.flatten(1)

    def reset(self, ids):
        self.fresh[ids] = True


class JumpActorCritic(ActorCritic):
    def __init__(self, num_actor_obs, num_critic_obs, num_actions, history_length=20,
                 current_dim=CURRENT_DIM, critic_dim=CRITIC_DIM, frame_dim=FRAME_DIM, **kwargs):
        self.current_dim, self.critic_dim = current_dim, critic_dim
        if isinstance(history_length, bool) or not isinstance(history_length, int) or history_length < 0:
            raise ValueError("history_length must be a nonnegative integer")
        self.history_size = history_length * frame_dim
        if (num_actor_obs, num_critic_obs, num_actions) != (
            self.current_dim + self.history_size,
            self.critic_dim + self.history_size,
            14,
        ):
            raise ValueError("Jump policy dimensions do not match its saved observation contract")
        # Keep the saved history architecture for old policies; current-frame policies have no encoder parameters.
        encoded_size = 64 if self.history_size else 0
        super().__init__(self.current_dim + encoded_size, self.critic_dim + encoded_size, num_actions, **kwargs)
        if self.history_size:
            self.actor_history = nn.Sequential(nn.Linear(self.history_size, 128), nn.ELU(), nn.Linear(128, 64))
            self.critic_history = nn.Sequential(nn.Linear(self.history_size, 128), nn.ELU(), nn.Linear(128, 64))

    def actor_input(self, observations):
        if not self.history_size:
            return observations
        return torch.cat((observations[:, :self.current_dim], self.actor_history(observations[:, self.current_dim:])), -1)

    def update_distribution(self, observations):
        super().update_distribution(self.actor_input(observations))

    def act_inference(self, observations):
        return self.actor(self.actor_input(observations))

    def evaluate(self, critic_observations, **kwargs):
        if not self.history_size:
            return self.critic(critic_observations)
        return self.critic(
            torch.cat((critic_observations[:, :self.critic_dim],
                       self.critic_history(critic_observations[:, self.critic_dim:])), -1)
        )


class JumpSensors(OfficialSensors):
    """Opt into cloth contacts locally, without changing other tasks' sensors."""

    def __init__(self, env, robot):
        proxy = copy(env)
        proxy.mixed_terrain = True
        mask = torch.zeros(env.model.particle_count, dtype=torch.int32, device=env.device)
        mask[env.terrain.cloth_indices] = 1
        proxy.terrain = SimpleNamespace(pad_particle=wp.from_torch(mask))
        super().__init__(proxy, robot)
        # The full-collision asset names its feet ankle_left/right, whereas the
        # walking asset uses other link names. Classify every shape on the foot
        # body, not just the explicitly named sole collider.
        roles = self.roles.numpy()
        bodies = env.model.shape_body.numpy()
        for side, role in (("left", 1), ("right", 2)):
            foot_bodies = {
                int(bodies[i])
                for i, label in enumerate(env.model.shape_label)
                if label.rsplit("/", 1)[-1] == f"{side}_foot_collision"
            }
            roles[np.isin(bodies, list(foot_bodies))] = role
        self.roles.assign(roles)
        self.bad_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def update(self):
        super().update()
        self.bad_contact |= self.robot_ground.data.found[:, 0] > self.feet.data.found.sum(-1)


class JumpEnv:
    num_actions = 14

    def __init__(self, config, num_envs=8, device="cuda:0", viewer=None, seed=42, height=None, physics=None,
                 *, auto_reset=True):
        self.config, self.task_cfg = deepcopy(config), config["task"]
        self.device, self.num_envs = torch.device(device), num_envs
        self.auto_reset = auto_reset
        self.completed = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.max_height = self.task_cfg.get("shaping_version", 1) >= 4
        self.rebound = self.task_cfg.get("shaping_version", 1) == 5
        self.contract = config["contract"]
        self.current_dim = CURRENT_DIM if self.max_height else CURRENT_DIM + 1
        self.critic_dim = CRITIC_DIM if self.max_height else CRITIC_DIM + 1
        self.frame_dim = SENSOR_DIM if self.rebound else FRAME_DIM
        if self.rebound:
            self.current_dim, self.critic_dim = SENSOR_DIM, CRITIC_DIM
        self.fixed_height = height
        self.validate_height(height)
        if not self.max_height:
            self.height_limit = self.task_cfg["height_initial_max"]
            self.outcomes = deque(maxlen=self.task_cfg["curriculum_window"])
        self.history = (
            JumpHistory(num_envs, self.task_cfg["history_length"], self.device, self.frame_dim)
            if self.task_cfg["history_length"] else None
        )
        self.cycle = (ReboundCycle if self.rebound else JumpCycle)(num_envs, self.device, self.task_cfg)
        # Match velocity's reduced collision asset for both physics and sensor body mappings.
        self.base = MicroDuckEnv(
            solver="synreal",
            num_envs=num_envs,
            device=device,
            task="velocity",
            terrain="trampoline",
            viewer=viewer,
            seed=seed,
            randomize_foot_friction=False,
            physics=physics or environment_config("mixed_soft_pad").physics,
        )
        self.step_dt = self.base.step_dt
        self.max_episode_length = math.ceil(self.task_cfg["episode_s"] / self.step_dt)
        self.episode_length_buf = self.base.episode_length_buf
        self.num_obs = self.current_dim + self.task_cfg["history_length"] * self.frame_dim
        self.num_privileged_obs = self.critic_dim + self.task_cfg["history_length"] * self.frame_dim
        robot, _, _ = build_robot(task="velocity")
        self.base.official.sensors = JumpSensors(self.base, robot)
        self.sensors = self.base.official.sensors
        self.model, self.terrain = self.base.model, self.base.terrain
        self.cfg = {
            "task": "trampoline_jump",
            "robot_asset": ASSET.name,
            "robot_collision_model": "velocity",
            "observation_contract": self.contract,
            "physics": self.base.cfg["physics"],
            "terrain": deepcopy(self.terrain.cfg),
            "jump": deepcopy(config),
            "assistance": False,
            "deployment_compatible": False,
        }
        self.previous = torch.zeros(num_envs, 14, device=self.device)
        if self.rebound:
            import xml.etree.ElementTree as ET

            site = ET.parse(ASSET).find(".//site[@name='imu']")
            if site is None or site.get("quat", "1 0 0 0") not in ("1 0 0 0", "1 -0 -0 -0"):
                raise ValueError("Expected a trunk-aligned body IMU site")
            self.imu_offset = torch.tensor([float(x) for x in site.attrib["pos"].split()], device=self.device)
            self.imu_velocity = torch.zeros(num_envs, 3, device=self.device)
            self.acceleration = torch.zeros_like(self.imu_velocity)
            self.sensor_previous = torch.zeros(num_envs, SENSOR_DIM, device=self.device)
            self.sensor_observation = torch.zeros_like(self.sensor_previous)
            self.sensor_bias = torch.zeros(num_envs, 3, device=self.device)
            self.cfg["actor_observation"] = "gyro, gravity_direction, specific_force, joint_pos, joint_vel, previous_action"
            self.cfg["imu_model"] = "control-interval point-velocity difference; trunk axes; one-step sensor delay"
            self.cfg["deployment_limits"] = "Requires raw acceleration runtime support and hardware/actuator calibration"

        self._cache = None
        self._surface_topology = self._world_triangles()
        self.reset()

    @property
    def unwrapped(self):
        return self

    def validate_height(self, value):
        if self.max_height:
            if value is not None:
                raise ValueError("Maximum-height jumping has no height command")
            return
        if value is not None and (not math.isfinite(value) or not 0 <= value <= self.task_cfg["height_max"]):
            raise ValueError(f"height must be in [0, {self.task_cfg['height_max']}] metres")

    def set_height(self, value):
        """Latch requests on the next reset; never change a flight target midway."""
        self.validate_height(value)
        self.fixed_height = value

    def draw_playback_ui(self, ui):
        ui.separator()
        ui.text("Trampoline jump | World 0")
        if self.max_height:
            ui.text("Jump as high as possible")
            ui.text(f"Peak COM rise: {(self.cycle.peak[0] - self.cycle.reference[0]).item() * 100:.1f} cm")
            ui.text("R: reset all environments | Right-drag: apply force | Space: pause")
            return
        # Show the latched command separately from a request made during flight.
        current = self.cycle.target[0].item()
        ui.text(f"Current target: {current * 100:.1f} cm")
        if self.fixed_height is not None:
            ui.text(f"Next target: {self.fixed_height * 100:.1f} cm")
        ui.text("Height = COM rise above settled reference")
        ui.text("+/-: next target | 0: stand | R: reset all environments")
        ui.text("Right-drag robot: apply force | Space: pause")

    def _world_triangles(self):
        topology = wp.to_torch(self.model.tri_indices).long()
        world = wp.to_torch(self.model.particle_world)[topology[:, 0]]
        return torch.stack([topology[world == i] for i in range(self.num_envs)])

    def surface(self, points):
        """Batched barycentric height, velocity and normal queries on current cloth faces."""
        tri = wp.to_torch(self.base.states[0].particle_q)[self._surface_topology]
        velocities = wp.to_torch(self.base.states[0].particle_qd)[self._surface_topology]
        a, b, c = tri.unbind(2)
        u, v = b - a, c - a
        delta = points[:, :, None, :2] - a[:, None, :, :2]
        det = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
        valid = det.abs() > 1e-10
        denominator = torch.where(valid, det, 1.0)[:, None]
        s = (delta[..., 0] * v[:, None, :, 1] - delta[..., 1] * v[:, None, :, 0]) / denominator
        t = (u[:, None, :, 0] * delta[..., 1] - u[:, None, :, 1] * delta[..., 0]) / denominator
        inside = valid[:, None] & (s >= -1e-5) & (t >= -1e-5) & (s + t <= 1.00001)
        z = a[:, None, :, 2] + s * u[:, None, :, 2] + t * v[:, None, :, 2]
        chosen = torch.where(inside, z, -torch.inf).argmax(-1)
        weights = torch.stack((1 - s - t, s, t), -1)
        rows = torch.arange(self.num_envs, device=self.device)[:, None]
        cols = torch.arange(points.shape[1], device=self.device)[None, :]
        bary = weights[rows, cols, chosen]
        height = z[rows, cols, chosen]
        speed = (velocities[rows, chosen] * bary[..., None]).sum(-2)
        normal = torch.nn.functional.normalize(torch.cross(u, v, dim=-1), dim=-1)[rows, chosen]
        hit = inside.any(-1)
        return torch.where(hit, height, 0.0), torch.where(hit[..., None], speed, 0.0), normal, hit

    def state(self):
        env = self.base
        q, qd, pose, linear, angular, gravity = env._kinematics()
        bodies = wp.to_torch(env.states[0].body_q).reshape(self.num_envs, -1, 7)
        twists = wp.to_torch(env.states[0].body_qd).reshape(self.num_envs, -1, 6)
        local_com = wp.to_torch(self.model.body_com).reshape(self.num_envs, -1, 3)
        masses = wp.to_torch(self.model.body_mass).reshape(self.num_envs, -1)
        centers = bodies[..., :3] + rotate(bodies[..., 3:], local_com)
        com = (centers * masses[..., None]).sum(1) / masses.sum(1, keepdim=True)
        com_v = (twists[..., :3] * masses[..., None]).sum(1) / masses.sum(1, keepdim=True)
        feet, foot_v = [], []
        for body, local in env.official.sites[1:]:
            p = bodies[:, body]
            feet.append(p[:, :3] + rotate(p[:, 3:], local[:3].expand(self.num_envs, -1)))
            offset = rotate(p[:, 3:], local[:3].expand(self.num_envs, -1) - local_com[:, body])
            foot_v.append(twists[:, body, :3] + torch.cross(twists[:, body, 3:], offset, dim=-1))
        feet, foot_v = torch.stack(feet, 1), torch.stack(foot_v, 1)
        points = torch.cat((feet, pose[:, None, :3]), 1)
        height, surface_v, normal, hit = self.surface(points)
        # Only cloth contact counts as support; standing on a frame or the floor is not a jump landing.
        contact = self.sensors.feet.data.soft_found > 0
        return SimpleNamespace(
            q=q[:, env.qi] - env.home,
            qd=qd[:, env.qdi],
            pose=pose,
            linear=linear,
            angular=angular,
            gravity=gravity,
            com_z=com[:, 2],
            com_vz=com_v[:, 2],
            support_relative_speed=(com_v - surface_v[:, 2]).norm(dim=-1),
            leg_height=com[:, 2] - feet[..., 2].mean(-1),
            leg_extension_speed=com_v[:, 2] - foot_v[..., 2].mean(-1),
            speed=com_v.norm(dim=-1),
            omega=angular.norm(dim=-1),
            drift=pose[:, :2].norm(dim=-1),
            tilt=torch.acos((-gravity[:, 2]).clamp(-1, 1)),
            contact=contact,
            foot_clearance=feet[..., 2] - height[:, :2],
            relative_foot_v=foot_v - surface_v[:, :2],
            bed_z=height[:, 2],
            root_clearance=pose[:, 2] - height[:, 2],
            height=height,
            surface_v=surface_v,
            normal=normal,
            outside=self.terrain.out_of_bounds(pose) | ~hit.all(-1),
            bad_contact=(self.sensors.bad_contact | (self.sensors.feet.data.found > self.sensors.feet.data.soft_found).any(-1)),
        )

    def _imu_point_velocity(self, state):
        # Root linear velocity is expressed in trunk axes at the link origin.
        local = state.linear + torch.cross(state.angular, self.imu_offset.expand_as(state.angular), dim=-1)
        return rotate(state.pose[:, 3:7], local)

    def _update_sensors(self, state):
        velocity = self._imu_point_velocity(state)
        raw = specific_force(self.imu_velocity, velocity, state.pose[:, 3:7], self.step_dt)
        self.imu_velocity.copy_(velocity)
        alpha = self.task_cfg["accel_filter_alpha"]
        self.acceleration.lerp_(raw, alpha)
        measured = (self.acceleration + self.sensor_bias
                    + torch.randn_like(raw) * self.task_cfg["accel_noise_std"]).clamp(-39.24, 39.24)
        frame = sensor_frame(state, measured, self.previous)
        frame[:, :3] += torch.randn_like(state.angular) * self.task_cfg["gyro_noise_std"] * 0.2
        frame[:, 3:6] += torch.randn_like(state.gravity) * self.task_cfg["gravity_noise_std"]
        frame[:, 3:6] = torch.nn.functional.normalize(frame[:, 3:6], dim=-1)
        frame[:, 9:23] += torch.randn_like(state.q) * self.task_cfg["joint_pos_noise_std"]
        frame[:, 23:37] += torch.randn_like(state.qd) * self.task_cfg["joint_vel_noise_std"] * 0.05
        self.sensor_observation.copy_(self.sensor_previous)
        self.sensor_previous.copy_(frame)
        # Action feedback is known immediately; only sensor channels have transport delay.
        self.sensor_observation[:, 37:] = self.previous

    def _reset(self, ids):
        env = self.base
        self.terrain.reset(ids)
        for state in env.states:
            q, qd, _, _ = env._views(state)
            q[ids] = env.initial_q[ids]
            q[ids, :2] = 0
            q[ids, 2] += SCENE_CONFIG.trampoline_height + 0.005
            qd[ids] = 0
            env.sync_selected_state(state, ids)
        env.actions[ids] = 0
        self.previous[ids] = 0
        self.episode_length_buf[ids] = 0
        self.completed[ids] = False
        self.sensors.reset(ids)
        self.sensors.bad_contact[ids] = False
        # Clear unused official caches, but never invoke its rewards, events or commands.
        env.official.observations.reset(ids)
        if self.history is not None:
            self.history.reset(ids)
        targets = None
        if not self.max_height:
            targets = torch.empty(len(ids), device=self.device).uniform_(self.task_cfg["height_min"], self.height_limit)
            targets[torch.rand(len(ids), device=self.device) < self.task_cfg["zero_probability"]] = 0
            if self.fixed_height is not None:
                targets.fill_(self.fixed_height)
        state = self.state()
        self.cycle.reset(ids, targets, state.com_z[ids])
        if self.rebound:
            self.imu_velocity[ids] = self._imu_point_velocity(state)[ids]
            self.acceleration[ids] = -state.gravity[ids] * 9.81
            self.sensor_bias[ids] = torch.randn(len(ids), 3, device=self.device) * self.task_cfg["accel_bias_std"]
            frame = sensor_frame(state, self.acceleration, self.previous)
            self.sensor_previous[ids] = frame[ids]
            self.sensor_observation[ids] = frame[ids]
        self._cache = None
        env._sync()

    def reset(self):
        self._reset(torch.arange(self.num_envs, device=self.device))
        return self.get_observations()

    def get_observations(self):
        if self._cache is None:
            s, cycle = self.state(), self.cycle
            contact_time = self.sensors.feet.data.current_contact_time.clamp(max=1)
            frame = torch.cat(
                (
                    s.angular * 0.2,
                    s.gravity,
                    s.q,
                    s.qd * 0.05,
                    self.previous,
                    s.linear,
                    s.contact.float(),
                    contact_time,
                    s.foot_clearance * 5,
                    s.relative_foot_v.flatten(1),
                    (s.height - SCENE_CONFIG.trampoline_height) * 5,
                    s.surface_v[..., 2],
                    s.normal[:, 2, :2],
                    s.pose[:, :2] * 5,
                    (s.com_z - cycle.reference)[:, None] * 5,
                    s.root_clearance[:, None] * 5,
                ),
                -1,
            ).clamp(-10, 10)
            phase = torch.nn.functional.one_hot(cycle.phase, 4).float()
            current = torch.cat(
                (
                    frame,
                    *(() if self.max_height else (cycle.target[:, None] * 5,)),
                    phase,
                    cycle.phase_age[:, None].clamp(max=6),
                    cycle.stable_time[:, None],
                    cycle.air_time[:, None],
                ),
                -1,
            )
            actor_current = self.sensor_observation if self.rebound else current
            extra = torch.stack(
                (cycle.peak - cycle.reference, cycle.compression, cycle.score, cycle.touch_time[:, 0], cycle.touch_time[:, 1]),
                -1,
            )
            # PPO retains this observation until after step(); sensor/reset buffers mutate in place.
            actor = actor_current.clone() if self.history is None else actor_current
            critic = torch.cat((current, extra), -1)
            if self.history is not None:
                history = self.history.read(actor_current if self.rebound else frame)
                actor = torch.cat((actor, history), -1)
                critic = torch.cat((critic, history), -1)
            if actor.shape[1] != self.num_obs or critic.shape[1] != self.num_privileged_obs:
                raise RuntimeError(f"Jump observation mismatch: {actor.shape}, {critic.shape}")
            if not torch.isfinite(actor).all() or not torch.isfinite(critic).all():
                raise FloatingPointError("Non-finite jump observation")
            self._cache = actor, {"observations": {"critic": critic}}
        return self._cache

    def curriculum(self, targets, successes):
        if self.max_height or self.fixed_height is not None:
            return
        # Only the upper half of the current range can unlock harder heights.
        eligible = targets >= (self.task_cfg["height_min"] + self.height_limit) / 2
        self.outcomes.extend(successes[eligible].float().cpu().tolist())
        if len(self.outcomes) == self.outcomes.maxlen and np.mean(self.outcomes) >= self.task_cfg["curriculum_success"]:
            self.height_limit = min(self.task_cfg["height_max"], self.height_limit + self.task_cfg["height_increment"])
            self.outcomes.clear()

    @torch.no_grad()
    def step(self, actions):
        if actions.shape != self.previous.shape or not torch.isfinite(actions).all():
            raise ValueError("Expected finite jump actions with shape (num_envs, 14)")
        actions = actions.to(self.device).clamp(-1, 1)
        if self.history is not None:
            self.history.advance()
        self.sensors.bad_contact.zero_()
        # Both modes need contact sensors; only playback accepts interactive viewer forces.
        torque = self.base.advance_physics(
            actions * self.task_cfg["action_scale"], training=True, apply_viewer_forces=not self.auto_reset
        )
        self.base.official.common_step_counter += 1
        self.base.sim_time += self.step_dt
        self.episode_length_buf += 1
        self.base._sync()
        for name in ("particle_q", "particle_qd"):
            if not torch.isfinite(wp.to_torch(getattr(self.base.states[0], name))).all():
                raise FloatingPointError(f"Non-finite trampoline {name}")
        if int(self.base.contacts.soft_contact_count.numpy()[0]) >= self.base.contacts.soft_contact_max:
            raise RuntimeError("Trampoline contact capacity exhausted")
        state = self.state()
        terms, success, failure = self.cycle.advance(state, self.step_dt)
        continuous = self.max_height and self.task_cfg.get("continuous_bounce", False)
        peak_height = (self.cycle.peak - self.cycle.reference).clamp_min(0).clone()
        stable_seconds = self.cycle.stable_time.clone()
        touchdown_delta = torch.where(
            (self.cycle.touch_time >= 0).all(-1),
            (self.cycle.touch_time[:, 0] - self.cycle.touch_time[:, 1]).abs(),
            -1.0,
        )
        phase_snapshot = self.cycle.phase.clone()
        terms.update(
            horizontal_speed=state.linear[:, :2].square().sum(-1) * self.step_dt,
            angular_speed=state.angular.square().sum(-1) * self.step_dt,
            slip=(state.relative_foot_v[..., :2].square().sum(-1) * state.contact).sum(-1) * self.step_dt,
            torque=torque.square().sum(-1) * self.step_dt,
            action_rate=(actions - self.previous).square().sum(-1) * self.step_dt,
        )
        reward = sum(self.config["rewards"][name] * value for name, value in terms.items())
        timeout = torch.zeros_like(failure)
        if self.rebound and self.auto_reset:
            timeout = (self.episode_length_buf >= self.max_episode_length) & ~failure
        done = (failure | timeout) if continuous else success | failure
        if not self.auto_reset:
            # Report each attempt once, but continue physics and policy until the user resets.
            done &= ~self.completed
            self.completed |= done
        self.previous.copy_(actions)
        if self.rebound:
            self._update_sensors(state)
        self._cache = None
        if continuous and not self.rebound:
            landed_ids = (success & ~failure).nonzero().flatten()
            self.cycle.rearm(landed_ids, state.com_z[landed_ids])
        result = {
            "height": peak_height,
            "success": success.clone(),
            "failure": failure.clone(),
            "tilt": state.tilt.clone(),
            "drift": state.drift.clone(),
            "stable_seconds": stable_seconds,
            "touchdown_delta": touchdown_delta,
        }
        if self.rebound:
            result.update(**{k: v.clone() for k, v in self.cycle.events.items()},
                          bounce_count=self.cycle.bounce_count.clone(),
                          contact_to_takeoff=self.cycle.contact_to_takeoff.clone(), timeout=timeout.clone())
        if not self.max_height:
            result["target"] = self.cycle.target.clone()
        ids = done.nonzero().flatten()
        log = {f"Jump/{name}": value.mean() for name, value in terms.items()}
        if not self.max_height:
            log["Jump/height_limit"] = self.height_limit
        log["Jump/peak_height_m"] = peak_height.mean()
        for phase, name in enumerate(("prepare", "load", "flight", "land")):
            log[f"Jump/phase_{name}"] = (phase_snapshot == phase).float().mean()
        if self.cycle.shaped:
            log["Jump/crouch_depth"] = self.cycle.crouch.mean()
            log["Jump/compression_depth"] = self.cycle.compression.mean()
            log["Jump/push_progress"] = self.cycle.push_progress.mean()
        if self.rebound:
            log["Jump/bounce_count"] = self.cycle.bounce_count.mean()
            log["Jump/last_landed_height_m"] = self.cycle.last_height.mean()
            log["Jump/contact_to_takeoff_s"] = self.cycle.contact_to_takeoff.mean()
            log["Jump/timeouts"] = timeout.float().mean()
            landed = success.float()
            log["Jump/landed_height_m"] = (self.cycle.last_height * landed).sum() / landed.sum().clamp_min(1)
            later = (success & (self.cycle.bounce_count >= 3)).float()
            log["Jump/later_height_m"] = (self.cycle.last_height * later).sum() / later.sum().clamp_min(1)
            log["Jump/later_landing_rate"] = later.mean()
        if self.auto_reset and len(ids):
            if not self.max_height:
                self.curriculum(self.cycle.target[ids], success[ids])
            self._reset(ids)
        observations, extras = self.get_observations()
        extras = {**extras, "log": log, "jump": result}
        if self.rebound:
            extras["time_outs"] = timeout
        if self.base.viewer:
            self.base.render()
        return observations, reward, done, extras

    def close(self):
        self.base.close()


def source_fingerprint():
    # Hash executable dependencies, not the unrelated official rewards or documentation.
    files = (
        "jump.py",
        "rebound.py",
        "env.py",
        "scenes.py",
        "sensors.py",
        "policy_spec.py",
        "adapter.py",
        "stability.py",
        "ppo.py",
        "recipe.py",
        "events.py",
        "observations.py",
        "configs/official_snapshot.json",
    )
    digest = hashlib.sha256()
    for name in files:
        digest.update((Path(__file__).parent / name).read_bytes())
    return digest.hexdigest()


def runner_config(config, num_envs):
    result = ppo_config(num_envs)
    for key, value in config["ppo"].items():
        if isinstance(value, dict):
            result[key].update(value)
        else:
            result[key] = value
    result["policy"].update(class_name="JumpActorCritic", history_length=config["task"]["history_length"])
    legacy = config["contract"] == LEGACY_CONTRACT
    result["policy"].update(current_dim=CURRENT_DIM + int(legacy), critic_dim=CRITIC_DIM + int(legacy))
    if config["contract"] in REBOUND_CONTRACTS:
        result["policy"].update(current_dim=SENSOR_DIM, critic_dim=CRITIC_DIM, frame_dim=SENSOR_DIM)
    # Physical units are explicitly scaled, equally for current and historical frames.
    result["empirical_normalization"] = False
    return result


def create_runner(env, config, log_dir=None):
    from rsl_rl.runners import on_policy_runner

    class JumpRunner(on_policy_runner.OnPolicyRunner):
        def save(self, path, infos=None):
            infos = {"contract": env.contract}
            if not env.max_height:
                infos.update(height_limit=env.height_limit, outcomes=list(env.outcomes))
            super().save(path, infos)

        def load(self, path, load_optimizer=True):
            infos = super().load(path, load_optimizer)
            if not infos or infos.get("contract") != env.contract:
                raise ValueError("Checkpoint jump objective/observation contract does not match")
            if not env.max_height:
                env.height_limit = float(infos["height_limit"])
                env.outcomes.clear()
                env.outcomes.extend(infos["outcomes"])
            return infos

    # Register a distinct name only; never replace the official policy or runner classes.
    on_policy_runner.JumpActorCritic = JumpActorCritic
    config = deepcopy(config)
    config["policy"].update(current_dim=env.current_dim, critic_dim=env.critic_dim, frame_dim=env.frame_dim)
    runner = JumpRunner(env, config, log_dir=log_dir, device=str(env.device))
    runner.git_status_repos = []
    return runner


def checkpoint_metadata(path, *, allow_source_mismatch=False):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Checkpoint does not exist: {path}")
    metadata = json.loads(path.with_name("config.json").read_text())
    if (metadata.get("observation_contract") not in SUPPORTED_CONTRACTS
            or metadata.get("task") != "trampoline_jump"):
        raise ValueError("Only trampoline_jump checkpoints are accepted; official policies are incompatible")
    if metadata["source_fingerprint"] != source_fingerprint():
        if not allow_source_mismatch:
            raise ValueError(
                "Jump source changed since checkpoint creation; strict replay/resume rejected. "
                "For diagnostic playback only, use play --allow-source-mismatch."
            )
        print(
            "WARNING: source mismatch accepted for diagnostic playback or explicit warm-start; "
            "this is not an exact reproduction of the training environment.",
            flush=True,
        )
    current = {key: value for key, value in vars(SCENE_CONFIG).items() if key.startswith("trampoline_")}
    saved = {key: value for key, value in metadata["terrain"]["config"].items() if key.startswith("trampoline_")}
    if current != saved:
        raise ValueError("Trampoline materials/geometry changed since checkpoint creation")
    return metadata


def playback_keys(max_height=False):
    from pyglet.window import key

    if max_height:
        return (("reset", (key.R,), None),)
    # ViewerGL string lookup excludes punctuation; pass native key symbols instead.
    return (
        ("increase", (key.EQUAL, key.PLUS, key.NUM_ADD), 0.01),
        ("decrease", (key.MINUS, key.UNDERSCORE, key.NUM_SUBTRACT), -0.01),
        ("stand", (key._0, key.NUM_0), 0.0),
        ("reset", (key.R,), None),
    )


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "play"))
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path, help="Required for play; resumes training when used with train")
    parser.add_argument(
        "--init-from", type=Path,
        help="Optional warm-start checkpoint for train; loads policy weights but resets optimizer and task state",
    )
    parser.add_argument(
        "--allow-source-mismatch", action="store_true",
        help="Diagnostic play only: allow changed source; retain contract and trampoline material checks",
    )
    parser.add_argument("--num-envs", type=int, help="Default: 8 for train, 1 for play")
    parser.add_argument("--iterations", type=int, default=1000, help="Additional PPO iterations")
    parser.add_argument("--height", type=float,
                        help="Legacy height-command checkpoints only; unavailable for maximum-height training")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--viewer", choices=("null", "gl"), default=None)
    parser.add_argument("--num-frames", type=int, default=3000)
    parser.add_argument("--log-root", type=Path, default=Path("logs/rsl_rl/microduck_jump"))
    parser.add_argument("--report", type=Path, help="Write per-attempt playback results as JSON")
    parser.add_argument("--record-dir", type=Path, help="Record single-world playback geometry for Blender in a new directory")
    return parser


def main(argv=None):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    parser = create_parser()
    args = parser.parse_args(argv)
    count = args.num_envs if args.num_envs is not None else (8 if args.mode == "train" else 1)
    if count < 1 or args.iterations < 1 or args.num_frames < 1:
        parser.error("num-envs, iterations and num-frames must be positive")
    if args.record_dir and (args.mode != "play" or count != 1):
        parser.error("--record-dir requires play with exactly one environment")
    if args.mode == "play" and args.checkpoint is None:
        parser.error("play requires --checkpoint")
    if args.mode == "play" and args.init_from is not None:
        parser.error("--init-from is only supported for train")
    if args.mode == "train" and args.checkpoint is not None and args.init_from is not None:
        parser.error("--checkpoint and --init-from are mutually exclusive")
    if args.allow_source_mismatch and args.mode != "play":
        parser.error("--allow-source-mismatch is only supported for play; training resume remains strict")
    metadata = (
        checkpoint_metadata(args.checkpoint, allow_source_mismatch=args.allow_source_mismatch)
        if args.checkpoint else None
    )
    init_metadata = (
        checkpoint_metadata(args.init_from, allow_source_mismatch=True)
        if args.init_from else None
    )
    config = metadata["jump"] if metadata else load_config(args.config)
    if init_metadata:
        if init_metadata["observation_contract"] != config.get("contract", CONTRACT):
            parser.error("--init-from checkpoint uses a different jump observation contract")
        if init_metadata["jump"]["task"].get("shaping_version") != config["task"].get("shaping_version"):
            parser.error("--init-from checkpoint uses a different jump shaping version")
    maximum = config["task"].get("shaping_version", 1) >= 4
    if args.mode == "train" and not maximum:
        parser.error("New training uses maximum-height jumping; legacy height-command checkpoints are play-only")
    if maximum and args.height is not None:
        parser.error("Maximum-height jumping has no --height command")
    height = args.height
    if args.mode == "train" and metadata:
        saved_height = metadata.get("fixed_height")
        if height is not None and height != saved_height:
            parser.error("Strict resume cannot change the saved height sampling mode")
        height = saved_height
    if args.mode == "play" and height is None and not maximum:
        height = 0.05
    if height is not None and (not math.isfinite(height) or not 0 <= height <= config["task"]["height_max"]):
        parser.error("height must be finite, nonnegative and no larger than task.height_max")
    wp.set_device(args.device)
    if not wp.get_device().is_cuda:
        parser.error("SynReal jump physics requires CUDA")
    login()
    viewer = None
    viewer_kind = args.viewer or ("null" if args.mode == "train" else "gl")
    if viewer_kind == "gl":
        from newton.viewer import ViewerGL

        viewer = ViewerGL(width=1280, height=900)
    env = None
    recorder = None
    recording_completed = False
    try:
        env = JumpEnv(
            config, count, args.device, viewer, args.seed, height, PhysicsConfig(**metadata["physics"]) if metadata else None,
            auto_reset=args.mode == "train",
        )
        ppo = metadata["ppo"] if metadata else runner_config(config, count)
        log_dir = None
        if args.mode == "train":
            stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d_%H-%M-%S_%f")
            log_dir = args.log_root / stamp
            log_dir.mkdir(parents=True)
            record = {
                **env.cfg,
                "ppo": ppo,
                "source_fingerprint": source_fingerprint(),
                **({} if maximum else {"fixed_height": height}),
                "seed": args.seed,
                "num_envs": count,
                "resume_from": str(args.checkpoint) if args.checkpoint else None,
                "init_from": str(args.init_from) if args.init_from else None,
            }
            (log_dir / "config.json").write_text(json.dumps(record, indent=2) + chr(10))
            # Keep the new untracked source reviewable without dumping unrelated worktree content.
            (log_dir / "jump.py").write_text(Path(__file__).read_text())
            (log_dir / "rebound.py").write_text(Path(__file__).with_name("rebound.py").read_text())
        runner = create_runner(env, ppo, str(log_dir) if log_dir else None)
        load_path = args.checkpoint or args.init_from
        if load_path:
            runner.load(str(load_path), load_optimizer=args.checkpoint is not None and args.mode == "train")
            if args.checkpoint and args.mode == "train":
                runner.current_learning_iteration += 1
                env.base.official.common_step_counter = runner.current_learning_iteration * ppo["num_steps_per_env"]
            else:
                runner.current_learning_iteration = 0
                env.base.official.common_step_counter = 0
            env.reset()
        if args.mode == "train":
            print(f"Training unassisted maximum-height jumps: {log_dir.resolve()}", flush=True)
            runner.learn(args.iterations, init_at_random_ep_len=False)
        else:
            if args.record_dir:
                from demos.microduck_soft_pad.recording import PlaybackRecording

                recorder = PlaybackRecording(env.base, args)
            if viewer:
                viewer.register_ui_callback(env.draw_playback_ui, position="side")
            policy = runner.get_inference_policy(device=args.device)
            obs, _ = env.get_observations()
            report, keys = [], {}
            bindings = playback_keys(maximum) if viewer else ()
            if maximum:
                print("Jump as high as possible. R: reset all environments. No auto reset.", flush=True)
            else:
                print("Height-command policy. +/-: next target; 0: stand; R: reset. No auto reset.", flush=True)
            print("Right-drag the robot to apply force; Space pauses/resumes physics.", flush=True)
            frame = 0
            with torch.inference_mode():
                while frame < args.num_frames:
                    if viewer and not viewer.is_running():
                        break
                    start = time.monotonic()
                    if viewer:
                        for action, symbols, delta in bindings:
                            down = any(viewer.is_key_down(symbol) for symbol in symbols)
                            if down and not keys.get(action, False):
                                if action == "reset":
                                    obs, _ = env.reset()
                                else:
                                    target = 0.0 if action == "stand" else env.fixed_height + delta
                                    env.set_height(float(np.clip(target, 0, config["task"]["height_max"])))
                                    print(f"Next jump height: {env.fixed_height:.3f} m", flush=True)
                            keys[action] = down
                        if not viewer.should_step():
                            env.base.render()
                            continue
                    obs, _, done, extras = env.step(policy(obs))
                    if recorder:
                        recorder.sample()
                    reported = done | extras["jump"]["success"]
                    if env.rebound:
                        reported |= extras["jump"]["takeoff"] | extras["jump"]["apex"] | extras["jump"]["touchdown"]
                    for index in reported.nonzero().flatten().tolist():
                        report.append(
                            {
                                "frame": frame,
                                "world": index,
                                **{key: value[index].item() for key, value in extras["jump"].items()},
                            }
                        )
                    if viewer:
                        time.sleep(max(0, env.step_dt - (time.monotonic() - start)))
                    frame += 1
            recording_completed = frame == args.num_frames
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps(report, indent=2) + chr(10))
            print(f"Recorded events: {len(report)}; successful landings: {sum(r['success'] for r in report)}")
    finally:
        if recorder:
            recorder.close(completed=recording_completed)
        if env:
            env.close()
        elif viewer:
            viewer.close()


if __name__ == "__main__":
    from examples.microduck.jump import main as jump_main

    jump_main()
