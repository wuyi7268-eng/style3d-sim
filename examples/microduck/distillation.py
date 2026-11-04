"""History-policy VelStand warm start and frozen SynReal teachers for RSL-RL 2.3."""

import copy
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from torch import nn

from .checkpoint import checkpoint_policy_variant, initialize_transfer, resolve_checkpoint
from .observations import CONTRACT, HISTORY_SIZE
from .policy import HistoryActorCritic, HistoryNormalization
from .policy_spec import EFFORT, KD, KP

BC_VERSION = 2


def finite_tensors(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tensors(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_tensors(item) for item in value)
    return True


def teacher_descriptor(path, task):
    path = resolve_checkpoint(path)
    metadata = json.loads(path.with_name("config.json").read_text())
    expected = {
        "task": task, "solver": "synreal", "actuator": "position_pd", "self_collisions": False,
        "observation_contract": CONTRACT,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"{task} teacher requires {key}={value!r}")
    if checkpoint_policy_variant(metadata) != "history" or not metadata["ppo"]["empirical_normalization"]:
        raise ValueError("VelStand teachers require normalized history policies")
    if metadata.get("recipe") != "official_adapted_v1":
        raise ValueError("Teacher requires official_adapted_v1")
    for key, value in (("kp", KP), ("kd", KD), ("effort_limit", EFFORT), ("policy_dt", 0.02)):
        if not math.isclose(metadata.get(key, float("nan")), value, rel_tol=1e-6):
            raise ValueError(f"Incompatible teacher {key}")
    if metadata.get("terrain", {}).get("kind", "default") != "default":
        raise ValueError("Initial VelStand distillation requires flat-ground teachers")
    alignment = metadata["official_alignment"]
    if (alignment["actor_dim"], alignment["critic_dim"]) != (61, 70 if task == "velocity" else 68):
        raise ValueError("Unexpected teacher observation dimensions")
    return {
        "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "task": task, "policy": metadata["ppo"]["policy"], "critic_dim": alignment["critic_dim"],
        "physics": metadata["physics"], "history": metadata["history"],
        "source_recipe_fingerprint": alignment["recipe_fingerprint"],
    }


def teacher_config(velocity, standup):
    walk, stand = teacher_descriptor(velocity, "velocity"), teacher_descriptor(standup, "standup")
    if walk["physics"] != stand["physics"] or walk["history"] != stand["history"]:
        raise ValueError("Teacher physics and history contracts must match")
    return {
        "version": BC_VERSION, "walk": walk, "stand": stand,
        "learning_rate": 3e-4, "epochs": 5, "mini_batches": 4, "min_mini_batch": 512,
        "min_samples": 64, "walk_coef": 1.0, "stand_coef": 1.0,
        "walk_tilt_deg": 25.0, "stand_tilt_deg": 35.0, "warm_start_optimizer": "inherit",
    }


def validate_environment(config, env):
    from .env import PhysicsConfig, physics_matches

    if env.task != "velstand" or env.backend != "synreal" or env.mixed_terrain or env.cfg["self_collisions"]:
        raise ValueError("Teacher BC requires SynReal flat VelStand without self collisions")
    source = PhysicsConfig(**config["walk"]["physics"])
    # Numerical refinement preserves the 50 Hz policy interface and actuator law.
    # Permit more joint solve iterations, but not unrelated solver/material changes.
    subdivision = env.physics.decimation >= source.decimation and env.physics.decimation % source.decimation == 0
    joint_refinement = env.physics.joint_constraint_primal_iterations >= source.joint_constraint_primal_iterations
    expected = replace(source.with_substeps(env.physics.decimation),
                       joint_constraint_primal_iterations=env.physics.joint_constraint_primal_iterations)
    if not subdivision or not joint_refinement or not physics_matches(env.physics, expected):
        raise ValueError("VelStand physics differs from its teachers; "
                         "only time subdivision and joint solve refinement are supported")


class FrozenTeacher(nn.Module):
    def __init__(self, descriptor, device):
        super().__init__()
        options = copy.deepcopy(descriptor["policy"])
        options.pop("class_name")
        self.policy = HistoryActorCritic(61 + HISTORY_SIZE, descriptor["critic_dim"] + HISTORY_SIZE, 14, **options)
        self.normalizer = HistoryNormalization(61)
        self.to(device)
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, raw, *, standing=False):
        if standing:
            raw = raw.clone()
            # Only current commands change: past proprioception includes the student's executed actions.
            raw[:, 48:51] = 0
        result = self.policy.act_inference(self.normalizer(raw))
        if not torch.isfinite(result).all():
            raise FloatingPointError("Non-finite teacher action")
        return result

    def load_source(self, descriptor):
        path = Path(descriptor["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != descriptor["sha256"]:
            raise ValueError("Teacher checkpoint changed after preflight")
        checkpoint = torch.load(path, map_location=next(self.parameters()).device, weights_only=True)
        if not finite_tensors(checkpoint):
            raise ValueError("Non-finite teacher checkpoint")
        self.policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.normalizer.load_state_dict(checkpoint["obs_norm_state_dict"], strict=True)


def teacher_masks(raw, config):
    gravity = raw[:, 3:6]
    upright_cos = -gravity[:, 2] / gravity.norm(dim=-1).clamp_min(1e-8)
    walk = upright_cos > math.cos(math.radians(config["walk_tilt_deg"]))
    fallen = upright_cos < math.cos(math.radians(config["stand_tilt_deg"]))
    # Standing body commands must not be anchored to a walk teacher that ignores body pose.
    standing = raw[:, 48:51].abs().amax(-1) <= 1e-6
    body_active = raw[:, [57, 58, 59]].abs().amax(-1) > 1e-6
    body_frames = standing & body_active
    body_tilt = body_frames & (raw[:, 57].abs() <= 1e-6)
    return walk & ~body_frames, fallen | body_tilt


class TeacherObservationEnv:
    """Capture raw actor input before normalization, including the initial frame and reset histories."""

    def __init__(self, env):
        self.env = env
        self.raw = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def get_observations(self):
        obs, extras = self.env.get_observations()
        self.raw = obs.detach().clone()
        return obs, extras

    def step(self, actions):
        obs, rewards, dones, extras = self.env.step(actions)
        self.raw = obs.detach().clone()
        return obs, rewards, dones, extras


class DualTeacherPPO(PPO):
    def attach(self, config, observation_env, student_normalizer):
        if self.is_multi_gpu or self.symmetry or self.rnd:
            raise ValueError("Teacher BC currently requires single-device PPO without symmetry/RND")
        if config["version"] != BC_VERSION:
            raise ValueError("Unsupported teacher BC contract")
        if not isinstance(self.policy, HistoryActorCritic):
            raise TypeError("Teacher BC requires a history student")
        for name in ("epochs", "mini_batches", "min_mini_batch", "min_samples"):
            if type(config[name]) is not int or config[name] < 1:
                raise ValueError(f"BC {name} must be a positive integer")
        for name in ("learning_rate", "walk_coef", "stand_coef"):
            if not math.isfinite(config[name]) or config[name] <= 0:
                raise ValueError(f"BC {name} must be finite and positive")
        if not 0 < config["walk_tilt_deg"] < config["stand_tilt_deg"] < 180:
            raise ValueError("BC tilt gates must leave a non-overlapping PPO band")
        if self.policy.actor_dim != 61 or self.policy.critic_dim != 70:
            raise ValueError("VelStand student must use current-frame dimensions 61/70")
        if config["warm_start_optimizer"] not in ("inherit", "reset"):
            raise ValueError("BC warm_start_optimizer must be inherit or reset")
        self.student_normalizer = student_normalizer
        self.bc_config = copy.deepcopy(config)
        self.observation_env = observation_env
        self.teachers = None
        self.actor_parameters = list(self.policy.actor.parameters()) + list(self.policy.actor_history_encoder.parameters())
        self.bc_optimizer = torch.optim.Adam(self.actor_parameters, lr=config["learning_rate"])
        shape = self.storage.observations.shape[:2]
        self.bc_raw = torch.empty_like(self.storage.observations)
        self.bc_targets = torch.zeros(*shape, 14, device=self.device)
        self.bc_masks = torch.zeros(*shape, 2, dtype=torch.bool, device=self.device)

    def ensure_teachers(self, states=None):
        if self.teachers is None:
            teachers = {}
            for name in ("walk", "stand"):
                # Teacher construction must not perturb the student's exploration RNG.
                devices = [torch.device(self.device)] if torch.device(self.device).type == "cuda" else []
                with torch.random.fork_rng(devices=devices):
                    teacher = FrozenTeacher(self.bc_config[name], self.device)
                if states is None:
                    teacher.load_source(self.bc_config[name])
                else:
                    teacher.load_state_dict(states[name], strict=True)
                teachers[name] = teacher
            self.teachers = teachers

    def act(self, obs, critic_obs):
        self.ensure_teachers()
        raw = self.observation_env.raw
        if raw is None or raw.shape != obs.shape or not torch.isfinite(raw).all():
            raise ValueError("Missing or invalid raw teacher observations")
        walk, stand = teacher_masks(raw, self.bc_config)
        index = self.storage.step
        self.bc_raw[index].copy_(raw)
        self.bc_masks[index, :, 0] = walk
        self.bc_masks[index, :, 1] = stand
        self.bc_targets[index].zero_()
        with torch.no_grad():
            if walk.any():
                self.bc_targets[index, walk] = self.teachers["walk"](raw[walk])
            if stand.any():
                self.bc_targets[index, stand] = self.teachers["stand"](raw[stand], standing=True)
        return super().act(obs, critic_obs)

    def update(self):
        # BC re-normalizes raw histories with the latest student statistics after PPO.
        observations = self.bc_raw.flatten(0, 1)
        targets = self.bc_targets.flatten(0, 1)
        masks = self.bc_masks.flatten(0, 1)
        result = super().update()
        result.update(self.bc_update(observations, targets, masks))
        return result

    def bc_update(self, observations, targets, masks):
        cfg = self.bc_config
        stats = {f"bc_{name}_fraction": masks[:, i].float().mean().item() for i, name in enumerate(("walk", "stand"))}
        selected = masks.clone()
        # Official distillation gates only recovery on sample count; walking remains active.
        if selected[:, 1].sum() < cfg["min_samples"]:
            selected[:, 1] = False
        ids = selected.any(-1).nonzero().flatten()
        weights = selected[:, 0] * cfg["walk_coef"] + selected[:, 1] * cfg["stand_coef"]
        totals = torch.zeros(2, device=self.device)
        counts = torch.zeros(2, device=self.device)
        group_updates = torch.zeros(2, device=self.device)
        loss_total, samples, updates = 0.0, 0, 0
        size = max(cfg["min_mini_batch"], len(ids) // cfg["mini_batches"])
        normalizer = self.student_normalizer
        was_training = normalizer.training
        normalizer.eval()
        try:
            for _ in range(cfg["epochs"]):
                shuffled = ids[torch.randperm(len(ids), device=ids.device)]
                for batch in shuffled.split(size):
                    if not len(batch):
                        continue
                    # BC must not count the same rollout again in empirical statistics.
                    with torch.no_grad():
                        normalized = normalizer(observations[batch])
                    prediction = self.policy.act_inference(normalized)
                    errors = (prediction - targets[batch]).square().mean(-1)
                    loss = (weights[batch] * errors).mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite teacher BC loss")
                    self.bc_optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.actor_parameters, self.max_grad_norm, error_if_nonfinite=True)
                    self.bc_optimizer.step()
                    group = selected[batch]
                    totals += (errors.detach()[:, None] * group).sum(0)
                    counts += group.sum(0)
                    group_updates += group.any(0)
                    loss_total += loss.item() * len(batch)
                    samples += len(batch)
                    updates += 1
        finally:
            normalizer.train(was_training)
        for i, name in enumerate(("walk", "stand")):
            stats[f"bc_{name}_samples"] = masks[:, i].sum().item()
            stats[f"bc_{name}_selected"] = selected[:, i].sum().item()
            stats[f"bc_{name}_mse"] = (totals[i] / counts[i].clamp_min(1)).item()
            stats[f"bc_{name}_updates"] = group_updates[i].item()
        return {**stats, "teacher_bc": loss_total / max(samples, 1), "bc_updates": float(updates)}

    def checkpoint_state(self):
        self.ensure_teachers()
        return {
            "version": BC_VERSION, "config": self.bc_config,
            "teachers": {name: teacher.state_dict() for name, teacher in self.teachers.items()},
            "optimizer": self.bc_optimizer.state_dict(),
        }

    def restore_state(self, state):
        if state["version"] != BC_VERSION or state["config"] != self.bc_config:
            raise ValueError("Checkpoint teacher BC configuration mismatch")
        if not finite_tensors(state):
            raise ValueError("Non-finite teacher BC checkpoint")
        self.teachers = None
        self.ensure_teachers(state["teachers"])
        self.bc_optimizer.load_state_dict(state["optimizer"])


class TeacherRunner(OnPolicyRunner):
    def save(self, path, infos=None):
        infos = dict(infos or {})
        infos["teacher_bc"] = self.alg.checkpoint_state()
        super().save(path, infos)

    def load(self, path, load_optimizer=True):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        state = (checkpoint.get("infos") or {}).get("teacher_bc")
        if state is None:
            raise ValueError("Missing teacher BC state; use teacher initialization for a fresh VelStand run")
        if not finite_tensors(checkpoint):
            raise ValueError("Non-finite VelStand checkpoint")
        infos = super().load(path, load_optimizer=load_optimizer)
        if load_optimizer:
            self.alg.learning_rate = self.alg.optimizer.param_groups[0]["lr"]
            self.alg.restore_state(state)
        return infos


def warm_start(runner, config):
    runner.alg.ensure_teachers()
    metadata = initialize_transfer(runner, Path(config["walk"]["path"]))
    mode = config["warm_start_optimizer"]
    if mode == "inherit":
        source_policy = runner.alg.teachers["walk"].policy
        source_layout = [(name, tuple(param.shape)) for name, param in source_policy.named_parameters()]
        student_layout = [(name, tuple(param.shape)) for name, param in runner.alg.policy.named_parameters()]
        if source_layout != student_layout:
            raise ValueError("Velocity optimizer requires identical student parameter layout")
        checkpoint = torch.load(config["walk"]["path"], map_location=runner.device, weights_only=True)
        state = checkpoint.get("optimizer_state_dict")
        if not state or not finite_tensors(state):
            raise ValueError("Velocity checkpoint has no valid Adam state; choose warm_start_optimizer=reset")
        groups = state["param_groups"]
        if len(groups) != 1 or len(groups[0]["params"]) != len(source_layout):
            raise ValueError("Velocity Adam parameter groups do not match the student")
        for key, (_, shape) in zip(groups[0]["params"], source_layout):
            entry = state["state"].get(key, {})
            for field in ("exp_avg", "exp_avg_sq"):
                if field not in entry or tuple(entry[field].shape) != shape:
                    raise ValueError(f"Velocity Adam has incompatible {field} state")
            if "step" not in entry or entry["step"].numel() != 1:
                raise ValueError("Velocity Adam has incompatible step state")
        if not math.isfinite(groups[0]["lr"]) or groups[0]["lr"] <= 0:
            raise ValueError("Velocity Adam learning rate must be finite and positive")
        runner.alg.optimizer.load_state_dict(state)
        # RSL-RL's adaptive schedule owns a separate LR scalar; synchronize both values.
        runner.alg.learning_rate = runner.alg.optimizer.param_groups[0]["lr"]
        metadata["optimizer_restored"] = True
    metadata["warm_start_optimizer"] = mode
    metadata["effective_ppo_learning_rate"] = runner.alg.learning_rate
    metadata["source_recipe_fingerprint"] = config["walk"]["source_recipe_fingerprint"]
    metadata["target_task"] = "velstand"
    return metadata
