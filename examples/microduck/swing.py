"""SwingPump training and manual-reset playback with --solver synreal|mujoco."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

CONFIG = Path(__file__).with_name("configs") / "swing.json"
CONTRACT = "synreal_swing_v1"
ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / "examples/microduck/microduck_rl-develop"
REVISION = "c5fcc50219fef01ac9931d0079c583ccbb29b689"
SOURCE = ROOT / "logs/pretrained/microduck-swing" / REVISION


def prepare_source():
    marker = SOURCE / "source-tree.json"
    if marker.is_file() and json.loads(marker.read_text())["revision"] == REVISION:
        return
    url = f"https://api.github.com/repos/Vottivott/microduck-playground/git/trees/{REVISION}?recursive=1"
    with urllib.request.urlopen(url, timeout=30) as response:
        tree = json.load(response)["tree"]
    files = [x for x in tree if x["type"] == "blob" and (
        x["path"].startswith("src/") or x["path"] in ("LICENSE", "AGENTS.md", "pyproject.toml"))]

    def fetch(entry):
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid upstream source path")
        destination = SOURCE / relative

        def matches(data):
            return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == entry["sha"]

        for candidate in (destination, OFFICIAL / relative):
            if candidate.is_file():
                data = candidate.read_bytes()
                if matches(data):
                    break
        else:
            raw = f"https://raw.githubusercontent.com/Vottivott/microduck-playground/{REVISION}/{relative.as_posix()}"
            with urllib.request.urlopen(raw, timeout=30) as response:
                data = response.read()
        if not matches(data):
            raise ValueError(f"Upstream source checksum mismatch: {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)

    print(f"Preparing upstream SwingPump source {REVISION[:7]}...", flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(fetch, files))
    marker.write_text(json.dumps({"revision": REVISION, "files": files}, indent=2) + "\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "play", "preview"))
    parser.add_argument("--solver", choices=("synreal", "mujoco"), default="synreal")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--pretrained", action="store_true", help="Replay the pinned upstream alpha050 actor")
    parser.add_argument("--allow-source-mismatch", action="store_true")
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--iterations", type=int, default=1000, help="Additional PPO iterations")
    parser.add_argument("--num-frames", type=int, default=1800)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--viewer", choices=("gl", "null"))
    parser.add_argument("--log-root", type=Path, default=Path("logs/rsl_rl/microduck_swing"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    count = args.num_envs if args.num_envs is not None else (8 if args.mode == "train" else 1)
    if min(count, args.iterations, args.num_frames) < 1:
        parser.error("Counts must be positive")
    if args.mode == "play" and args.checkpoint is None and not args.pretrained:
        parser.error("play requires --checkpoint or --pretrained")
    if args.pretrained and (args.mode != "play" or args.checkpoint is not None):
        parser.error("--pretrained is for play only and cannot be combined with --checkpoint")
    if args.mode == "preview" and args.checkpoint is not None:
        parser.error("preview uses HOME targets; use play for checkpoints")
    if args.allow_source_mismatch and args.mode != "play":
        parser.error("Source mismatch is allowed only for diagnostic play")
    if args.config and (args.solver != "synreal" or args.checkpoint):
        parser.error("--config is only for SynReal fresh training/preview/pretrained play; checkpoints restore their config")
    if args.report and args.mode == "train":
        parser.error("--report is for play/preview; training writes TensorBoard logs")
    args.viewer = args.viewer or ("null" if args.mode == "train" else "gl")
    if args.solver == "mujoco" and args.mode == "train" and args.viewer != "null":
        parser.error("MuJoCo training uses --viewer null; use play to view saved checkpoints")
    args.num_envs = count
    args.device = args.device or ("cpu" if args.solver == "mujoco" and args.mode != "train" else "cuda:0")
    for name in ("config", "checkpoint", "log_root", "report"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    return args


def new_log_dir(args):
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d_%H-%M-%S_%f")
    path = args.log_root / f"{stamp}_{args.solver}"
    path.mkdir(parents=True)
    return path


def mujoco_checkpoint_metadata(path, *, allow_source_mismatch=False):
    metadata = json.loads(Path(path).with_name("config.json").read_text())
    if metadata.get("task") != "swing" or metadata.get("solver") != "mujoco":
        raise ValueError("Expected a MuJoCo Swing checkpoint; other tasks/backends are incompatible")
    if metadata.get("revision") != REVISION:
        raise ValueError("MuJoCo Swing checkpoint uses a different upstream revision")
    if metadata["launcher_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        if not allow_source_mismatch:
            raise ValueError("Swing source changed; diagnostic play can use --allow-source-mismatch")
        print("Diagnostic playback with changed source; this does not reproduce the saved training environment.")
    return metadata


def run_mujoco(args):
    executable = OFFICIAL / ".venv/bin/python"
    if not executable.is_file():
        raise FileNotFoundError(f"Official environment is missing: {executable}")
    metadata = (mujoco_checkpoint_metadata(args.checkpoint, allow_source_mismatch=args.allow_source_mismatch)
                if args.checkpoint else None)
    prepare_source()
    if Path(sys.prefix).resolve() != (OFFICIAL / ".venv").resolve():
        environment = os.environ.copy()
        environment.update(PYTHONPATH=os.pathsep.join((str(SOURCE / "src"), str(ROOT))),
                           VIRTUAL_ENV=str(OFFICIAL / ".venv"), PYTHONUNBUFFERED="1")
        # Re-enter this module before importing Newton: the two backends need different RSL-RL versions.
        command = [str(executable), "-m", "examples.microduck.swing", args.mode]
        for key, value in vars(args).items():
            if key != "mode" and value is not None and value is not False:
                command.append("--" + key.replace("_", "-"))
                if value is not True:
                    command.append(str(value))
        os.execve(executable, command, environment)

    # Upstream exposes optional physics/optimizer overrides through environment variables.
    # Restore them before importing the task so resumed runs cannot silently change recipes.
    overrides = {k: v for k, v in os.environ.items() if k.startswith("MICRODUCK_SWING_")}
    if metadata:
        for key in overrides:
            del os.environ[key]
        overrides = metadata["upstream_overrides"]
        os.environ.update(overrides)

    import mjlab_microduck
    import mjlab_microduck.tasks
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends
    from mjlab.viewer import NativeMujocoViewer
    from mjlab_microduck.tasks import mdp

    if not Path(mjlab_microduck.__file__).resolve().is_relative_to(SOURCE):
        raise RuntimeError("The pinned SwingPump source was not selected")
    configure_torch_backends()
    training = args.mode == "train"
    task = "Mjlab-SwingPump-MicroDuck"
    cfg = load_env_cfg(task, play=not training)
    agent = deepcopy(metadata["ppo"]) if metadata else asdict(load_rl_cfg(task))
    cfg.seed = agent["seed"] = args.seed
    cfg.scene.num_envs = args.num_envs
    if not training:
        cfg.terminations.clear()
        # Match the upstream evaluator while keeping geometry violations visible without respawning.
        cfg.rewards.pop("invalid_episode_debt", None)
        cfg.episode_length_s = max(36.0, args.num_frames * 0.02) + 1
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=args.device), clip_actions=agent["clip_actions"])
    try:
        log_dir = None
        if training:
            log_dir = new_log_dir(args)
            record = {"task": "swing", "solver": "mujoco", "revision": REVISION, "ppo": agent,
                      "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      "upstream_overrides": overrides, "num_envs": args.num_envs, "seed": args.seed,
                      "resume_from": str(args.checkpoint) if args.checkpoint else None}
            (log_dir / "config.json").write_text(json.dumps(record, indent=2) + "\n")
            dump_yaml(log_dir / "params/env.yaml", asdict(cfg))
            dump_yaml(log_dir / "params/agent.yaml", agent)
        runner = None
        if args.mode != "preview":
            runner_type = load_runner_cls(task) or MjlabOnPolicyRunner
            runner = runner_type(env, deepcopy(agent), str(log_dir) if log_dir else None, device=args.device)
            checkpoint = args.checkpoint
            if args.pretrained:
                from .swing_policy import pretrained_checkpoint

                checkpoint = pretrained_checkpoint()
            if checkpoint:
                runner.load(str(checkpoint), load_cfg=None if training else {"actor": True},
                            strict=True, map_location=args.device)
                if training:
                    runner.current_learning_iteration += 1
        if training:
            print(f"Swing training (MuJoCo/BAM): {log_dir.resolve()}", flush=True)
            runner.learn(args.iterations, init_at_random_ep_len=False)
        else:
            policy = (runner.get_inference_policy(device=args.device) if runner else
                      lambda obs: torch.zeros((args.num_envs, 14), device=args.device))
            frames = []
            step = env.step
            completed = 0

            def playback_step(actions):
                nonlocal completed
                result = step(actions)
                if not torch.isfinite(result[1]).all() or result[2].any():
                    raise RuntimeError("MuJoCo playback produced an invalid reward or unexpected reset")
                if args.report:
                    angle, rate, lateral, slack, imbalance = mdp._swing_kinematics(env.unwrapped)
                    values = {"angle": angle, "angular_rate": rate, "lateral": lateral * 0.38,
                              "slack": slack, "imbalance": imbalance, "lengths": mdp._swing_string_lengths(env.unwrapped)}
                    frames.append({"frame": completed, **{k: v.cpu().tolist() for k, v in values.items()}})
                completed += 1
                return result

            env.step = playback_step
            print(f"Swing MuJoCo/BAM ({args.device}): R resets, Space pauses. No automatic reset.", flush=True)
            with torch.inference_mode():
                if args.viewer == "gl":
                    NativeMujocoViewer(env, policy).run(num_steps=args.num_frames)
                else:
                    obs = env.get_observations()
                    for _ in range(args.num_frames):
                        obs, _, _, _ = env.step(policy(obs))
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps({"frames": frames, "resets": 0}, indent=2) + "\n")
            print(f"Completed {completed} frames; automatic resets: 0")
    finally:
        env.close()
    return 0


# Dispatch before importing SynReal dependencies, which are absent from the official MuJoCo environment.
if __name__ == "__main__":
    args = parse_args()
    if args.solver == "mujoco":
        raise SystemExit(run_mujoco(args))


import newton
import torch
import warp as wp

from .adapter import rotate
from .env import MicroDuckEnv, PhysicsConfig
from .ppo import create_runner, ppo_config
from .swing_scene import ASSETS, apply_cords, build_swing_robot, mechanism_state
from .swing_task import SwingObjective


def load_config(path=CONFIG):
    config = json.loads(Path(path).read_text())
    validate_config(config)
    return config


def validate_config(config):
    if config["contract"] != CONTRACT:
        raise ValueError("Expected a SynReal SwingPump configuration")
    if len(config["joint_reference"]) != 14 or not all(math.isfinite(v) for v in config["joint_reference"]):
        raise ValueError("Swing needs 14 finite seated reference joint positions")
    g = config["geometry"]
    if not all(math.isfinite(v) and v > 0 for v in g.values()):
        raise ValueError("Swing geometry and stiffness must be finite and positive")
    if not g["length"] < g["hang_length"] < g["limit"] < g["anchor_height"] - g["attachment_z"]:
        raise ValueError("Expected rest length < hanging length < limit < available anchor height")
    if any(not math.isfinite(config[k]) or config[k] <= 0 for k in ("episode_s", "action_scale")):
        raise ValueError("Episode duration and action scale must be finite and positive")
    physics = PhysicsConfig(**config["physics"])
    if not math.isclose(physics.physics_dt * physics.decimation, 0.02) or physics.physics_dt > 0.001:
        raise ValueError("Swing requires 50 Hz control and cord-force substeps no larger than 1 ms")
    if any(not math.isfinite(v) or v < 0 for v in config["noise"].values()):
        raise ValueError("Sensor noise bounds must be finite and nonnegative")
    for term in config["rewards"].values():
        if not math.isfinite(term["weight"]) or any(not math.isfinite(v) or v <= 0 for v in term["params"].values()):
            raise ValueError("Reward weights must be finite and reward scales positive")


class SwingEnv(MicroDuckEnv):
    num_obs, num_privileged_obs = 61, 68

    def __init__(self, config=None, num_envs=8, device="cuda:0", viewer=None, seed=42, *, auto_reset=True):
        self.config = deepcopy(config or load_config())
        validate_config(self.config)
        self.auto_reset = auto_reset
        self.observation_cache = None
        # The shared environment supplies robot topology, saturated PD, collision and state isolation.
        # Its walking recipe is not constructed: swing has independent observations/reset/rewards.
        super().__init__(solver="synreal", num_envs=num_envs, device=device, viewer=viewer, seed=seed,
                         randomize_foot_friction=False, reset_on_fall=False,
                         physics=PhysicsConfig(**self.config["physics"]))
        self.task = self.family = "swing"
        # The pinned source uses SWING_SEATED_FRAME as its default joint/action reference.
        self.home = torch.tensor(self.config["joint_reference"], device=self.device)
        self.common_step_counter = 0
        self.max_episode_length = math.ceil(self.config["episode_s"] / self.step_dt)
        self.objective = SwingObjective(num_envs, self.device, self.config)
        self.policy_actions = torch.zeros_like(self.actions)
        bias = self.config["noise"]["encoder_bias"] if auto_reset else 0.0
        self.encoder_bias = torch.empty_like(self.actions).uniform_(-bias, bias)
        limits = wp.to_torch(self.model.joint_limit_lower).reshape(num_envs, -1)[:, self.qdi]
        upper = wp.to_torch(self.model.joint_limit_upper).reshape(num_envs, -1)[:, self.qdi]
        midpoint, half = (limits + upper) / 2, (upper - limits) * 0.475
        self.soft_lower, self.soft_upper = midpoint - half, midpoint + half
        self.cfg.update(task="swing", observation_contract=CONTRACT, swing=self.config,
                        assistance=False, deployment_compatible=False,
                        adaptation={"actuator": "shared saturated PD; no BAM/electrical randomization",
                                    "cords": "two tension-only springs; explicit substep forces",
                                    "upper_limit": "configured compliant penalty; not MuJoCo tendon solref",
                                    "sensors": "noise and encoder bias; no upstream IMU mounting bias or sensor delay",
                                    "ppo": "local RSL-RL PPO; no upstream planar-correction optimizer"})
        self.cfg.pop("task_recipe", None)
        self.cfg.pop("recipe", None)
        self.reset()
        if viewer:
            viewer.set_world_offsets((1.2, 1.0, 0))
            viewer.set_camera(pos=(1.1, -1.3, 0.85), pitch=-15, yaw=135)

    def build_robot(self, builder_type, **kwargs):
        return build_swing_robot(builder_type, self.config["geometry"], self.config["joint_reference"])

    def create_recipe(self, robot):
        return None

    def apply_task_forces(self):
        g = self.config["geometry"]
        wp.launch(apply_cords, dim=self.num_envs, inputs=[
            self.states[0].body_q, self.model.body_com, self.states[0].body_f,
            self.model.body_count // self.num_envs, self.root, g["attachment_y"], g["attachment_z"],
            g["anchor_height"], g["length"], g["stiffness"], g["limit"], g["limit_stiffness"],
        ], device=self.model.device)

    def _reset(self, ids):
        self._sync()
        for state in self.states:
            q, qd, _, _ = self._views(state)
            q[ids] = self.initial_q[ids]
            qd[ids] = 0
            self.sync_selected_state(state, ids)
        targets = wp.to_torch(self.control.joint_target_q).reshape(self.num_envs, -1)
        columns = self.qi if newton.use_coord_layout_targets else self.qdi
        targets[ids[:, None], columns] = self.home
        self.actions[ids] = 0
        self.policy_actions[ids] = 0
        self.episode_length_buf[ids] = 0
        self.objective.reset(ids)
        self.observation_cache = None
        self._sync()

    def mechanism(self, pose, linear, angular):
        horizon = self.config["rewards"]["swing_predictive_geometry_barrier"]["params"]["horizon_s"]
        return mechanism_state(pose[:, :3], pose[:, 3:], rotate(pose[:, 3:], linear),
                               rotate(pose[:, 3:], angular), self.config["geometry"], horizon)

    def get_observations(self):
        if self.observation_cache is None:
            q, qd, pose, linear, angular, gravity = self._kinematics()
            position, velocity = self.encoder_state(q, qd)
            state = self.mechanism(pose, linear, angular)
            parts = [angular, gravity, position - self.home, velocity]
            clean = torch.cat((*parts, self.policy_actions), -1)
            noisy = []
            for name, value in zip(("angular", "gravity", "position", "velocity"), parts, strict=True):
                value = value.clone()
                if self.auto_reset:
                    bound = self.config["noise"][name]
                    value += torch.empty_like(value).uniform_(-bound, bound)
                    if name == "position":
                        value += self.encoder_bias
                noisy.append(value)
            zeros = torch.zeros(self.num_envs, 10, device=self.device)
            actor = torch.cat((*noisy, self.policy_actions, state["cue"], zeros), -1)
            progress = (self.episode_length_buf / self.max_episode_length).clamp(max=1)
            frontier, validity, mechanism = self.objective.critic_context(state, progress)
            critic = torch.cat((linear, clean, frontier, zeros[:, :4], validity, mechanism), -1)
            self.observation_cache = actor, {"observations": {"critic": critic}}
        return self.observation_cache

    @torch.no_grad()
    def step(self, actions):
        if actions.shape != self.policy_actions.shape or not torch.isfinite(actions).all():
            raise ValueError("Swing expects finite [num_envs, 14] actions")
        previous = self.policy_actions.clone()
        self.policy_actions.copy_(actions.to(self.device).clamp(-1, 1))
        torque = self.advance_physics(self.policy_actions * self.config["action_scale"],
                                      apply_viewer_forces=not self.auto_reset)
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.sim_time += self.step_dt
        q, _, pose, linear, angular, _ = self._kinematics()
        state = self.mechanism(pose, linear, angular)
        progress = (self.episode_length_buf / self.max_episode_length).clamp(max=1)
        terms = self.objective.update(state, progress, self.step_dt)
        position = q[:, self.qi]
        terms.update(dof_pos_limits=((self.soft_lower - position).clamp_min(0)
                                    + (position - self.soft_upper).clamp_min(0)).sum(-1),
                     action_rate_l2=(self.policy_actions - previous).square().sum(-1),
                     joint_torques_l2=torque.square().sum(-1))
        weighted = self.objective.weighted(terms, self.common_step_counter)
        reward = torch.stack(list(weighted.values())).sum(0) * self.step_dt
        if not torch.isfinite(reward).all():
            raise FloatingPointError("Non-finite swing reward")
        timeout = self.episode_length_buf >= self.max_episode_length
        log = {f"Task/{name}": value.mean() for name, value in weighted.items()}
        log.update({"Swing/span_degrees": self.objective.peaks.sum(-1).mean() * (180 / math.pi),
                    "Swing/invalid_fraction": self.objective.invalid.float().mean(),
                    "Swing/max_cord_length": state["lengths"].max(),
                    "Swing/min_cord_length": state["lengths"].min(),
                    "Swing/lateral_abs": state["lateral"].abs().mean()})
        # Clone diagnostics before resets mutate episode-local state.
        metrics = {k: state[k].clone() for k in ("angle", "lengths", "lateral", "alignment")}
        metrics["invalid"] = self.objective.invalid.clone()
        self.observation_cache = None
        if self.auto_reset:
            ids = timeout.nonzero().flatten()
            if len(ids):
                self.total_resets += len(ids)
                self._reset(ids)
        obs, extras = self.get_observations()
        extras = {**extras, "time_outs": timeout, "log": log, "swing": metrics}
        if self.viewer:
            self.render()
        return obs, reward, timeout, extras

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.states[0])
        _, _, pose, linear, angular, _ = self._kinematics()
        state = self.mechanism(pose, linear, angular)
        offsets = self.viewer.world_offsets
        offsets = wp.to_torch(offsets) if isinstance(offsets, wp.array) else torch.as_tensor(offsets, device=self.device)
        starts = state["anchors"][None].expand(self.num_envs, -1, -1) + offsets[:, None]
        ends = state["attach"] + offsets[:, None]
        self.viewer.log_lines("swing/cords", wp.from_torch(starts.reshape(-1, 3).contiguous(), dtype=wp.vec3),
                              wp.from_torch(ends.reshape(-1, 3).contiguous(), dtype=wp.vec3), (0.92, 0.87, 0.70))
        self.viewer.end_frame()


def source_fingerprint():
    root = Path(__file__).parent
    files = [root / name for name in ("swing.py", "swing_scene.py", "swing_task.py", "env.py", "policy_spec.py",
                                      "scenes.py", "stability.py", "adapter.py", "ppo.py", "events.py")]
    files += sorted(ASSETS.glob("*.stl"))
    files += sorted((ASSETS.parent.parent / "microduck_rl-develop/src/mjlab_microduck/robot/microduck").rglob("*.xml"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def runner_config(config, count):
    result = ppo_config(count)
    for key, value in config["ppo"].items():
        if isinstance(value, dict):
            result[key].update(value)
        else:
            result[key] = value
    return result


def checkpoint_metadata(path, *, allow_source_mismatch=False):
    metadata = json.loads(Path(path).with_name("config.json").read_text())
    if metadata.get("task") != "swing" or metadata.get("observation_contract") != CONTRACT:
        raise ValueError("Expected a local Swing checkpoint; walking and upstream BAM checkpoints are incompatible")
    if metadata["source_fingerprint"] != source_fingerprint():
        if not allow_source_mismatch:
            raise ValueError("Swing source changed; diagnostic play can use --allow-source-mismatch")
        print("Diagnostic playback with changed source; this does not reproduce the saved training environment.")
    validate_config(metadata["swing"])
    return metadata


def run_synreal(args):
    count = args.num_envs
    metadata = (checkpoint_metadata(args.checkpoint, allow_source_mismatch=args.allow_source_mismatch)
                if args.checkpoint else None)
    config = metadata["swing"] if metadata else load_config(args.config or CONFIG)
    wp.set_device(args.device)
    if not wp.get_device().is_cuda:
        raise ValueError("SynReal requires a CUDA device")
    from .scenes import login

    login()
    viewer = None
    if args.viewer == "gl":
        from newton.viewer import ViewerGL

        viewer = ViewerGL(width=1280, height=900)
    env = None
    try:
        env = SwingEnv(config, count, args.device, viewer, args.seed, auto_reset=args.mode == "train")
        ppo = metadata["ppo"] if metadata else runner_config(config, count)
        log_dir = None
        if args.mode == "train":
            log_dir = new_log_dir(args)
            record = {**env.cfg, "solver": "synreal", "ppo": ppo, "source_fingerprint": source_fingerprint(), "num_envs": count,
                      "seed": args.seed, "resume_from": str(args.checkpoint) if args.checkpoint else None}
            (log_dir / "config.json").write_text(json.dumps(record, indent=2) + "\n")
        runner = None
        if args.mode != "preview" and not args.pretrained:
            runner = create_runner(env, ppo, str(log_dir) if log_dir else None, args.device, training=args.mode == "train")
            runner.git_status_repos = []
            if args.checkpoint:
                runner.load(str(args.checkpoint), load_optimizer=args.mode == "train")
                if args.mode == "train":
                    runner.current_learning_iteration += 1
                env.common_step_counter = (runner.current_learning_iteration * ppo["num_steps_per_env"])
        if args.mode == "train":
            print(f"Swing training: {log_dir.resolve()}", flush=True)
            runner.learn(args.iterations, init_at_random_ep_len=False)
        else:
            if args.pretrained:
                from .swing_policy import PretrainedSwingPolicy, pretrained_checkpoint

                policy = PretrainedSwingPolicy(pretrained_checkpoint(), args.device)
                print("Upstream alpha050 actor in SynReal PD dynamics; this is a cross-solver diagnostic.", flush=True)
            else:
                policy = (runner.get_inference_policy(device=args.device) if runner
                          else lambda obs: torch.zeros_like(env.actions))
            obs, _ = env.get_observations()
            report, down_before, frame = [], False, 0
            print("Swing: R resets all worlds; Space pauses. No automatic reset. Preview uses HOME PD only.", flush=True)
            with torch.inference_mode():
                while frame < args.num_frames and (viewer is None or viewer.is_running()):
                    start = time.monotonic()
                    if viewer:
                        down = viewer.is_key_down("r")
                        if down and not down_before:
                            obs, _ = env.reset()
                        down_before = down
                        if not viewer.should_step():
                            env.render()
                            continue
                    obs, _, _, extras = env.step(policy(obs))
                    if args.report:
                        report.append({"frame": frame, **{key: value.cpu().tolist() for key, value in extras["swing"].items()}})
                    if viewer:
                        time.sleep(max(0, env.step_dt - (time.monotonic() - start)))
                    frame += 1
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps({"frames": report, "resets": env.total_resets}, indent=2) + "\n")
            print(f"Completed {frame} frames; automatic resets: {env.total_resets}")
    finally:
        if env:
            env.close()
        elif viewer:
            viewer.close()


if __name__ == "__main__":
    run_synreal(args)
