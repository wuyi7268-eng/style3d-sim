"""Train, replay or preview blind basketball balance with SynReal or MuJoCo."""

import argparse
import hashlib
import json
import math
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import newton
import torch
import warp as wp

from examples.locomotion.math_utils import quat_rotate_inverse

from .adapter import rotate
from .basketball_scene import build_basketball_robot, hold_forces
from .basketball_task import advance_hold, task_terms
from .env import MicroDuckEnv, PhysicsConfig
from .policy_spec import ASSET
from .ppo import create_runner, ppo_config

CONFIG = Path(__file__).with_name("configs") / "basketball.json"
CONTRACT = "basketball_blind_lstm_pd_v1"


def load_config(path=CONFIG):
    config = json.loads(Path(path).read_text())
    validate_config(config)
    return config


def validate_config(config):
    if config["contract"] != CONTRACT:
        raise ValueError("Expected a blind LSTM basketball configuration")
    physics = PhysicsConfig(**config["physics"])
    if not math.isclose(physics.physics_dt * physics.decimation, 0.02):
        raise ValueError("Basketball control must run at 50 Hz")
    for value in (config["episode_s"], config["action_scale"], *[config["geometry"][k]
                  for k in ("radius", "mass", "root_height")]):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Episode, action scale and geometry must be positive and finite")
    hold = config["hold"]
    levels = hold["levels"]
    if (not levels or any(not math.isfinite(v) or not 0 <= v <= 1 for v in levels)
            or sorted(levels, reverse=True) != levels or levels[-1] != 0):
        raise ValueError("Hold levels must decrease from [0, 1] to a free ball (0)")
    if not 0 < hold["demote_s"] < hold["promote_s"] <= config["episode_s"]:
        raise ValueError("Hold curriculum requires 0 < demote < promote <= episode length")
    if any(not math.isfinite(hold[k]) or hold[k] < 0 for k in
           ("ball_kp", "ball_kd", "ball_kd_rot", "duck_kp", "duck_kd", "duck_kp_rot", "duck_kd_rot")):
        raise ValueError("Hold gains must be finite and nonnegative")
    friction = config["geometry"]["friction"]
    if len(friction) != 3 or any(not math.isfinite(v) or v < 0 for v in friction):
        raise ValueError("Expected three finite nonnegative friction coefficients")
    end = config["termination"]
    if (not 0 < end["tilt_deg"] < 90
            or any(not math.isfinite(end[k]) or end[k] <= 0 for k in ("min_height", "max_offset"))):
        raise ValueError("Invalid basketball failure thresholds")
    for group in (config["noise"], config["reset"]):
        if any(not math.isfinite(v) or v < 0 for v in group.values()):
            raise ValueError("Noise must be finite and nonnegative")
    commands = config["commands"]
    if (len(commands["limits"]) != 3 or any(not math.isfinite(v) or v < 0 for v in commands["limits"])
            or not 0 <= commands["standing_probability"] <= 1):
        raise ValueError("Invalid basketball command distribution")
    low, high = commands["resample_s"]
    if not 0 < low <= high or not math.isfinite(high):
        raise ValueError("Command resampling bounds must be finite, positive and ordered")
    for term in config["rewards"].values():
        if not math.isfinite(term["weight"]):
            raise ValueError("Reward weights must be finite")
        if any(not math.isfinite(v) or v <= 0 for k, v in term.items() if k != "weight"):
            raise ValueError("Reward scales must be positive and finite")
    policy = config["ppo"]["policy"]
    if (policy["class_name"] != "ActorCriticRecurrent" or policy["rnn_type"] != "lstm"
            or policy["rnn_hidden_dim"] != 256 or policy["rnn_num_layers"] != 1):
        raise ValueError("Basketball requires the one-layer LSTM256 policy contract")


class BasketballEnv(MicroDuckEnv):
    num_obs, num_privileged_obs = 61, 64

    def __init__(self, config=None, num_envs=8, device="cuda:0", viewer=None, seed=42, *, solver="synreal",
                 auto_reset=True):
        self.config = deepcopy(config or load_config())
        validate_config(self.config)
        if solver == "mujoco" and torch.device(device).type == "cpu" and num_envs != 1:
            raise ValueError("MuJoCo CPU supports one world; use --num-envs 1 or --device cuda:0")
        self.auto_reset, self.observation_cache = auto_reset, None
        # The released actor was trained with the encoder velocity one control
        # tick late. Keep the raw velocity from the start of the previous tick
        # so the recurrent input follows that firmware observation contract.
        self.delayed_velocity = None
        super().__init__(solver=solver, num_envs=num_envs, device=device, viewer=viewer, seed=seed,
                         randomize_foot_friction=False, reset_on_fall=False,
                         physics=PhysicsConfig(**self.config["physics"]))
        self.task = self.family = "basketball"
        self.common_step_counter = 0
        self.max_episode_length = math.ceil(self.config["episode_s"] / self.step_dt)
        self.policy_actions = torch.zeros_like(self.actions)
        self.level = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.hold = torch.zeros(num_envs, device=self.device)
        self.command_time = torch.zeros(num_envs, device=self.device)
        self.failed_once = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        lower = wp.to_torch(self.model.joint_limit_lower).reshape(num_envs, -1)[:, self.qdi]
        upper = wp.to_torch(self.model.joint_limit_upper).reshape(num_envs, -1)[:, self.qdi]
        midpoint, half = (lower + upper) / 2, (upper - lower) * 0.475
        self.soft_lower, self.soft_upper = midpoint - half, midpoint + half
        self.cfg.update(task="basketball", observation_contract=CONTRACT, basketball=self.config,
                        deployment_compatible=False, policy_variant="lstm",
                        adaptation={"actuator": "shared saturated PD, not upstream BAM",
                                    "actor": "61D blind; one-step joint velocity delay; no IMU bias/delay randomization",
                                    "critic": "64D; ball state included, upstream force channels omitted",
                                    "rewards": "basketball terms plus local regularizers; not full upstream walking stack",
                                    "randomization": "spawn/noise/commands; no upstream motor or push randomization",
                                    "hold": "training-only ball/trunk assistance, updated every physics substep",
                                    "contacts": "shared Newton geometry; solver-specific friction/contact laws"})
        self.cfg.pop("task_recipe", None)
        self.cfg.pop("recipe", None)
        self.reset()
        if viewer:
            viewer.set_world_offsets((2.5, 2.5, 0))
            viewer.set_camera(pos=(1.0, -1.4, 0.85), pitch=-17, yaw=130)

    def build_robot(self, builder_type, **kwargs):
        robot, qi, qdi, self.ball_body, self.ball_q = build_basketball_robot(builder_type, self.config)
        self.foot_sites = []
        for name in ("left_foot", "right_foot"):
            shape = next(i for i, label in enumerate(robot.shape_label) if label.endswith("/" + name))
            self.foot_sites.append((robot.shape_body[shape],
                                    torch.tensor(list(robot.shape_transform[shape])[:3], device=self.device)))
        return robot, qi, qdi

    def create_recipe(self, robot):
        return None

    def apply_task_forces(self):
        if not self.auto_reset:
            return
        hold = self.config["hold"]
        wp.launch(hold_forces, dim=self.num_envs, inputs=[
            self.states[0].body_q, self.states[0].body_qd, self.model.body_com, self.states[0].body_f,
            wp.from_torch(self.hold), self.model.body_count // self.num_envs, self.root, self.ball_body,
            *[hold[k] for k in ("ball_kp", "ball_kd", "ball_kd_rot", "duck_kp", "duck_kd", "duck_kp_rot", "duck_kd_rot")],
        ], device=self.model.device)

    def sample_commands(self, ids):
        config = self.config["commands"]
        self.commands[ids] = 0
        if self.auto_reset:
            limits = torch.tensor(config["limits"], device=self.device)
            self.commands[ids, :3] = (torch.rand(len(ids), 3, device=self.device) * 2 - 1) * limits
            standing = torch.rand(len(ids), device=self.device) < config["standing_probability"]
            self.commands[ids[standing], :3] = 0
        self.command_time[ids] = torch.empty(len(ids), device=self.device).uniform_(*config["resample_s"])

    def set_command(self, command):
        value = torch.as_tensor(command, device=self.device, dtype=torch.float32)
        if value.shape != (3,) or not torch.isfinite(value).all():
            raise ValueError("Expected finite forward/lateral/yaw command")
        self.commands[:, :3] = value
        self.observation_cache = None

    def _reset(self, ids):
        self._sync()
        spawn = self.initial_q[ids].clone()
        if self.auto_reset:
            noise = self.config["reset"]
            spawn[:, :2] += torch.empty(len(ids), 2, device=self.device).uniform_(-noise["xy_noise"], noise["xy_noise"])
            yaw = torch.empty(len(ids), device=self.device).uniform_(-math.pi, math.pi)
            roll, pitch = torch.empty(2, len(ids), device=self.device).uniform_(
                -math.radians(noise["tilt_deg"]), math.radians(noise["tilt_deg"]))
            cr, sr = torch.cos(roll / 2), torch.sin(roll / 2)
            cp, sp = torch.cos(pitch / 2), torch.sin(pitch / 2)
            cy, sy = torch.cos(yaw / 2), torch.sin(yaw / 2)
            spawn[:, 3:7] = torch.stack((cy * cp * sr - sy * sp * cr, cy * sp * cr + sy * cp * sr,
                                         sy * cp * cr - cy * sp * sr, cy * cp * cr + sy * sp * sr), dim=-1)
            spawn[:, self.qi] += torch.empty(len(ids), 14, device=self.device).uniform_(
                -noise["joint_noise"], noise["joint_noise"])
        for state in self.states:
            q, qd, _, _ = self._views(state)
            q[ids], qd[ids] = spawn, 0
            self.sync_selected_state(state, ids)
        targets = wp.to_torch(self.control.joint_target_q).reshape(self.num_envs, -1)
        columns = self.qi if newton.use_coord_layout_targets else self.qdi
        targets[ids[:, None], columns] = self.home
        self.actions[ids], self.policy_actions[ids] = 0, 0
        if self.delayed_velocity is None:
            self.delayed_velocity = torch.zeros((self.num_envs, len(self.qdi)), device=self.device)
        self.delayed_velocity[ids] = 0
        self.episode_length_buf[ids], self.failed_once[ids] = 0, False
        table = torch.tensor(self.config["hold"]["levels"], device=self.device)
        self.hold[ids] = table[self.level[ids]] if self.auto_reset else 0
        self.sample_commands(ids)
        self.observation_cache = None
        self._sync()

    def ball_state(self):
        poses = wp.to_torch(self.states[0].body_q).reshape(self.num_envs, -1, 7)
        velocities = wp.to_torch(self.states[0].body_qd).reshape(self.num_envs, -1, 6)
        feet = torch.stack([poses[:, body, :3] + rotate(poses[:, body, 3:], local.expand(self.num_envs, -1))
                            for body, local in self.foot_sites], dim=1)
        return poses[:, self.ball_body, :3], velocities[:, self.ball_body], feet

    def get_observations(self):
        if self.observation_cache is None:
            q, qd, pose, linear, angular, gravity = self._kinematics()
            position, velocity = self.encoder_state(q, qd)
            if self.delayed_velocity is None:
                self.delayed_velocity = torch.zeros_like(velocity)
            observed_velocity = self.delayed_velocity
            ball, ball_velocity, _ = self.ball_state()
            # Match the release layout: gyro, gravity, joint position, joint velocity.
            # A duplicated previous-velocity block would keep width 61 while shifting
            # actions and commands into the wrong network inputs.
            parts = [angular, gravity, position - self.home, observed_velocity]
            noisy = []
            for name, value in zip(("angular", "gravity", "position", "velocity"), parts, strict=True):
                bound = self.config["noise"][name] if self.auto_reset else 0
                noisy.append(value + torch.empty_like(value).uniform_(-bound, bound))
            actor = torch.cat((*noisy, self.policy_actions, self.commands[:, :3],
                               torch.zeros(self.num_envs, 10, device=self.device)), dim=-1)
            relative = quat_rotate_inverse(pose[:, 3:], ball - pose[:, :3]) * 3
            ball_linear = quat_rotate_inverse(pose[:, 3:], ball_velocity[:, :3])[:, :2]
            ball_spin = quat_rotate_inverse(pose[:, 3:], ball_velocity[:, 3:])[:, 1:2] * 0.2
            critic = torch.cat((linear, angular, gravity, position - self.home, observed_velocity, self.policy_actions,
                                self.commands[:, :3], torch.zeros(self.num_envs, 4, device=self.device),
                                relative, ball_linear, ball_spin), dim=-1)
            if not torch.isfinite(actor).all() or not torch.isfinite(critic).all():
                raise FloatingPointError("Non-finite basketball observations")
            self.observation_cache = actor, {"observations": {"critic": critic}}
        return self.observation_cache

    @torch.no_grad()
    def step(self, actions):
        if actions.shape != self.actions.shape or not torch.isfinite(actions).all():
            raise ValueError("Basketball expects finite [num_envs, 14] actions")
        previous = self.policy_actions.clone()
        q_before, qd_before, *_ = self._kinematics()
        _, velocity_before = self.encoder_state(q_before, qd_before)
        self.policy_actions.copy_(actions.to(self.device).clamp(-1, 1))
        torque = self.advance_physics(self.policy_actions * self.config["action_scale"],
                                      apply_viewer_forces=not self.auto_reset)
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.sim_time += self.step_dt
        q, _, pose, linear, angular, gravity = self._kinematics()
        self.delayed_velocity = velocity_before.detach().clone()
        ball, ball_velocity, feet = self.ball_state()
        terms, failed, metrics = task_terms(self.config, pose[:, :3], gravity, linear, angular, ball,
                                           ball_velocity[:, :3], feet, self.commands)
        # State-buffer views would otherwise report the respawn instead of the failed transition.
        metrics = {key: value.clone() for key, value in metrics.items()}
        self.failed_once |= failed
        metrics.update(failed_once=self.failed_once.clone(), hold=self.hold.clone(), command=self.commands[:, :3].clone())
        position = q[:, self.qi]
        terms.update(action_rate=(self.policy_actions - previous).square().sum(-1),
                     joint_torques=torque.square().sum(-1),
                     joint_limits=((self.soft_lower - position).clamp_min(0)
                                   + (position - self.soft_upper).clamp_min(0)).sum(-1))
        weighted = {k: value * self.config["rewards"][k]["weight"] for k, value in terms.items()}
        reward = torch.stack(list(weighted.values())).sum(0) * self.step_dt
        if not torch.isfinite(reward).all():
            raise FloatingPointError("Non-finite basketball reward")
        timeout = self.episode_length_buf >= self.max_episode_length
        done = failed | timeout
        log = {f"Task/{k}": value.mean() for k, value in weighted.items()}
        log.update({"Basketball/offset": metrics["offset"].mean(), "Basketball/fall_fraction": failed.float().mean(),
                    "Basketball/hold": self.hold.mean(), "Basketball/free_fraction": (self.hold == 0).float().mean()})
        self.observation_cache = None
        if self.auto_reset:
            self.command_time -= self.step_dt
            self.sample_commands((self.command_time <= 0).nonzero().flatten())
            ids = done.nonzero().flatten()
            if len(ids):
                self.level[ids] = advance_hold(self.level[ids], self.episode_length_buf[ids] * self.step_dt,
                                               self.config["hold"])
                self.total_resets += len(ids)
                self._reset(ids)
        obs, extras = self.get_observations()
        # Playback exposes failures as metrics without starting a new episode or clearing memory.
        extras = {**extras, "time_outs": timeout & ~failed, "log": log, "basketball": metrics}
        if self.viewer:
            self.render()
        return obs, reward, done if self.auto_reset else torch.zeros_like(done), extras


def source_fingerprint():
    root = Path(__file__).parent
    paths = [root / name for name in ("basketball.py", "basketball_scene.py", "basketball_task.py", "env.py",
                                      "policy_spec.py", "adapter.py", "ppo.py", "scenes.py", "stability.py", "events.py")]
    paths += sorted(ASSET.parent.rglob("*.xml"))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def runner_config(config, count):
    result = ppo_config(count)
    for key, value in config["ppo"].items():
        if isinstance(value, dict):
            result[key].update(value)
        else:
            result[key] = value
    result["policy"].pop("noise_std_type", None)
    return result


def attach_checkpoint_state(runner, env):
    save = runner.save

    def save_task(path, infos=None):
        save(path, {**(infos or {}), "basketball": {"common_step_counter": env.common_step_counter,
                                                   "hold_level": env.level.detach().cpu()}})

    runner.save = save_task


def checkpoint_metadata(path, *, allow_source_mismatch=False):
    metadata = json.loads(Path(path).with_name("config.json").read_text())
    if metadata.get("task") != "basketball" or metadata.get("observation_contract") != CONTRACT:
        raise ValueError("Expected a local basketball checkpoint and its config.json")
    if metadata["source_fingerprint"] != source_fingerprint() and not allow_source_mismatch:
        raise ValueError("Source changed; diagnostic play can use --allow-source-mismatch")
    validate_config(metadata["basketball"])
    return metadata


def keyboard_command(viewer, command, previous_keys, limits):
    """Change commands without resetting recurrent memory, bounded to the training ranges."""
    command = list(command)
    increments = {"i": (0, 0.05), "k": (0, -0.05), "j": (1, 0.05), "l": (1, -0.05),
                  "u": (2, 0.1), "o": (2, -0.1)}
    for key, (axis, amount) in increments.items():
        down = bool(viewer.is_key_down(key))
        if down and not previous_keys.get(key, False):
            command[axis] = max(-limits[axis], min(limits[axis], command[axis] + amount))
        previous_keys[key] = down
    if viewer.is_key_down("x"):
        command = [0.0, 0.0, 0.0]
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "play", "preview"))
    parser.add_argument("--solver", choices=("synreal", "mujoco"), default="synreal")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--policy-file", type=Path, help="Locally downloaded release policy.onnx (with --pretrained)")
    parser.add_argument("--allow-source-mismatch", action="store_true")
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--iterations", type=int, default=1000, help="Additional PPO iterations")
    parser.add_argument("--num-frames", type=int, default=1500)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--viewer", choices=("gl", "null"))
    parser.add_argument("--command", type=float, nargs=3, default=(0, 0, 0), metavar=("VX", "VY", "YAW"))
    parser.add_argument("--log-root", type=Path, default=Path("logs/rsl_rl/microduck_basketball"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    count = args.num_envs if args.num_envs is not None else (8 if args.mode == "train" else 1)
    if min(count, args.iterations, args.num_frames) < 1 or not all(math.isfinite(v) for v in args.command):
        parser.error("Counts must be positive and commands finite")
    if args.mode == "play" and not (args.checkpoint or args.pretrained):
        parser.error("play requires --checkpoint or --pretrained")
    if args.pretrained and (args.mode != "play" or args.checkpoint):
        parser.error("--pretrained is for play and cannot be combined with --checkpoint")
    if args.policy_file and not args.pretrained:
        parser.error("--policy-file requires --pretrained")
    if args.mode == "preview" and args.checkpoint:
        parser.error("preview uses HOME PD; use play for checkpoints")
    if args.allow_source_mismatch and args.mode != "play":
        parser.error("Source mismatch is allowed only for diagnostic play")
    metadata = (checkpoint_metadata(args.checkpoint, allow_source_mismatch=args.allow_source_mismatch)
                if args.checkpoint else None)
    if metadata and args.mode == "train" and args.num_envs is None:
        count = metadata["num_envs"]
    if metadata and args.mode == "train" and (metadata["solver"] != args.solver or metadata["num_envs"] != count):
        parser.error("Strict resume requires the saved solver and num-envs (per-world curriculum)")
    config = metadata["basketball"] if metadata else load_config(args.config)
    # Fetch before allocating physics resources, so an unavailable artifact fails cheaply.
    release = None
    if args.pretrained:
        from .basketball_policy import pretrained_policy

        release = pretrained_policy(args.policy_file)
    wp.set_device(args.device)
    if args.solver == "synreal":
        if not wp.get_device().is_cuda:
            parser.error("SynReal requires a CUDA device")
        from .scenes import login

        login()
    viewer = env = None
    try:
        if (args.viewer or ("null" if args.mode == "train" else "gl")) == "gl":
            from newton.viewer import ViewerGL

            viewer = ViewerGL(width=1280, height=900)
        env = BasketballEnv(config, count, args.device, viewer, args.seed, solver=args.solver, auto_reset=args.mode == "train")
        ppo = metadata["ppo"] if metadata else runner_config(config, count)
        directory = None
        if args.mode == "train":
            stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d_%H-%M-%S_%f")
            directory = args.log_root / (stamp + "_" + args.solver)
            directory.mkdir(parents=True)
            record = {**env.cfg, "ppo": ppo, "source_fingerprint": source_fingerprint(), "num_envs": count,
                      "seed": args.seed, "resume_from": str(args.checkpoint) if args.checkpoint else None}
            (directory / "config.json").write_text(json.dumps(record, indent=2) + "\n")
        runner = None
        if args.mode != "preview" and not args.pretrained:
            runner = create_runner(env, ppo, str(directory) if directory else None, args.device, training=args.mode == "train")
            runner.git_status_repos = []
            attach_checkpoint_state(runner, env)
            if args.checkpoint:
                infos = runner.load(str(args.checkpoint), load_optimizer=args.mode == "train")
                if args.mode == "train":
                    saved = infos["basketball"]
                    env.level.copy_(saved["hold_level"].to(env.device))
                    env.common_step_counter = saved["common_step_counter"]
                    runner.current_learning_iteration += 1
                    env.reset()
        if args.mode == "train":
            print(f"Basketball {args.solver} LSTM training: {directory.resolve()}", flush=True)
            runner.learn(args.iterations, init_at_random_ep_len=False)
        else:
            if release:
                from .basketball_policy import BasketballOnnxPolicy

                policy = BasketballOnnxPolicy(release, count, args.device)
                reset_policy = policy.reset
                print("Released b11 actor in local PD dynamics; cross-simulator diagnostic, not BAM reproduction.", flush=True)
            elif runner:
                policy = runner.get_inference_policy(device=args.device)
                reset_policy = lambda: runner.alg.policy.reset(torch.ones(count, dtype=torch.bool, device=env.device))
            else:
                policy = lambda obs: torch.zeros_like(env.actions)
                reset_policy = lambda: None
            command, previous_keys = list(args.command), {}
            env.set_command(command)
            obs, _ = env.get_observations()
            frames, down_before, frame, manual_resets = [], False, 0, 0
            print("Basketball: free ball, no assistance; R resets robot/ball/LSTM, Space pauses. "
                  "I/K forward, J/L lateral, U/O yaw, X stop. No automatic reset.", flush=True)
            with torch.inference_mode():
                while frame < args.num_frames and (viewer is None or viewer.is_running()):
                    start = time.monotonic()
                    if viewer:
                        down = viewer.is_key_down("r")
                        if down and not down_before:
                            env.reset()
                            env.set_command(command)
                            reset_policy()
                            obs, _ = env.get_observations()
                            manual_resets += 1
                        down_before = down
                        changed = keyboard_command(viewer, command, previous_keys, config["commands"]["limits"])
                        if changed != command:
                            command = changed
                            env.set_command(command)
                            obs, _ = env.get_observations()
                        if not viewer.should_step():
                            env.render()
                            continue
                    obs, reward, _, extras = env.step(policy(obs))
                    if args.report:
                        frames.append({"frame": frame, "reward": reward.cpu().tolist(),
                                       **{k: v.cpu().tolist() for k, v in extras["basketball"].items()}})
                    frame += 1
                    if viewer:
                        time.sleep(max(0, env.step_dt - (time.monotonic() - start)))
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps({"solver": args.solver, "frames": frames, "manual_resets": manual_resets,
                                                   "automatic_resets": env.total_resets,
                                                   "adaptation": env.cfg["adaptation"]}, indent=2) + "\n")
            print(f"Completed {frame} frames; automatic resets: {env.total_resets}")
    finally:
        if env:
            env.close()
        elif viewer:
            viewer.close()


if __name__ == "__main__":
    main()
