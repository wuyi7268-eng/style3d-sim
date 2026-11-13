"""MicroDuck play entry point for MuJoCo and SynReal."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import time

import torch
import warp as wp

from examples.microduck.adapter import rotate
from examples.microduck.common import create_env, setup
from examples.microduck.policy_spec import HOME

# Commands are offsets from HOME. This fixed command requests the neutral,
# raised-head pose while leaving the policy's head actions unconstrained.
HEAD_UP_COMMAND = (-HOME[5], -HOME[6], 0.0, 0.0)


def manual_reset_requested(viewer, keys):
    """Consume a rising edge so holding R never repeatedly resets simulation state."""
    down = bool(viewer is not None and viewer.is_key_down("r"))
    requested = down and not keys.get("reset", False)
    keys["reset"] = down
    return requested


def configure_playback_boundaries(env):
    if env.mixed_terrain:
        # Crossing the pad edge is observable in playback; training keeps its surface-only curriculum.
        env.cfg["terrain"]["pad_edge_timeout_margin"] = None
        env.cfg["terrain"]["boundary_curriculum"] = "playback_cross_surface_locomotion"
    elif getattr(env.terrain, "kind", None) == "elastic_bridge":
        env.terrain.playback_center_spawn = True


def wait_for_viewer_close(viewer, keep_open):
    if viewer is None or not keep_open:
        return
    print("Playback complete; close the viewer window to exit.", flush=True)
    # Keep the event loop alive so a finite replay does not look like a viewer crash.
    while viewer.is_running():
        time.sleep(0.1)


def update_velocity_command(env, viewer, command, key_was_down=None, previous_command=None):
    key_was_down = {} if key_was_down is None else key_was_down
    if viewer is not None and hasattr(viewer, "is_key_down"):
        key = viewer.is_key_down
        increments = {"i": (0, 0.1), "k": (0, -0.1), "j": (1, 0.1), "l": (1, -0.1), "u": (2, 0.1), "o": (2, -0.1)}
        for name, (axis, amount) in increments.items():
            down = bool(key(name))
            if down and not key_was_down.get(name, False):
                command[axis] += amount
            key_was_down[name] = down
    # Resets and headless command sampling can replace the command between frames.
    env.commands[:, :3] = torch.tensor(command, device=env.device)
    if command != previous_command:
        print(f"Velocity command: vx={command[0]:.2f}, vy={command[1]:.2f} m/s, yaw={command[2]:.2f} rad/s", flush=True)
    return list(command)


def update_playback_observation(env, obs):
    terms = env.official.observations.spec["actor"]["terms"]
    if getattr(env, "task", None) == "running":
        # Match the released running controller's neutral, unused pose commands.
        env.commands[:, 3:] = 0.0
    elif not terms["head_command"]["func"].endswith(".zero_command_padding"):
        env.commands[:, 3:7] = torch.tensor(HEAD_UP_COMMAND, device=env.device, dtype=obs.dtype)
    # Unused command slots must remain zero even when a cached observation is refreshed.
    # Their saved normalizer variance is zero; injecting a pose here shifts policy inputs dramatically.
    for name, start, end in (("command", 0, 3), ("head_command", 3, 7), ("body_command", 7, 13)):
        target = obs[:, 48 + start : 48 + end]
        if terms[name]["func"].endswith(".zero_command_padding"):
            target.zero_()
        else:
            target.copy_(env.commands[:, start:end])


class PolicyPlayback:
    def __init__(self, env):
        self.env = env
        self.observations = env.official.observations
        allowed = {
            "base_ang_vel",
            "projected_gravity",
            "joint_pos",
            "joint_vel",
            "actions",
            "command",
            "head_command",
            "body_command",
        }
        if set(self.observations.spec["actor"]["terms"]) - allowed:
            raise ValueError("Fast playback requires the current proprioceptive actor contract")
        # A skipped critic must not consume RNG or carry delays that influence actor sampling.
        critic = self.observations.spec["critic"]
        if critic["enable_corruption"] or any(group == "critic" for group, _ in self.observations.delays):
            raise ValueError("Fast playback requires a clean, undelayed critic")

    def observe(self):
        env = self.env
        processor = self.observations
        if processor.cache is not None:
            processor.cache[0][:, 48:] = env.commands
            return processor.with_history(processor.cache[0])[0]
        q, qd, _, _, angular, gravity = env._kinematics()
        if hasattr(env, "encoder_state"):
            position, joint_velocity = env.encoder_state(q, qd)
        else:
            position, joint_velocity = q[:, env.qi], qd[:, env.qdi]
        values = {
            "base_ang_vel": angular,
            "projected_gravity": gravity,
            "joint_pos": position - env.home,
            "joint_vel": joint_velocity,
            "actions": env.actions,
            "command": env.commands[:, :3],
            "head_command": env.commands[:, 3:7],
            "body_command": env.commands[:, 7:13],
        }
        cfg = processor.spec["actor"]
        parts = []
        for name, term in cfg["terms"].items():
            value = values[name].clone()
            func = term["func"].rsplit(".", 1)[-1]
            if func == "zero_command_padding":
                value.zero_()
            if func.endswith("imu_misaligned"):
                value = rotate(processor.imu_rotation, value)
            if name == "joint_pos" and term["params"].get("biased", False):
                value += processor.encoder_bias
            noise = term["noise"] if cfg["enable_corruption"] else None
            if noise:
                if noise["operation"] != "add":
                    raise NotImplementedError("Non-additive observation noise")
                value += torch.empty_like(value).uniform_(noise["n_min"], noise["n_max"])
            if term["clip"] is not None:
                value.clamp_(*term["clip"])
            if term["scale"] is not None:
                value *= torch.as_tensor(term["scale"], device=env.device)
            if ("actor", name) in processor.delays:
                value = processor.delays["actor", name](value)
            parts.append(value)
        actor = torch.cat(parts, -1)
        if actor.shape != (env.num_envs, 61) or not torch.isfinite(actor).all():
            raise FloatingPointError("Invalid playback actor observations")
        processor.cache = actor, None
        return processor.with_history(actor)[0]

    def step(self, actions):
        return self.env.step_playback(actions)


def run_walk(args):
    if args.terrain == "mixed_soft_pad":
        return run_walk_on_pad(args)
    from newton.viewer import ViewerGL, ViewerNull

    from examples.example_robot_microduck import Example

    wp.set_device(args.device)
    viewer = ViewerGL() if args.viewer == "gl" else ViewerNull(num_frames=args.num_frames)
    # Use the original constructor and step verbatim: no PD, timestep or solver changes for these weights.
    original_args = Example.create_parser().parse_args([])
    for name in ("solver", "device", "force_login", "disable_cuda_graph", "command"):
        setattr(original_args, name, getattr(args, name))
    original_args.incremental_command = True
    frames = 0
    warmup = min(20, args.num_frames // 4)
    try:
        example = Example(viewer, original_args)
        print(f"Official walk.onnx: solver={args.solver}, graph={example.graph is not None}, physics=original example")
        print("Press I/K: forward/back; J/L: lateral; U/O: turn; R: reset all environments (P also supported).")
        keys = {}
        previous_command = None
        measured_start = time.monotonic()
        while frames < args.num_frames:
            if not viewer.is_running():
                break
            if manual_reset_requested(viewer, keys):
                example.reset()
            if viewer.is_paused():
                example.render()
                continue
            if frames == warmup:
                measured_start = time.monotonic()
            example.step()
            # The policy observation contains the exact command used for this step, including viewer keys.
            command = tuple(float(x) for x in example.policy.observation[0, 48:51])
            if command != previous_command:
                print(f"Velocity command: vx={command[0]:.2f}, vy={command[1]:.2f} m/s, yaw={command[2]:.2f} rad/s")
                previous_command = command
            example.render()
            frames += 1
        elapsed = time.monotonic() - measured_start
        measured_frames = max(0, frames - warmup)
        if measured_frames:
            fps = measured_frames / elapsed
            print(f"Playback: {fps:.1f} FPS, simulation/wall={fps * example.frame_dt:.2f}x")
        print(f"Official policy playback complete: frames={frames}")
        wait_for_viewer_close(viewer, args.keep_open)
    finally:
        viewer.close()
    return 0


def run_walk_on_pad(args):
    from examples.example_robot_microduck import WalkPolicy
    from examples.microduck_assets import ASSET_DIR

    policy = WalkPolicy(ASSET_DIR / "policies/walk.onnx")
    env = create_env(args)
    try:
        if policy.joint_names != env.joint_names[:14]:
            raise ValueError("Official walk policy and soft-pad servo order differ")
        configure_playback_boundaries(env)
        viewer = env.viewer
        if viewer is not None:
            viewer.set_camera(pos=wp.vec3(0.8, -1.0, 0.72), pitch=-26.0, yaw=129.0)
            viewer.renderer.set_title("MicroDuck | Official walk.onnx | Soft pad")
        print("Official walk.onnx on current soft-pad physics (50 Hz, shared position PD).", flush=True)
        print("I/K: forward/back; J/L: lateral; U/O: turn; X: stop; R: reset all environments. Space: pause.", flush=True)
        command, keys, previous = list(args.command), {}, None
        frames = 0
        with torch.inference_mode():
            while frames < args.num_frames:
                if viewer is not None and not viewer.is_running():
                    break
                if manual_reset_requested(viewer, keys):
                    env.reset()
                    policy.reset()
                if viewer is not None and viewer.is_paused():
                    env.render()
                    time.sleep(0.02)
                    continue
                start = time.monotonic()
                if viewer is not None and viewer.is_key_down("x"):
                    command[:] = [0.0, 0.0, 0.0]
                previous = update_velocity_command(env, viewer, command, keys, previous)
                q, qd, pose, velocity = env._views()
                # Use the published inference contract and embedded normalization,
                # without adding the local training recipe's sensor bias or delays.
                target = policy.infer(
                    pose[0, 3:7].cpu().numpy(), velocity[0, 3:6].cpu().numpy(),
                    q[0, env.qi].cpu().numpy(), qd[0, env.qdi].cpu().numpy(), command,
                )
                actions = torch.as_tensor(target, device=env.device).unsqueeze(0) - env.home
                env.step_playback(actions)
                frames += 1
                if viewer is not None and not args.benchmark:
                    time.sleep(max(0.0, env.step_dt - (time.monotonic() - start)))
        wait_for_viewer_close(viewer, args.keep_open)
    finally:
        env.close()
    return 0


def main(argv=None):
    args, _training_config, play_cfg = setup("play", argv)
    heading_hold = getattr(args, "heading_hold", False)
    adaptive_forward = getattr(args, "adaptive_forward", False)
    if adaptive_forward and not heading_hold:
        raise ValueError("--adaptive-forward requires --heading-hold")
    if heading_hold and (args.policy == "walk" or args.num_envs != 1 or args.task != "velocity"
                         or args.command[0] <= 0 or any(args.command[1:])):
        raise ValueError("--heading-hold requires one Velocity checkpoint with positive forward and zero lateral/yaw command")
    if args.record_dir and (args.policy == "walk" or args.num_envs != 1
                            or args.terrain not in ("mixed_soft_pad", "cloth", "elastic_bridge")):
        raise ValueError("--record-dir requires a single-environment soft-pad or cloth checkpoint replay")
    if args.record_until_pad_edge and not args.record_dir:
        raise ValueError("--record-until-pad-edge requires --record-dir")
    if args.policy == "walk":
        return run_walk(args)
    env = create_env(args)
    configure_playback_boundaries(env)
    if getattr(args, "manual_command", False):
        # Preserve the platform spawn configured above, but also disable feedback inside env.step/reset.
        env.official.bridge_path = None
        print("Manual velocity commands enabled; bridge path feedback disabled.", flush=True)
    if heading_hold and env.official.bridge_path is not None:
        env.close()
        raise ValueError("Bridge path feedback already controls heading; omit --heading-hold")
    # Keep the trained observation layout, delays, and fixed calibration offsets,
    # while removing per-tick random actor noise for deterministic visual replay.
    env.official.observations.spec["actor"]["enable_corruption"] = False
    print("Playback actor observation noise: disabled", flush=True)
    print("Playback training interval events: disabled (including random velocity pushes)", flush=True)
    viewer = env.viewer
    recorder = None
    try:
        from examples.microduck.ppo import create_runner

        runner = create_runner(env, play_cfg, log_dir=None, device=args.device, training=False)
        runner.load(str(args.checkpoint), load_optimizer=False)
        if env.official is not None:
            env.official.common_step_counter = (runner.current_learning_iteration + 1) * play_cfg["num_steps_per_env"]
            # Playback starts from the checkpoint's recovery-pose distribution.
            env.reset()
        # RSL-RL's inference wrapper restores and applies the saved observation normalizer.
        policy = runner.get_inference_policy(device=args.device)
        print("Automatic resets disabled. R: reset all environments, including while paused.", flush=True)
        if env.official.bridge_path is not None:
            print("Bridge centerline feedback enabled; stop on the outlet platform. Pose feedback supplies policy commands.",
                  flush=True)
        elif viewer is not None and env.family in ("velocity", "velstand"):
            print(
                "Press I/K: forward/back; J/L: lateral; U/O: turn. Each press adds one increment; release keeps it.", flush=True
            )
        playback = PolicyPlayback(env)
        previous_command = None
        key_was_down = {}
        command = list(args.command)
        frames = 0
        reached_edge = False
        warmup = min(20, args.num_frames // 4)
        measured_start = time.monotonic()
        obs, _ = env.get_observations()
        steering = None
        if heading_hold:
            from examples.microduck.heading_control import HeadingHold, ProgressHold

            steering_class = ProgressHold if adaptive_forward else HeadingHold
            steering_options = ({"nominal_forward": command[0], "target_speed": args.target_forward_speed}
                                if adaptive_forward else {})
            steering = steering_class(env._views()[2][0].cpu().tolist(), args.heading_max_yaw, **steering_options)
            print(f"Pose feedback enabled: adaptive_forward={adaptive_forward}; policy commands only.", flush=True)
        if args.record_dir:
            from demos.microduck_soft_pad.recording import PlaybackRecording

            recorder = PlaybackRecording(env, args)
        with torch.inference_mode():
            while frames < args.num_frames:
                if viewer is not None and not viewer.is_running():
                    break
                if manual_reset_requested(viewer, key_was_down):
                    env.reset()
                    obs, _ = env.get_observations()
                    if steering is not None:
                        steering = steering_class(env._views()[2][0].cpu().tolist(), args.heading_max_yaw, **steering_options)
                if viewer is not None and viewer.is_paused():
                    env.render()
                    time.sleep(0.02)
                    continue
                start = time.monotonic()
                if frames == warmup:
                    measured_start = start
                if env.official.bridge_path is not None:
                    env.official.bridge_path.update()
                elif env.family in ("velocity", "velstand"):
                    previous_command = update_velocity_command(env, viewer, command, key_was_down, previous_command)
                if steering is not None:
                    pose = env._views()[2][0].cpu().tolist()
                    if adaptive_forward:
                        env.commands[0, 0], env.commands[0, 2] = steering.update_commands(pose, env.step_dt)
                    else:
                        env.commands[0, 2] = steering.update(pose, command[0], env.step_dt)
                update_playback_observation(env, obs)
                actions = policy(obs)
                if args.diagnostics and frames == 0:
                    zero_command_obs = obs.clone()
                    zero_command_obs[:, 48:51] = 0.0
                    zero_actions = policy(zero_command_obs)
                    action_delta = (actions[0] - zero_actions[0]).norm().item()
                    print(
                        "Command diagnostic: "
                        f"actor[48:51]={obs[0, 48:51].tolist()}, "
                        f"policy_action_delta_from_zero={action_delta:.6f}",
                        flush=True,
                    )
                obs = playback.step(actions)
                if recorder is not None:
                    recorder.sample()
                frames += 1
                if recorder is not None and args.record_until_pad_edge and recorder.reached_pad_edge():
                    reached_edge = True
                    print(f"Recording stopped at the soft-pad edge after {frames} frames", flush=True)
                    break
                if args.diagnostics and (frames == 1 or frames % 100 == 0):
                    _, _, _, linear, angular, _ = env._kinematics()
                    print(
                        "Replay state: "
                        f"frame={frames}, command={env.commands[0, :3].tolist()}, "
                        f"base_linear_velocity={linear[0, :3].tolist()}, "
                        f"base_angular_velocity={angular[0, :3].tolist()}, resets={env.total_resets}",
                        flush=True,
                    )
                if viewer is not None and not args.benchmark:
                    time.sleep(max(0.0, env.cfg["policy_dt"] - (time.monotonic() - start)))
        elapsed = time.monotonic() - measured_start
        measured_frames = max(0, frames - warmup)
        if measured_frames:
            fps = measured_frames / elapsed
            print(f"Playback: {fps:.1f} FPS, simulation/wall={fps * env.cfg['policy_dt']:.2f}x, graph={env.graph is not None}")
        print(
            f"Policy playback complete: frames={frames}, resets={env.total_resets}, checkpoint={args.checkpoint}",
            flush=True,
        )
        if recorder is not None:
            recorder.close(completed=frames == args.num_frames or reached_edge)
        wait_for_viewer_close(viewer, args.keep_open)
    finally:
        if recorder is not None:
            recorder.close(completed=False)
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
