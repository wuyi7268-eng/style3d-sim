"""MicroDuck play entry point for MuJoCo and SynReal."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import time

import torch

from examples.microduck.common import create_env, setup
from examples.microduck.policy_spec import HOME


# Commands are offsets from HOME. This fixed command requests the neutral,
# raised-head pose while leaving the policy's head actions unconstrained.
HEAD_UP_COMMAND = (-HOME[5], -HOME[6], 0.0, 0.0)


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
    if not terms["head_command"]["func"].endswith(".zero_command_padding"):
        env.commands[:, 3:7] = torch.tensor(HEAD_UP_COMMAND, device=env.device, dtype=obs.dtype)
    # Unused command slots must remain zero even when a cached observation is refreshed.
    # Their saved normalizer variance is zero; injecting a pose here shifts policy inputs dramatically.
    for name, start, end in (("command", 0, 3), ("head_command", 3, 7), ("body_command", 7, 13)):
        target = obs[:, 48 + start : 48 + end]
        if terms[name]["func"].endswith(".zero_command_padding"):
            target.zero_()
        else:
            target.copy_(env.commands[:, start:end])


def main(argv=None):
    args, _training_config, play_cfg = setup("play", argv)
    if args.policy == "walk":
        from examples.microduck.official_play import run_walk

        return run_walk(args)
    env = create_env(args)
    # Keep the trained observation layout, delays, and fixed calibration offsets,
    # while removing per-tick random actor noise for deterministic visual replay.
    env.official.observations.spec["actor"]["enable_corruption"] = False
    print("Playback actor observation noise: disabled", flush=True)
    viewer = env.viewer
    try:
        from rsl_rl.runners import OnPolicyRunner

        runner = OnPolicyRunner(env, play_cfg, log_dir=None, device=args.device)
        runner.load(str(args.checkpoint), load_optimizer=False)
        if env.official is not None:
            env.official.common_step_counter = (runner.current_learning_iteration + 1) * play_cfg["num_steps_per_env"]
            # Playback starts from the checkpoint's recovery-pose distribution.
            env.reset()
        # RSL-RL's inference wrapper restores and applies the saved observation normalizer.
        policy = runner.get_inference_policy(device=args.device)
        if viewer is not None and env.task in ("velocity", "velstand"):
            print(
                "Press I/K: forward/back; J/L: lateral; U/O: turn. Each press adds one increment; release keeps it.", flush=True
            )
        from examples.microduck.playback import PolicyPlayback

        playback = None if args.full_step else PolicyPlayback(env)
        previous_command = None
        key_was_down = {}
        command = list(args.command)
        frames = 0
        warmup = min(20, args.num_frames // 4)
        measured_start = time.monotonic()
        obs, _ = env.get_observations()
        with torch.inference_mode():
            for _ in range(args.num_frames):
                if viewer is not None and not viewer.is_running():
                    break
                start = time.monotonic()
                if frames == warmup:
                    measured_start = start
                if env.task in ("velocity", "velstand"):
                    previous_command = update_velocity_command(env, viewer, command, key_was_down, previous_command)
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
                obs = env.step_playback(actions) if playback is None else playback.step(actions)
                frames += 1
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
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
