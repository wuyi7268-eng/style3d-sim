"""Replay the published running ONNX through the shared 50 Hz MicroDuck physics."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp

from examples.example_robot_microduck import WalkPolicy

from .common import create_env, positive_int, setup
from .play import configure_playback_boundaries, manual_reset_requested, update_velocity_command
from .policy_spec import HOME, JOINT_NAMES


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, default=Path("logs/pretrained/microduck-running/policy.onnx"))
    parser.add_argument("--solver", choices=("synreal", "mujoco"), default="synreal")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--viewer", choices=("gl", "null"), default="gl")
    parser.add_argument("--command", type=float, nargs=3, default=(1., 0., 0.), metavar=("VX", "VY", "YAW"))
    parser.add_argument("--num-frames", type=positive_int, default=1500)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if not args.onnx.is_file():
        parser.error(f"ONNX file not found: {args.onnx}; download policy.onnx from HannesVonEssen/microduck-running")
    if not np.isfinite(args.command).all():
        parser.error("--command must be finite")
    policy = WalkPolicy(args.onnx)
    if tuple(policy.joint_names) != JOINT_NAMES or not np.allclose(policy.default_pose, HOME, atol=5e-4):
        raise ValueError("ONNX joint order/HOME does not match the running robot")
    if policy.action_scale != 1.0:
        raise ValueError("Published running policy requires action_scale=1")
    # Resolve current shared physics without interpreting an upstream checkpoint
    # as a local RSL-RL checkpoint. No runner or optimizer is constructed.
    settings, _, _ = setup("train", ["--task", "running", "--solver", args.solver, "--num-envs", "1",
                                    "--device", args.device, "--viewer", args.viewer, "--no-randomize-foot-friction"])
    settings.mode = "play"
    env = create_env(settings)
    frames, speeds, upright = 0, [], []
    try:
        configure_playback_boundaries(env)
        if env.viewer is not None:
            env.viewer.set_camera(pos=wp.vec3(.8, -1., .72), pitch=-26., yaw=129.)
            env.viewer.renderer.set_title("MicroDuck | Published running ONNX | " + args.solver)
        print("Published running ONNX: 50 Hz, embedded normalization, unclipped actions, local PD (not BAM).", flush=True)
        print("I/K: forward; J/L: lateral; U/O: yaw; X: stop; R: reset; Space: pause.", flush=True)
        command, keys, previous = list(args.command), {}, None
        with torch.inference_mode():
            while frames < args.num_frames:
                viewer = env.viewer
                if viewer is not None and not viewer.is_running():
                    break
                if manual_reset_requested(viewer, keys):
                    env.reset()
                    policy.reset()
                if viewer is not None and viewer.is_paused():
                    env.render()
                    time.sleep(.02)
                    continue
                start = time.monotonic()
                if viewer is not None and viewer.is_key_down("x"):
                    command[:] = [0., 0., 0.]
                previous = update_velocity_command(env, viewer, command, keys, previous)
                q, qd, pose, velocity = env._views()
                # The ONNX already contains normalization. Feed deployment-style
                # proprioception with neutral head/body slots, not local noisy observations.
                target = policy.infer(pose[0, 3:7].cpu().numpy(), velocity[0, 3:6].cpu().numpy(),
                                      q[0, env.qi].cpu().numpy(), qd[0, env.qdi].cpu().numpy(), command)
                env.step_playback(torch.as_tensor(target, device=env.device).unsqueeze(0) - env.home)
                _, _, _, linear, _, gravity = env._kinematics()
                speeds.append(float(linear[0, 0]))
                upright.append(float(-gravity[0, 2]))
                frames += 1
                if viewer is not None and not args.benchmark:
                    time.sleep(max(0., env.step_dt - (time.monotonic() - start)))
        report = {"onnx": str(args.onnx), "solver": args.solver, "frames": frames, "resets": env.total_resets,
                  "mean_forward_speed": float(np.mean(speeds)) if speeds else None,
                  "minimum_upright_cos": min(upright) if upright else None}
        print(json.dumps(report), flush=True)
        if args.report:
            args.report.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
