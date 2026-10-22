"""Replay official MicroDuck ONNX policies with MuJoCo or inspect their experimental SynReal transfer."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import queue
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "examples/microduck/microduck_rl-develop"
MODELS = ROOT / "logs/pretrained/microduck"


class KeyboardCommand:
    """Queue viewer-thread events and apply velocity changes on the simulation thread."""

    def __init__(self):
        self.events = queue.SimpleQueue()

    def on_key(self, key):
        self.events.put(key)

    def update(self, command):
        changed = False
        while not self.events.empty():
            key = self.events.get()
            # MuJoCo forwards GLFW key codes; movement keys change persistent speed targets.
            if key in (ord("W"), 265):
                command[0] += 0.05
            elif key in (ord("S"), 264):
                command[0] -= 0.05
            elif key in (ord("A"), 263):
                command[2] += 0.2
            elif key in (ord("D"), 262):
                command[2] -= 0.2
            elif key == ord("Q"):
                command[1] += 0.05
            elif key == ord("E"):
                command[1] -= 0.05
            elif key in (ord(" "), ord("X")):
                command[:] = [0.0, 0.0, 0.0]
            else:
                continue
            for i, limit in enumerate((0.3, 0.2, 1.0)):
                command[i] = max(-limit, min(limit, round(command[i], 6)))
            changed = True
        return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", choices=["mujoco"], default="mujoco")
    parser.add_argument("--policy", choices=["velstand", "alpha_walking", "alpha_stand"], default="alpha_walking")
    parser.add_argument("--viewer", choices=["gl", "null"], default="gl")
    parser.add_argument("--seconds", type=float, help="duration; default 20 seconds, or until window closes with --keyboard")
    parser.add_argument("--velocity", type=float, default=0.2, help="forward velocity command in m/s")
    parser.add_argument("--lateral-velocity", type=float, default=0.0, help="leftward velocity command in m/s")
    parser.add_argument("--yaw-rate", type=float, default=0.0, help="left turn command in rad/s")
    parser.add_argument("--keyboard", action="store_true", help="start at rest and control velocity with viewer keyboard")
    parser.add_argument("--actuator", choices=["metadata", "bam"], default="metadata")
    args = parser.parse_args()
    if args.seconds is not None and args.seconds <= 0:
        parser.error("--seconds must be positive")
    if args.keyboard and args.viewer != "gl":
        parser.error("--keyboard requires --viewer gl")
    if args.policy == "alpha_stand" and args.keyboard:
        parser.error("keyboard locomotion requires --policy alpha_walking or velstand")
    executable = PROJECT / ".venv/bin/python"
    if Path(sys.prefix).resolve() != (PROJECT / ".venv").resolve():
        return subprocess.call([str(executable), str(Path(__file__).resolve()), *sys.argv[1:]], cwd=ROOT)

    import mujoco
    import numpy as np
    import onnxruntime as ort

    weights = MODELS / f"{args.policy}.onnx"
    if not weights.is_file():
        parser.error(f"download the official pretrained weights first: {weights}")
    script = PROJECT / "scripts/infer_policy.py"
    # Reuse upstream observation, normalization, action and BAM logic without opening its interactive viewer.
    spec = importlib.util.spec_from_file_location("microduck_upstream_inference", script)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    os.chdir(PROJECT)
    controller = None
    metadata = ort.InferenceSession(str(weights)).get_modelmeta().custom_metadata_map
    if args.actuator == "bam":
        bam = upstream.load_bam_model(kp_fw=200.0, vin=7.4, max_current=None)
        model, data, controller, _ = upstream.load_mujoco_with_bam(
            upstream.MICRODUCK_XML, bam, timestep=0.005, vin_drop_gain=0.1, vin_min=6.0
        )
    else:
        model = mujoco.MjModel.from_xml_path(upstream.MICRODUCK_XML)
        model.opt.timestep = 0.005
        names = [model.joint(int(j)).name for j in model.actuator_trnid[:, 0]]
        if names != metadata["joint_names"].split(","):
            raise ValueError("Policy and MuJoCo actuator joint order differ")
        kp = np.fromstring(metadata["joint_stiffness"], sep=",")
        kd = np.fromstring(metadata["joint_damping"], sep=",")
        if kp.shape != (model.nu,) or kd.shape != (model.nu,):
            raise ValueError("Policy actuator metadata has an invalid shape")
        # The published graph records position gains; preserve the XML's passive joint damping and limits.
        model.actuator_gainprm[:, 0] = kp
        model.actuator_biasprm[:, 1] = -kp
        model.actuator_biasprm[:, 2] = -kd
        data = mujoco.MjData(model)
    policy = upstream.PolicyInference(
        model,
        data,
        walking_onnx_path=str(weights),
        bam_ctrl=controller,
        new_cmd_obs=True,
        use_projected_gravity=True,
        action_scale=float(metadata["action_scale"]),
    )
    command = [0.0, 0.0, 0.0] if args.keyboard else [args.velocity, args.lateral_velocity, args.yaw_rate]
    policy.set_vel_cmd(*command)
    start = policy._trunk_qpos_adr
    data.qpos[start : start + 7] = [0.0, 0.0, 0.125, 1.0, 0.0, 0.0, 0.0]
    data.qpos[policy.joint_qpos_indices] = policy.default_pose
    if controller is not None:
        controller.reset(data.qpos)
    policy.set_position_targets(policy.default_pose)
    mujoco.mj_forward(model, data)
    initial_x = float(data.qpos[start])
    heights, upright, errors, yaw_rates = [], [], [], []
    seconds = args.seconds if args.seconds is not None else (None if args.keyboard else 20.0)
    frames = max(1, round(seconds * 50)) if seconds is not None else None
    keyboard = KeyboardCommand()
    viewer = None
    if args.viewer == "gl":
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(model, data, key_callback=keyboard.on_key if args.keyboard else None)
        viewer.cam.distance = 0.7
        viewer.cam.elevation = -20
        if args.keyboard:
            print(
                "Click the viewer to focus. W/S or Up/Down: forward/back speed; A/D or Left/Right: turn; "
                "Q/E: strafe; Space/X: zero all velocity commands. Targets persist after key release. "
                "Close the window to exit.",
                flush=True,
            )
    completed = 0
    while frames is None or completed < frames:
        if viewer is not None and not viewer.is_running():
            break
        started = time.monotonic()
        if args.keyboard and keyboard.update(command):
            policy.set_vel_cmd(*command)
        action = policy.infer()
        if not np.isfinite(action).all():
            raise FloatingPointError("Pretrained policy produced a non-finite action")
        policy.apply_action(action)
        for _ in range(4):
            if controller is not None:
                controller.update()
            mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise FloatingPointError("MuJoCo produced a non-finite state")
        heights.append(float(data.qpos[start + 2]))
        upright.append(float(-policy.get_projected_gravity()[2]))
        errors.append(float(abs(data.qvel[0] - command[0])))
        yaw_rates.append(float(policy.get_base_ang_vel()[2]))
        completed += 1
        if viewer is not None:
            viewer.cam.lookat[:] = data.qpos[start : start + 3]
            viewer.sync()
            time.sleep(max(0.0, 0.02 - (time.monotonic() - started)))
    if viewer is not None:
        viewer.close()
    if not completed:
        return 0
    report = {
        "policy": str(weights),
        "backend": "mujoco_cpu",
        "actuator": args.actuator,
        "frames": completed,
        "simulation_seconds": completed / 50,
        "command_velocity": command[0],
        "final_command": command,
        "keyboard": args.keyboard,
        "mean_yaw_rate": float(np.mean(yaw_rates)),
        "distance_x": float(data.qpos[start] - initial_x),
        "mean_velocity_x": float((data.qpos[start] - initial_x) / (completed / 50)),
        "mean_world_vx_error": float(np.mean(errors)),
        "min_trunk_height": min(heights),
        "min_upright_cos": min(upright),
        "fraction_upright": float(np.mean(np.array(upright) > 0.7)),
        "resets": 0,
    }
    target = MODELS / f"evaluation_{args.policy}_{args.actuator}.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
