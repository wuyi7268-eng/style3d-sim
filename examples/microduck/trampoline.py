"""MicroDuck trampoline demo with explicit external assistance, not a learned jumping policy.

Run: .venv/bin/python -m examples.microduck.trampoline
Use --passive to disable assistance and inspect the natural cloth/robot dynamics.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp

from .env import MicroDuckEnv, PhysicsConfig
from .scenes import SCENE_CONFIG, login


@wp.kernel
def assist_bounce(
    poses: wp.array[wp.transform], velocities: wp.array[wp.spatial_vector], masses: wp.array[float],
    forces: wp.array[wp.spatial_vector], root: int, acceleration: float,
):
    body = wp.tid()
    pose = poses[root]
    position = wp.transform_get_translation(pose)
    velocity = velocities[root]
    # Distributed forces avoid loading the tiny neck/leg joints through a single link.
    force = masses[body] * wp.vec3(
        wp.clamp(-30.0 * position[0] - 8.0 * velocity[0], -10.0, 10.0),
        wp.clamp(-30.0 * position[1] - 8.0 * velocity[1], -10.0, 10.0), acceleration,
    )
    torque = wp.vec3(0.0)
    if body == root:
        q = wp.normalize(wp.transform_get_rotation(pose))
        sign = float(1.0)
        if q[3] < 0.0:
            sign = -1.0
        torque = -2.0 * sign * wp.vec3(q[0], q[1], q[2]) - 0.12 * wp.spatial_bottom(velocity)
        length = wp.length(torque)
        if length > 0.6:
            torque = torque * (0.6 / length)
    forces[body] = forces[body] + wp.spatial_vector(force, torque)


class TrampolineDemo:
    def __init__(self, viewer, args):
        self.viewer, self.args = viewer, args
        self.acceleration = 0.0
        self.pulse_remaining = 0.0
        self.cooldown = 0.0
        self.jump_count = 0
        self.reset_key = False
        self.samples = []
        if not args.preview:
            login(args.force_login)
        self.env = MicroDuckEnv(
            solver="synreal", num_envs=1, device=args.device, terrain="trampoline",
            interactive=True, controller="home", preview=args.preview, clearance=args.drop_height,
            randomize_foot_friction=False, viewer=self, physics=PhysicsConfig.home(),
        )
        self.model = self.env.model
        viewer.set_model(self.model)
        self.env.terrain.configure_viewer(viewer)
        self.cloth = self.env.terrain.cloth_indices.cpu().numpy()
        flags = self.model.particle_flags.numpy()[self.cloth]
        self.fixed = self.cloth[(flags & 1) == 0]
        self.rest = self.model.particle_q.numpy().copy()
        self.rest_height = float(self.env.initial_state.body_q.numpy()[self.env.root, 2]
                                 - SCENE_CONFIG.trampoline_height - args.drop_height)
        self.targets = torch.zeros_like(self.env.actions)
        print("MicroDuck trampoline: " + ("static preview" if args.preview else
              "passive drop" if args.passive else "ASSISTED bounce (external lift and balance; no policy)"))
        print("R: reset robot and cloth. Space: pause. --passive: disable all assistance.", flush=True)

    def __getattr__(self, name):
        return getattr(self.viewer, name)

    def apply_forces(self, state):
        self.viewer.apply_forces(state)
        if not self.args.passive:
            wp.launch(assist_bounce, dim=self.model.body_count, inputs=[
                state.body_q, state.body_qd, self.model.body_mass, state.body_f, self.env.root, self.acceleration,
            ], device=self.model.device)

    def reset(self):
        self.env.reset()
        self.acceleration = self.pulse_remaining = self.cooldown = 0.0
        self.jump_count = 0
        self.samples.clear()

    def step(self):
        pressed = self.viewer.is_key_down("r")
        if pressed and not self.reset_key:
            self.reset()
        self.reset_key = pressed
        if self.args.preview:
            return
        env = self.env
        pose = wp.to_torch(env.states[0].body_q)[env.root]
        velocity = wp.to_torch(env.states[0].body_qd)[env.root]
        support = float(env.terrain.surface_heights(pose[None, None, :3])[0, 0])
        height, vz = float(pose[2]), float(velocity[2])
        self.cooldown = max(0.0, self.cooldown - env.step_dt)
        # A contact-height gate supplies a finite takeoff pulse; flight remains ballistic.
        # This is intentionally visible demo assistance, never part of train/play.
        near_bed = height - support < self.rest_height + 0.012
        if not self.args.passive and near_bed and vz < 0.25 and self.cooldown == 0:
            self.pulse_remaining = 0.10
            self.cooldown = 0.40
            self.jump_count += 1
        target_speed = math.sqrt(2 * 9.81 * self.args.jump_height)
        self.acceleration = (float(np.clip((target_speed - vz) / 0.06 + 9.81, 0, 45))
                             if self.pulse_remaining > 0 else 0.0)
        self.pulse_remaining = max(0.0, self.pulse_remaining - env.step_dt)
        env.step(self.targets)
        self.check_state()
        pose = env.states[0].body_q.numpy()[env.root]
        bed_z = env.states[0].particle_q.numpy()[self.cloth, 2]
        self.samples.append({"time": env.sim_time, "root_z": float(pose[2]),
                             "bed_min_z": float(bed_z.min()), "lift_acceleration": self.acceleration})

    def check_state(self):
        for name in ("body_q", "body_qd", "particle_q", "particle_qd"):
            if not np.isfinite(getattr(self.env.states[0], name).numpy()).all():
                raise FloatingPointError(f"Trampoline has non-finite {name}")
        particles = self.env.states[0].particle_q.numpy()
        if not np.allclose(particles[self.fixed], self.rest[self.fixed], atol=1e-6, rtol=0):
            raise AssertionError("Trampoline perimeter moved")
        count = int(self.env.contacts.soft_contact_count.numpy()[0])
        if count >= self.env.contacts.soft_contact_max:
            raise RuntimeError("Trampoline soft contact capacity exhausted")

    def metrics(self):
        return {"mode": "preview" if self.args.preview else "passive" if self.args.passive else "assisted",
                "seconds": self.env.sim_time, "takeoff_pulses": self.jump_count,
                "physics": self.env.cfg["physics"], "samples": self.samples}


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer", choices=("gl", "null"), default="gl")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-frames", type=int, default=36000)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--preview", action="store_true", help="Static scene; no login or simulation")
    parser.add_argument("--passive", action="store_true", help="Disable external takeoff and balance forces")
    parser.add_argument("--drop-height", type=float, default=0.08, help="Initial foot clearance in metres")
    parser.add_argument("--jump-height", type=float, default=0.10, help="Assisted ballistic height target in metres")
    parser.add_argument("--force-login", action="store_true")
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--report", type=Path, help="Write dynamics samples as JSON")
    return parser


def main(argv=None):
    parser = create_parser()
    args = parser.parse_args(argv)
    if args.num_frames < 1:
        parser.error("num-frames must be positive")
    if not math.isfinite(args.drop_height) or args.drop_height < 0:
        parser.error("drop-height must be finite and nonnegative")
    if not math.isfinite(args.jump_height) or not 0 < args.jump_height <= 0.3:
        parser.error("jump-height must be finite and in (0, 0.3] metres")
    if args.screenshot and args.viewer != "gl":
        parser.error("screenshot requires --viewer gl")
    wp.set_device(args.device)
    from newton.viewer import ViewerGL, ViewerNull

    viewer = ViewerGL(width=1280, height=900, headless=args.headless) if args.viewer == "gl" else ViewerNull(args.num_frames)
    try:
        demo = TrampolineDemo(viewer, args)
        frame = None
        for _ in range(args.num_frames):
            if not viewer.is_running():
                break
            start = time.monotonic()
            if viewer.should_step():
                demo.step()
            demo.env.render()
            if args.screenshot:
                frame = viewer.get_frame(target_image=frame, render_ui=False)
            if args.viewer == "gl" and not args.headless:
                time.sleep(max(0.0, demo.env.step_dt - (time.monotonic() - start)))
        if args.screenshot and frame is not None:
            from PIL import Image

            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(frame.numpy()).save(args.screenshot)
        report = demo.metrics()
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: value for key, value in report.items() if key != "samples"}, indent=2))
    finally:
        viewer.close()


if __name__ == "__main__":
    from examples.microduck.trampoline import main as demo_main

    demo_main()
