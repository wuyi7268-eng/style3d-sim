"""Shared MicroDuck scenes: geometry, contacts, resets, and standalone pad playback.

Training/checkpoint playback select terrain through train.py/play.py.
Standalone pad preview: python -m examples.microduck.scenes --preview
Standalone pad playback: python -m examples.microduck.scenes --checkpoint <model.pt>
"""

from __future__ import annotations

import argparse
import getpass
import json
import math
import os
import sys
import time
from copy import copy
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from numbers import Real
from pathlib import Path
from types import SimpleNamespace

import newton
import numpy as np
import torch
import warp as wp

from .policy_spec import EFFORT, KD, KP, filter_authored_collider

# Optional hammock, sand, stair, and fluid-pool scenes

SCENE_KINDS = ("hammock", "cloth", "sand", "steps", "pool", "trampoline", "elastic_bridge")


@dataclass(frozen=True)
class SceneConfig:
    """Geometry and material choices for the experimental support scenes."""

    hammock_length: float = 1.8
    hammock_width: float = 0.9
    hammock_resolution: int = 15
    hammock_sag: float = 0.18
    hammock_density: float = 2.0
    hammock_stretch: float = 1200.0
    hammock_bend: float = 2.0e-4
    cloth_length: float = 1.2
    cloth_width: float = 1.2
    cloth_resolution: int = 25
    cloth_sag: float = 0.22
    cloth_density: float = 2.0
    cloth_stretch: float = 400.0
    cloth_bend: float = 2.0e-4
    # Keep both legacy dimensions; the circular bed fits inside the smaller one.
    trampoline_length: float = 0.8
    trampoline_width: float = 0.8
    trampoline_height: float = 0.40
    trampoline_resolution: int = 25
    trampoline_density: float = 0.3
    trampoline_stretch: float = 1800.0
    trampoline_damping: float = 0.05
    elastic_bridge_length: float = 1.2
    elastic_bridge_width: float = 0.29
    elastic_bridge_height: float = 0.65
    elastic_bridge_sag: float = 0.04
    elastic_bridge_cells_x: int = 40
    elastic_bridge_cells_y: int = 10
    elastic_bridge_density: float = 0.6
    elastic_bridge_stretch: float = 1600.0
    elastic_bridge_bend: float = 2.0
    elastic_bridge_damping: float = 0.2
    elastic_bridge_approach_length: float = 0.8
    elastic_bridge_approach_width: float = 0.8
    elastic_bridge_spawn_distance: float = 0.4
    sand_length: float = 0.32
    sand_width: float = 0.24
    sand_depth: float = 0.02
    sand_spacing: float = 0.002
    sand_density: float = 1500.0
    sand_stiffness: float = 12000.0
    sand_friction: float = 0.8
    sand_cohesion: float = 0.0
    pool_length: float = 2
    pool_width: float = 0.4
    pool_depth: float = 0.30
    pool_wall_height: float = 0.40
    pool_wall_thickness: float = 0.025
    pool_spacing: float = 0.02
    pool_density: float = 5000.0
    pool_bulk_modulus: float = 200000.0
    pool_viscosity: float = 0.001
    pool_water_color: tuple[float, float, float] = (0.30, 0.72, 0.82)
    pool_water_opacity: float = 0.22
    pool_water_roughness: float = 0.08
    step_count: int = 6
    step_run: float = 0.24
    step_rise: float = 0.055
    step_width: float = 1.4


SCENE_CONFIG = SceneConfig()

_BRIDGE_EQUILIBRIA = {}


def settled_bridge_state(config, physics, device):
    """Relax an unloaded template without changing its constitutive rest shape or runtime damping."""
    import style3d_sim as s3d

    key = (json.dumps(asdict(config), sort_keys=True), json.dumps(asdict(physics), sort_keys=True), str(device))
    if key in _BRIDGE_EQUILIBRIA:
        return _BRIDGE_EQUILIBRIA[key]
    builder = s3d.ModelBuilder()
    add_elastic_bridge(builder, config)
    model = builder.finalize(device=str(device))
    model.set_gravity((0.0, 0.0, -9.81))
    solver = s3d.NewtonSolver(model)
    physics.configure_synreal(solver)
    collision = newton.CollisionPipeline(model, enable_rigid_soft_full_surface_contact=True, soft_contact_max=8192)
    contacts = collision.contacts()
    solver.prepare_contacts(contacts)
    current, next_state = model.state(), model.state()
    control = model.control()
    indices = np.unique(model.tri_indices.numpy())
    rest = model.particle_q.numpy()[indices].copy()
    quiet_steps = 0
    print("Elastic bridge: relaxing unloaded support before robot spawn...", flush=True)
    for step in range(4000):
        current.clear_forces()
        collision.collide(current, contacts)
        solver.step(current, next_state, control, contacts, physics.physics_dt)
        current, next_state = next_state, current
        velocity = wp.to_torch(current.particle_qd)
        speed = velocity.norm(dim=1).max().item()
        if not math.isfinite(speed) or not torch.isfinite(wp.to_torch(current.particle_q)).all():
            raise FloatingPointError("Elastic bridge relaxation produced non-finite state")
        # Numerical relaxation is initialization only; normal playback uses the authored material damping.
        velocity.mul_(0.9)
        quiet_steps = quiet_steps + 1 if speed < 1.0e-4 else 0
        if quiet_steps >= 100:
            break
    else:
        raise RuntimeError("Elastic bridge failed to settle within 4000 substeps; check geometry/material parameters")
    equilibrium = current.particle_q.numpy()[indices].copy()
    current.particle_qd.zero_()
    # Releasing all artificial damping must not restart a visible oscillation.
    drift = 0.0
    for _ in range(math.ceil(0.5 / physics.physics_dt)):
        current.clear_forces()
        collision.collide(current, contacts)
        solver.step(current, next_state, control, contacts, physics.physics_dt)
        current, next_state = next_state, current
        drift = max(drift, float(np.linalg.norm(current.particle_q.numpy()[indices] - equilibrium, axis=1).max()))
        if not np.isfinite(current.particle_q.numpy()).all() or drift > 5.0e-5:
            raise RuntimeError("Elastic bridge equilibrium drifts after relaxation; refusing an unsettled reset state")
    report = {"relaxation_substeps": step + 1, "release_drift_m": drift,
              "max_settling_displacement_m": float(np.linalg.norm(equilibrium - rest, axis=1).max())}
    result = (rest, equilibrium, report)
    _BRIDGE_EQUILIBRIA[key] = result
    print(f"Elastic bridge ready: release drift={drift:.3g} m, relaxation steps={step + 1}", flush=True)
    return result


class _TransparentWaterMesh:
    """Local GL adapter for Newton's opaque-only mesh material API."""

    cast_shadow = False
    draw_edge = False

    def __init__(self, mesh, gl, opacity):
        self.mesh, self.gl, self.opacity = mesh, gl, opacity

    def __getattr__(self, name):
        return getattr(self.mesh, name)

    def render(self):
        if self.hidden:
            return
        gl = self.gl
        enabled = gl.glIsEnabled(gl.GL_BLEND)
        depth = gl.GLboolean()
        gl.glGetBooleanv(gl.GL_DEPTH_WRITEMASK, depth)
        color = (gl.GLfloat * 4)()
        gl.glGetFloatv(gl.GL_BLEND_COLOR, color)
        names = (gl.GL_BLEND_SRC_RGB, gl.GL_BLEND_DST_RGB, gl.GL_BLEND_SRC_ALPHA, gl.GL_BLEND_DST_ALPHA,
                 gl.GL_BLEND_EQUATION_RGB, gl.GL_BLEND_EQUATION_ALPHA)
        values = []
        for name in names:
            value = gl.GLint()
            gl.glGetIntegerv(name, value)
            values.append(value.value)
        try:
            # Draw after opaque objects, keeping their depth occlusion but allowing
            # the pool bottom and submerged bodies to show through the water.
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendEquationSeparate(gl.GL_FUNC_ADD, gl.GL_FUNC_ADD)
            gl.glBlendColor(0.0, 0.0, 0.0, self.opacity)
            gl.glBlendFuncSeparate(gl.GL_CONSTANT_ALPHA, gl.GL_ONE_MINUS_CONSTANT_ALPHA, gl.GL_ZERO, gl.GL_ONE)
            gl.glDepthMask(False)
            self.mesh.render()
        finally:
            gl.glDepthMask(depth.value)
            gl.glBlendColor(*color)
            gl.glBlendFuncSeparate(*values[:4])
            gl.glBlendEquationSeparate(*values[4:])
            if not enabled:
                gl.glDisable(gl.GL_BLEND)


def _box(builder, center, half, color, label, cfg=None):
    shape = builder.add_shape_box(
        -1,
        xform=wp.transform(wp.vec3(*center), wp.quat_identity()),
        hx=half[0],
        hy=half[1],
        hz=half[2],
        cfg=cfg,
        label=label,
        color=color,
    )
    return shape


def add_scene(builder, kind, config=SCENE_CONFIG):
    """Author one scene template; callers replicate it for batched training."""
    if kind not in SCENE_KINDS:
        raise ValueError(f"unsupported optional scene: {kind}")
    cfg = builder.default_shape_cfg.copy()
    cfg.mu = config.sand_friction
    if kind == "cloth":
        return add_corner_cloth(builder, config)
    if kind == "trampoline":
        return add_trampoline(builder, config)
    if kind == "elastic_bridge":
        return add_elastic_bridge(builder, config)
    if kind == "pool":
        if not 0 < config.pool_depth < config.pool_wall_height:
            raise ValueError("Pool water depth must be positive and below the wall height")
        spacing = config.pool_spacing
        if spacing <= 0 or min(config.pool_length, config.pool_width, config.pool_depth) < spacing:
            raise ValueError("Pool dimensions must each contain at least one fluid particle")
        half_x, half_y = config.pool_length / 2, config.pool_width / 2
        thick, height = config.pool_wall_thickness, config.pool_wall_height
        if thick <= 0:
            raise ValueError("Pool wall thickness must be positive")
        cfg.ke, cfg.kd, cfg.mu, cfg.kf = 2.0e5, 20.0, 0.0, 0.0
        _box(builder, (0, 0, -thick / 2), (half_x + thick, half_y + thick, thick / 2),
             (0.60, 0.76, 0.80), "pool_floor", cfg)
        for sign in (-1, 1):
            _box(builder, (sign * (half_x + thick / 2), 0, height / 2),
                 (thick / 2, half_y + thick, height / 2), (0.68, 0.82, 0.86), f"pool_wall_x_{sign}", cfg)
            _box(builder, (0, sign * (half_y + thick / 2), height / 2),
                 (half_x, thick / 2, height / 2), (0.68, 0.82, 0.86), f"pool_wall_y_{sign}", cfg)
        lower = np.array([-half_x, -half_y, 0.0])
        counts = np.floor(np.array([config.pool_length, config.pool_width, config.pool_depth]) / spacing + 1e-6).astype(int)
        axes = [(np.arange(n) + 0.5) * spacing + low for n, low in zip(counts, lower, strict=True)]
        points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        # Calibrate mass against the discrete Poly6 kernel, matching the native fluid example.
        offsets = np.stack(np.meshgrid(*([np.arange(-2, 3)] * 3), indexing="ij"), axis=-1).reshape(-1, 3) * spacing
        h = 2 * spacing
        weights = 315 / (64 * math.pi) * np.maximum(1 - np.sum(offsets**2, axis=1) / h**2, 0)**3
        volume = h**3 / weights.sum()
        builder.add_particles(pos=points, vel=np.zeros_like(points),
                              mass=np.full(len(points), config.pool_density * volume),
                              radius=np.full(len(points), 0.45 * spacing))
        return {"kind": kind, "config": asdict(config), "support": "native_particle_fluid",
                "fluid_count_per_world": len(points), "fluid_rest_volume": float(volume)}
    if kind == "steps":
        for index in range(config.step_count):
            height = config.step_rise * (index + 1)
            _box(
                builder,
                (index * config.step_run, 0.0, height / 2),
                (config.step_run / 2, config.step_width / 2, height / 2),
                (0.58, 0.51, 0.43),
                f"training_step_{index}",
                cfg,
            )
        _box(builder, (-0.65, 0.0, -0.06), (0.65, 2.0, 0.06), (0.42, 0.48, 0.40), "approach_ground", cfg)
        return {"kind": kind, "config": asdict(config), "support": "rigid_stair_treads"}
    _box(
        builder,
        (0.0, 0.0, -0.08),
        (config.sand_length / 2 + 0.2, config.sand_width / 2 + 0.2, 0.08),
        (0.37, 0.32, 0.22),
        "scene_bed_tray",
        cfg,
    )
    if kind == "sand":
        spacing = config.sand_spacing
        axes = [
            np.arange(-extent / 2 + spacing / 2, extent / 2, spacing)
            for extent in (config.sand_length, config.sand_width)
        ]
        zs = np.arange(spacing / 2, config.sand_depth, spacing)
        points = np.stack(np.meshgrid(*axes, zs, indexing="ij"), axis=-1).reshape(-1, 3)
        radius = 0.48 * spacing
        masses = np.full(len(points), config.sand_density * spacing**3, dtype=np.float32)
        builder.add_particles(
            pos=points, vel=np.zeros_like(points), mass=masses, radius=np.full(len(points), radius, dtype=np.float32)
        )
        return {
            "kind": kind,
            "config": asdict(config),
            "support": "native_granular_bed",
            "grain_count_per_world": len(points),
            "grain_radius": radius,
        }
    from newton.solvers import style3d

    # Foot colliders can fit between cloth vertices. Full-face contacts need
    # signed-distance fields on the robot meshes before model finalization.
    provision_pad_contacts(builder)
    newton.solvers.SolverStyle3D.register_custom_attributes(builder)

    resolution = config.hammock_resolution
    start = len(builder.particle_q)
    xs = np.linspace(-config.hammock_length / 2, config.hammock_length / 2, resolution)
    ys = np.linspace(-config.hammock_width / 2, config.hammock_width / 2, resolution)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    zz = 1.15 - config.hammock_sag * (1 - (2 * xx / config.hammock_length) ** 2)
    vertices = np.stack((xx, yy, zz), axis=-1).reshape(-1, 3).astype(np.float32)
    faces = []
    for row in range(resolution - 1):
        for col in range(resolution - 1):
            a = row * resolution + col
            faces.extend(((a, a + resolution, a + resolution + 1), (a, a + resolution + 1, a + 1)))
    style3d.add_cloth_mesh(
        builder,
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        vertices=vertices.tolist(),
        indices=np.asarray(faces, dtype=np.int32).reshape(-1).tolist(),
        density=config.hammock_density,
        tri_aniso_ke=wp.vec3(config.hammock_stretch, config.hammock_stretch, config.hammock_stretch),
        edge_aniso_ke=wp.vec3(config.hammock_bend),
        tri_kd=1.0,
        edge_kd=0.01,
        particle_radius=0.006,
        label="microduck_hammock",
    )
    flags = np.asarray(builder.particle_flags)
    for row in (0, resolution - 1):
        flags[start + row * resolution : start + (row + 1) * resolution] &= ~int(newton.ParticleFlags.ACTIVE)
    builder.particle_flags = flags.tolist()
    for x in (-config.hammock_length / 2, config.hammock_length / 2):
        _box(builder, (x, 0, 1.17), (0.035, config.hammock_width / 2 + 0.08, 0.035), (0.24, 0.25, 0.27), "hammock_rail")
    return {
        "kind": kind,
        "config": asdict(config),
        "support": "sagging_cloth_between_fixed_ends",
        "particle_range": [start, start + resolution * resolution],
    }


def add_corner_cloth(builder, config):
    """Create a square sagging cloth supported only at its four corners."""
    from newton.solvers import style3d

    provision_pad_contacts(builder)
    newton.solvers.SolverStyle3D.register_custom_attributes(builder)
    n = config.cloth_resolution
    if isinstance(n, bool) or not isinstance(n, int) or n < 3:
        raise ValueError("cloth_resolution must be an integer >= 3")
    if min(config.cloth_length, config.cloth_width, config.cloth_sag) <= 0:
        raise ValueError("Cloth dimensions and sag must be positive")
    xs = np.linspace(-config.cloth_length / 2, config.cloth_length / 2, n)
    ys = np.linspace(-config.cloth_width / 2, config.cloth_width / 2, n)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    shape = (1 - (2 * xx / config.cloth_length) ** 2) * (1 - (2 * yy / config.cloth_width) ** 2)
    zz = 1.15 - config.cloth_sag * shape
    vertices = np.stack((xx, yy, zz), axis=-1).reshape(-1, 3).astype(np.float32)
    faces = []
    for row in range(n - 1):
        for col in range(n - 1):
            a = row * n + col
            faces.extend(((a, a + n, a + n + 1), (a, a + n + 1, a + 1)))
    start = len(builder.particle_q)
    style3d.add_cloth_mesh(
        builder,
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        vertices=vertices.tolist(),
        indices=np.asarray(faces, dtype=np.int32).reshape(-1).tolist(),
        density=config.cloth_density,
        tri_aniso_ke=wp.vec3(config.cloth_stretch, config.cloth_stretch, config.cloth_stretch),
        edge_aniso_ke=wp.vec3(config.cloth_bend),
        tri_kd=1.0,
        edge_kd=0.01,
        particle_radius=0.006,
        label="microduck_corner_cloth",
    )
    for row, col in ((0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)):
        index = start + row * n + col
        builder.particle_flags[index] &= ~int(newton.ParticleFlags.ACTIVE)
        builder.particle_mass[index] = 0.0
        x, y = xs[row], ys[col]
        _box(builder, (x, y, 1.15), (0.045, 0.045, 0.045), (0.20, 0.24, 0.27), "cloth_corner_mount")
        _box(builder, (x, y, 0.575), (0.018, 0.018, 0.575), (0.32, 0.38, 0.39), "cloth_support_post")
    return {"kind": "cloth", "config": asdict(config), "support": "sagging_cloth_fixed_at_four_corners",
            "particle_range": [start, start + n * n]}


def add_trampoline(builder, config):
    """A circular elastic membrane with a pinned perimeter and open space underneath."""
    from newton.solvers import style3d

    n = config.trampoline_resolution
    if isinstance(n, bool) or not isinstance(n, int) or n < 3:
        raise ValueError("trampoline_resolution must be an integer >= 3")
    for name in ("length", "width", "height", "density", "stretch"):
        value = getattr(config, f"trampoline_{name}")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"trampoline_{name} must be finite and positive")
    if not math.isfinite(config.trampoline_damping) or config.trampoline_damping < 0:
        raise ValueError("trampoline_damping must be finite and nonnegative")
    provision_pad_contacts(builder)
    newton.solvers.SolverStyle3D.register_custom_attributes(builder)
    radius = min(config.trampoline_length, config.trampoline_width) / 2
    z = config.trampoline_height
    xx, yy = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n), indexing="ij")
    # Map the square grid onto a disk without duplicated center vertices or collapsed faces.
    distance = np.hypot(xx, yy)
    scale = np.divide(radius * np.maximum(np.abs(xx), np.abs(yy)), distance,
                      out=np.zeros_like(distance), where=distance > 0)
    xx, yy = xx * scale, yy * scale
    vertices = np.stack((xx, yy, np.full_like(xx, z)), axis=-1).reshape(-1, 3)
    faces = []
    for row in range(n - 1):
        for col in range(n - 1):
            a = row * n + col
            faces.extend(((a, a + n, a + n + 1), (a, a + n + 1, a + 1)))
    start = len(builder.particle_q)
    style3d.add_cloth_mesh(
        builder, pos=wp.vec3(0.0), rot=wp.quat_identity(), vel=wp.vec3(0.0),
        vertices=vertices.tolist(), indices=np.asarray(faces).ravel().tolist(),
        density=config.trampoline_density, tri_aniso_ke=wp.vec3(config.trampoline_stretch),
        edge_aniso_ke=wp.vec3(2.0e-4), tri_kd=config.trampoline_damping, edge_kd=0.001,
        particle_radius=0.003, label="microduck_trampoline",
    )
    for row in range(n):
        for col in range(n):
            if row in (0, n - 1) or col in (0, n - 1):
                index = start + row * n + col
                builder.particle_flags[index] &= ~int(newton.ParticleFlags.ACTIVE)
                builder.particle_mass[index] = 0.0
    frame_radius = radius + 0.025
    # The same overlapping tube segments provide both the visible ring and rigid collision.
    segments = 48
    for segment in range(segments):
        angles = np.array([segment, segment + 1]) * math.tau / segments
        a, b = [wp.vec3(frame_radius * math.cos(angle), frame_radius * math.sin(angle), z) for angle in angles]
        delta = b - a
        builder.add_shape_capsule(
            -1, radius=0.025, half_height=float(wp.length(delta)) / 2,
            xform=wp.transform((a + b) * 0.5, wp.quat_between_vectors(wp.vec3(0, 0, 1), wp.normalize(delta))),
            color=(0.06, 0.55, 0.65), label="trampoline_frame",
        )
    for angle in np.arange(4) * math.pi / 2 + math.pi / 4:
        _box(builder, (frame_radius * math.cos(angle), frame_radius * math.sin(angle), z / 2),
             (0.018, 0.018, z / 2), (0.22, 0.25, 0.28), "trampoline_leg")
    _box(builder, (0, 0, -0.025), (radius + 0.35, radius + 0.35, 0.025),
         (0.42, 0.46, 0.43), "trampoline_floor")
    return {"kind": "trampoline", "config": asdict(config), "support": "elastic_cloth_fixed_perimeter",
            "shape": "circle", "radius": radius,
            "particle_range": [start, start + n * n]}


def add_elastic_bridge(builder, config):
    """A compliant walking strip clamped at its two short ends, with free long edges."""
    from newton.solvers import style3d

    for name in ("length", "width", "height", "density", "stretch", "approach_length", "approach_width", "spawn_distance"):
        value = getattr(config, f"elastic_bridge_{name}")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"elastic_bridge_{name} must be finite and positive")
    for name in ("sag", "bend", "damping"):
        value = getattr(config, f"elastic_bridge_{name}")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"elastic_bridge_{name} must be finite and nonnegative")
    for name in ("cells_x", "cells_y"):
        value = getattr(config, f"elastic_bridge_{name}")
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError(f"elastic_bridge_{name} must be an integer >= 2")
    if config.elastic_bridge_sag >= config.elastic_bridge_height:
        raise ValueError("Elastic bridge sag must leave space above the floor")
    if min(config.elastic_bridge_length, config.elastic_bridge_width) <= 0.08:
        raise ValueError("Elastic bridge dimensions must exceed the training edge margins")
    if config.elastic_bridge_approach_width < config.elastic_bridge_width:
        raise ValueError("Approach must be at least as wide as the bridge")
    if not 0.1 < config.elastic_bridge_spawn_distance < config.elastic_bridge_approach_length - 0.1:
        raise ValueError("Spawn must leave 10 cm clearance from both approach ends")
    provision_pad_contacts(builder)
    newton.solvers.SolverStyle3D.register_custom_attributes(builder)
    hx, hy, height = config.elastic_bridge_length / 2, config.elastic_bridge_width / 2, config.elastic_bridge_height
    nx, ny = config.elastic_bridge_cells_x + 1, config.elastic_bridge_cells_y + 1
    xx, yy = np.meshgrid(np.linspace(-hx, hx, nx), np.linspace(-hy, hy, ny), indexing="ij")
    zz = height - config.elastic_bridge_sag * (1 - (xx / hx) ** 2)
    vertices = np.stack((xx, yy, zz), axis=-1).reshape(-1, 3)
    faces = []
    for row in range(nx - 1):
        for col in range(ny - 1):
            a = row * ny + col
            faces.extend(((a, a + ny, a + ny + 1), (a, a + ny + 1, a + 1)))
    start = len(builder.particle_q)
    style3d.add_cloth_mesh(
        builder, pos=wp.vec3(0.0), rot=wp.quat_identity(), vel=wp.vec3(0.0),
        vertices=vertices.tolist(), indices=np.asarray(faces).ravel().tolist(),
        density=config.elastic_bridge_density, tri_aniso_ke=wp.vec3(config.elastic_bridge_stretch),
        edge_aniso_ke=wp.vec3(config.elastic_bridge_bend), tri_kd=config.elastic_bridge_damping,
        edge_kd=0.001, particle_radius=0.003, label="microduck_elastic_bridge",
    )
    # Pin every vertex across each end; the free sides must remain able to twist and sag.
    for row in (0, nx - 1):
        for col in range(ny):
            index = start + row * ny + col
            builder.particle_flags[index] &= ~int(newton.ParticleFlags.ACTIVE)
            builder.particle_mass[index] = 0.0
    for sign in (-1, 1):
        anchor_half_width = config.elastic_bridge_approach_width / 2 if sign == -1 else hy + 0.07
        _box(builder, (sign * (hx + 0.04), 0, height - 0.025), (0.04, anchor_half_width, 0.025),
             (0.10, 0.46, 0.53), "elastic_bridge_anchor")
        for side in (-1, 1):
            _box(builder, (sign * (hx + 0.04), side * (hy + 0.04), (height - 0.05) / 2),
                 (0.025, 0.025, (height - 0.05) / 2), (0.25, 0.29, 0.31), "elastic_bridge_post")
    approach = config.elastic_bridge_approach_length
    # Meet the inlet beam exactly: avoid a step, gap, or overlapping coplanar contact surfaces.
    _box(builder, (-hx - (approach + 0.08) / 2, 0, height / 2),
         ((approach - 0.08) / 2, config.elastic_bridge_approach_width / 2, height / 2),
         (0.58, 0.62, 0.60), "elastic_bridge_approach")
    # The floor catches falls; no rigid shape extends into the flexible span.
    _box(builder, (-approach / 2, 0, -0.025),
         (hx + approach / 2 + 0.35, max(hy, config.elastic_bridge_approach_width / 2) + 0.4, 0.025),
         (0.42, 0.46, 0.43), "elastic_bridge_floor")
    return {"kind": "elastic_bridge", "config": asdict(config), "support": "elastic_strip_fixed_short_ends",
            "particle_range": [start, start + nx * ny]}


class SceneTerrain:
    """Terrain interface for optional scenes without the soft-pad guard."""

    def __init__(self, env, kind, metadata):
        self.env, self.kind = env, kind
        self.metadata = metadata
        self.origins = torch.zeros(env.num_envs, 3, device=env.device)
        self.spawn_on_pad = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.cfg = {"kind": kind, **metadata}
        env.cfg["terrain"] = self.cfg

    def finalize(self):
        if self.kind in ("trampoline", "cloth", "elastic_bridge"):
            model = self.env.model
            # Resolve after finalize: affine proxy insertion can reorder particles.
            indices = np.unique(model.tri_indices.numpy())
            self.cloth_indices = torch.as_tensor(indices, device=self.env.device, dtype=torch.long)
            self.cloth_worlds = wp.to_torch(model.particle_world)[self.cloth_indices].long().clamp_min(0)
            if self.kind in ("cloth", "elastic_bridge"):
                # Reuse the soft-contact sensor's topology mask; exclude rigid affine proxies.
                mask = torch.zeros(model.particle_count, dtype=torch.int32, device=self.env.device)
                mask[self.cloth_indices] = 1
                self.pad_particle = wp.from_torch(mask)
                topology = wp.to_torch(model.tri_indices).long()
                worlds = wp.to_torch(model.particle_world)[topology[:, 0]].clamp_min(0)
                self.surface_triangles = torch.stack([topology[worlds == w] for w in range(self.env.num_envs)])
            return
        if self.kind not in ("sand", "pool"):
            return
        from style3d_sim import configure_granular

        model = self.env.model
        # Finalization appends and reorders affine proxies, whose positive radii
        # do not make them grains. Resolve membership from finalized topology.
        grains = np.ones(model.particle_count, dtype=bool)
        for name in ("synreal_affine_body_indices", "tri_indices", "tet_indices", "spring_indices", "edge_indices"):
            topology = getattr(model, name, None)
            if topology is not None:
                occupied = topology.numpy().ravel()
                grains[occupied[occupied >= 0]] = False
        indices = np.flatnonzero(grains)
        worlds = model.particle_world.numpy()[indices]
        if self.env.num_envs == 1:
            # Standalone templates use world -1; replicated training uses world 0.
            worlds = np.maximum(worlds, 0)
        expected = self.metadata["grain_count_per_world" if self.kind == "sand" else "fluid_count_per_world"]
        counts = np.array([np.count_nonzero(worlds == world) for world in range(self.env.num_envs)])
        if len(indices) != expected * self.env.num_envs or not np.all(counts == expected):
            raise ValueError(f"{self.kind} particle topology mismatch: expected {expected} per world, got {counts.tolist()}")
        config = SceneConfig(**self.cfg["config"])
        if self.kind == "pool":
            from style3d_sim import configure_fluid

            spacing = config.pool_spacing
            lower = np.array([-config.pool_length / 2, -config.pool_width / 2, 0.0])
            upper = np.array([config.pool_length / 2, config.pool_width / 2, config.pool_wall_height])
            cells = np.ceil((upper - lower) / spacing).astype(int)
            axes = [(np.arange(-2, n + 2) + 0.5) * spacing + low for n, low in zip(cells, lower, strict=True)]
            samples = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
            # An open pool has density support outside the floor and walls, but no lid.
            outside = (samples[:, 2] < 0) | np.any((samples[:, :2] < lower[:2]) | (samples[:, :2] > upper[:2]), axis=1)
            samples = samples[outside & (samples[:, 2] < upper[2])]
            configure_fluid(model, indices, smoothing_radius=2 * spacing, rest_density=config.pool_density,
                            bulk_modulus=config.pool_bulk_modulus, viscosity=config.pool_viscosity,
                            boundary_positions=samples,
                            boundary_volumes=np.full(len(samples), self.metadata["fluid_rest_volume"]))
            self.cfg["fluid_particle_count"] = len(indices)
            return
        configure_granular(
            model,
            indices,
            contact_stiffness=config.sand_stiffness,
            friction=config.sand_friction,
            friction_damping=40.0,
            cohesion=config.sand_cohesion,
        )
        self.cfg["granular_particle_count"] = len(indices)

    def initialize_rest_state(self):
        if self.kind != "elastic_bridge" or self.env.solver is None:
            return
        rest, equilibrium, report = settled_bridge_state(SceneConfig(**self.cfg["config"]),
                                                        self.env.physics, self.env.device)
        self.reset_particle_q = wp.to_torch(self.env.model.particle_q).clone()
        for world in range(self.env.num_envs):
            selected = self.cloth_indices[self.cloth_worlds == world]
            # Finalization may insert rigid proxies; never assume an authored particle offset survives.
            if not np.allclose(self.reset_particle_q[selected].cpu().numpy(), rest, rtol=0, atol=1e-6):
                raise ValueError("Bridge equilibrium particle ordering does not match the replicated template")
            self.reset_particle_q[selected] = torch.as_tensor(equilibrium, device=self.env.device)
        self.cfg["initial_equilibrium"] = dict(report)
        self.bridge_sleeping = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for state in self.env.states:
            wp.to_torch(state.particle_q)[self.cloth_indices] = self.reset_particle_q[self.cloth_indices]
            wp.to_torch(state.particle_qd)[self.cloth_indices] = 0
        self.env._sync()

    def wake_bridge_contacts(self, contacts):
        """Wake per world before solving a robot contact, including its contact-generation margin."""
        if not hasattr(self, "bridge_sleeping") or not self.bridge_sleeping.any():
            return
        count = int(wp.to_torch(contacts.soft_contact_count)[0])
        indices = wp.to_torch(contacts.soft_contact_indices)[:count].long()
        shapes = wp.to_torch(contacts.soft_contact_shape)[:count].long()
        valid = (shapes >= 0) & (indices[:, 0] >= 0)
        particles = indices[valid, 0]
        bodies = wp.to_torch(self.env.model.shape_body)[shapes[valid]]
        cloth = wp.to_torch(self.pad_particle)[particles] != 0
        worlds = wp.to_torch(self.env.model.particle_world)[particles[cloth & (bodies >= 0)]].long().clamp_min(0)
        self.bridge_sleeping[worlds] = False
        # Manual perturbations must wake the membrane too, even with no robot contact.
        state = self.env.states[0]
        velocity = wp.to_torch(state.particle_qd)[self.cloth_indices]
        force = wp.to_torch(state.particle_f)[self.cloth_indices]
        disturbed = (velocity.abs().amax(-1) > 1.0e-5) | (force.abs().amax(-1) > 1.0e-5)
        self.bridge_sleeping[self.cloth_worlds[disturbed]] = False

    def hold_sleeping_bridges(self):
        if not hasattr(self, "bridge_sleeping"):
            return
        selected = self.cloth_indices[self.bridge_sleeping[self.cloth_worlds]]
        # Rigid solver residuals must not re-excite an unloaded, equilibrated membrane.
        # Once touched, it stays fully dynamic until the next explicit/episode reset.
        for state in self.env.states:
            wp.to_torch(state.particle_q)[selected] = self.reset_particle_q[selected]
            wp.to_torch(state.particle_qd)[selected] = 0

    def reset(self, ids):
        self.origins[ids] = 0
        if hasattr(self, "bridge_sleeping"):
            self.bridge_sleeping[ids] = True
        if self.kind in ("trampoline", "cloth", "elastic_bridge"):
            selected = self.cloth_indices[torch.isin(self.cloth_worlds, ids)]
            for state in self.env.states:
                reset_q = getattr(self, "reset_particle_q", wp.to_torch(self.env.model.particle_q))
                wp.to_torch(state.particle_q)[selected] = reset_q[selected]
                wp.to_torch(state.particle_qd)[selected] = 0
            self.env._sync()
        if self.kind == "pool":
            model = self.env.model
            indices = wp.to_torch(model.synreal_fluid_particle_indices).long()
            worlds = wp.to_torch(model.particle_world)[indices].long().clamp_min(0)
            selected = indices[torch.isin(worlds, ids)]
            # Only reset the selected worlds' water; affine proxy state belongs to FK.
            for state in self.env.states:
                wp.to_torch(state.particle_q)[selected] = wp.to_torch(model.particle_q)[selected]
                wp.to_torch(state.particle_qd)[selected] = 0
            self.env._sync()

    def adjust_spawn(self, ids):
        spawn = self.env.task_logic.spawn
        if self.kind == "trampoline":
            spawn[ids, :2] = 0
            self.origins[ids, 2] = self.cfg["config"]["trampoline_height"]
            spawn[ids, 2] += self.origins[ids, 2] + 0.01
        elif self.kind == "hammock":
            spawn[ids, 0] = 0
            spawn[ids, 1] = 0
            self.origins[ids, 2] = 1.15 - self.cfg["config"]["hammock_sag"]
            spawn[ids, 2] += self.origins[ids, 2]
        elif self.kind in ("cloth", "elastic_bridge"):
            spawn[ids, :2] = 0
            config = self.cfg["config"]
            if self.kind == "elastic_bridge":
                # Reset restores the relaxed membrane before sampling its actual support height.
                points = torch.zeros(self.env.num_envs, 1, 3, device=self.env.device)
                self.origins[ids, 2] = self.surface_heights(points)[ids, 0]
            else:
                self.origins[ids, 2] = 1.15 - config["cloth_sag"]
            spawn[ids, 2] += self.origins[ids, 2]
        elif self.kind == "sand":
            spawn[ids, :2] = 0
            self.origins[ids, 2] = self.cfg["config"]["sand_depth"]
            spawn[ids, 2] += self.origins[ids, 2]
        elif self.kind == "pool":
            spawn[ids, :2] = 0
            spawn[ids, 2] += self.cfg["config"]["pool_depth"]
            # Water is not a rigid support surface for terrain contact observations.
        else:
            spawn[ids, 0] = -0.42
            spawn[ids, 1] = 0

    def heights(self, positions, quaternions=None):
        support = self.origins[:, None, 2]
        if self.kind == "trampoline":
            support = self.surface_heights(positions)
        if self.kind == "hammock":
            config = self.cfg["config"]
            x = positions[..., 0]
            support = 1.15 - config["hammock_sag"] * (1 - (2 * x / config["hammock_length"]) ** 2)
        elif self.kind in ("cloth", "elastic_bridge"):
            support = self.surface_heights(positions)
        elif self.kind == "steps":
            config = self.cfg["config"]
            x, y = positions[..., 0], positions[..., 1]
            index = torch.floor((x + config["step_run"] / 2) / config["step_run"]).long() + 1
            support = index.clamp(0, config["step_count"]) * config["step_rise"]
            support = torch.where(y.abs() <= config["step_width"] / 2, support, 0.0)
        return (positions[..., 2] - support).clamp(0.0, 1.0)

    def surface_heights(self, positions):
        """Query the deformed triangles, including the edges of a small foot collider."""
        points = positions.reshape(self.env.num_envs, -1, 3)
        if self.kind in ("cloth", "elastic_bridge"):
            # Batched queries avoid 64 Python/CUDA launch sequences per observation.
            triangles = wp.to_torch(self.env.states[0].particle_q)[self.surface_triangles]
            a, b, c = triangles.unbind(2)
            u, v = b[..., :2] - a[..., :2], c[..., :2] - a[..., :2]
            delta = points[:, :, None, :2] - a[:, None, :, :2]
            det = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
            valid = det.abs() > 1e-12
            denom = torch.where(valid, det, 1.0)[:, None]
            s = (delta[..., 0] * v[:, None, :, 1] - delta[..., 1] * v[:, None, :, 0]) / denom
            t = (u[:, None, :, 0] * delta[..., 1] - u[:, None, :, 1] * delta[..., 0]) / denom
            inside = valid[:, None] & (s >= -1e-5) & (t >= -1e-5) & (s + t <= 1 + 1e-5)
            height = a[:, None, :, 2] + s * (b - a)[:, None, :, 2] + t * (c - a)[:, None, :, 2]
            result = torch.where(inside, height, 0.0).amax(-1).reshape(positions.shape[:-1])
            if self.kind == "elastic_bridge":
                config = self.cfg["config"]
                x, y = positions[..., 0], positions[..., 1]
                inlet = -config["elastic_bridge_length"] / 2
                on_approach = ((x >= inlet - config["elastic_bridge_approach_length"]) & (x <= inlet)
                               & (y.abs() <= config["elastic_bridge_approach_width"] / 2))
                result = torch.where(on_approach, config["elastic_bridge_height"], result)
            return result
        result = torch.zeros(points.shape[:2], device=points.device)
        topology = wp.to_torch(self.env.model.tri_indices).long()
        worlds = wp.to_torch(self.env.model.particle_world)[topology[:, 0]].clamp_min(0)
        particles = wp.to_torch(self.env.states[0].particle_q)
        # Chunk queries by world to avoid a cross-world point/triangle product.
        for world in range(self.env.num_envs):
            tri = particles[topology[worlds == world]]
            a, b, c = tri.unbind(1)
            u, v = b[:, :2] - a[:, :2], c[:, :2] - a[:, :2]
            delta = points[world, :, None, :2] - a[None, :, :2]
            det = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
            valid = det.abs() > 1e-12
            denom = torch.where(valid, det, torch.ones_like(det))
            s = (delta[..., 0] * v[:, 1] - delta[..., 1] * v[:, 0]) / denom
            t = (u[:, 0] * delta[..., 1] - u[:, 1] * delta[..., 0]) / denom
            inside = valid & (s >= -1e-5) & (t >= -1e-5) & (s + t <= 1 + 1e-5)
            height = a[:, 2] + s * (b[:, 2] - a[:, 2]) + t * (c[:, 2] - a[:, 2])
            result[world] = torch.where(inside, height, 0.0).amax(-1)
        return result.reshape(positions.shape[:-1])

    def update_support(self, pose):
        if self.kind in ("trampoline", "cloth", "elastic_bridge"):
            self.origins[:, 2] = self.surface_heights(pose[:, None, :3]).squeeze(1)
        if self.kind == "steps":
            x = pose[:, 0]
            index = torch.floor((x + SCENE_CONFIG.step_run / 2) / SCENE_CONFIG.step_run).long() + 1
            height = index.clamp(0, SCENE_CONFIG.step_count) * SCENE_CONFIG.step_rise
            self.origins[:, 2] = torch.where(x < -SCENE_CONFIG.step_run / 2, 0.0, height)

    def side_out_of_bounds(self, positions):
        """Follow the wider rigid approach and the narrower flexible span without an inlet timeout."""
        config = self.cfg["config"]
        width = torch.where(positions[:, 0] < -config["elastic_bridge_length"] / 2,
                            config["elastic_bridge_approach_width"], config["elastic_bridge_width"])
        return positions[:, 1].abs() > width / 2 - 0.04

    def out_of_bounds(self, positions):
        if self.kind == "elastic_bridge":
            config = self.cfg["config"]
            hx = config["elastic_bridge_length"] / 2
            return ((positions[:, 0] < -hx - config["elastic_bridge_approach_length"] + 0.04)
                    | (positions[:, 0] > hx - 0.04) | self.side_out_of_bounds(positions))
        if self.kind == "trampoline":
            config = self.cfg["config"]
            radius = min(config["trampoline_length"], config["trampoline_width"]) / 2
            return torch.linalg.vector_norm(positions[:, :2], dim=-1) > radius - 0.04
        if self.kind == "cloth":
            config = self.cfg["config"]
            prefix = self.kind
            return ((positions[:, 0].abs() > config[f"{prefix}_length"] / 2 - 0.04)
                    | (positions[:, 1].abs() > config[f"{prefix}_width"] / 2 - 0.04))
        if self.kind == "pool":
            config = self.cfg["config"]
            return ((positions[:, 0].abs() > config["pool_length"] / 2 + config["pool_wall_thickness"])
                    | (positions[:, 1].abs() > config["pool_width"] / 2 + config["pool_wall_thickness"]))
        return (positions[:, :2].abs() > 2.5).any(-1)

    def check_state(self):
        return None

    def configure_viewer(self, viewer):
        if self.kind == "elastic_bridge" and self.env.num_envs == 1:
            viewer.set_camera(pos=wp.vec3(2.0, -2.8, 1.9), pitch=-22.0, yaw=132.0)
        if self.kind == "trampoline" and self.env.num_envs == 1:
            viewer.set_camera(pos=wp.vec3(1.1, -1.4, 1.0), pitch=-20.0, yaw=128.0)
        if self.kind == "pool" and self.env.num_envs == 1:
            viewer.set_camera(pos=wp.vec3(0.8, -1.0, 0.85), pitch=-30.0, yaw=129.0)
        if self.kind == "sand" and self.env.num_envs == 1:
            viewer.set_camera(pos=wp.vec3(0.43, -0.55, 0.43), pitch=-30.0, yaw=129.0)

    def render_particles(self, viewer, display_state):
        if self.kind == "pool":
            model = self.env.model
            indices = model.synreal_fluid_particle_indices
            if not hasattr(self, "_water_points"):
                self._water_points = wp.empty(len(indices), dtype=wp.vec3, device=model.device)
                self._water_colors = wp.full(len(indices), wp.vec3(0.04, 0.40, 0.72), device=model.device)
            wp.launch(gather_surface, dim=len(indices), inputs=[display_state.particle_q, indices],
                      outputs=[self._water_points], device=model.device)
            if model.device.is_cuda and (hasattr(viewer, "objects") or hasattr(viewer, "stage")):
                from style3d_sim import ParticleSurface, log_surface_mesh

                spacing = self.cfg["config"]["pool_spacing"]
                if not hasattr(self, "_water_surface"):
                    self._water_surface = ParticleSurface(spacing, self._water_points, voxel_size=spacing / 2)
                # The render mesh follows the fluid; it never becomes a collision lid.
                mesh = self._water_surface.rebuild(self._water_points)
                config = self.cfg["config"]
                opacity = config.get("pool_water_opacity", SCENE_CONFIG.pool_water_opacity)
                if not 0 <= opacity <= 1:
                    raise ValueError("pool_water_opacity must be between 0 and 1")
                log_surface_mesh(
                    viewer, "/scene/water", mesh,
                    color=config.get("pool_water_color", SCENE_CONFIG.pool_water_color),
                    roughness=config.get("pool_water_roughness", SCENE_CONFIG.pool_water_roughness),
                )
                name = viewer._qualify("/scene/water")
                if hasattr(viewer, "objects"):
                    # log_surface_mesh recreates the object at the end of the draw
                    # list each frame, so opaque geometry is already available.
                    viewer.objects[name] = _TransparentWaterMesh(viewer.objects[name], viewer.renderer.gl, opacity)
                elif hasattr(viewer, "_meshes"):
                    viewer._meshes[name].CreateDisplayOpacityAttr([opacity])
                return
            viewer.log_points("/scene/water", self._water_points, 0.5 * self.cfg["config"]["pool_spacing"],
                              self._water_colors)
            return
        if self.kind != "sand":
            return
        model = self.env.model
        indices = model.synreal_granular_particle_indices
        if not hasattr(self, "_sand_points"):
            self._sand_points = wp.empty(len(indices), dtype=wp.vec3, device=model.device)
            self._sand_radii = wp.array(
                model.particle_radius.numpy()[indices.numpy()], dtype=wp.float32, device=model.device
            )
            self._sand_colors = wp.full(len(indices), wp.vec3(0.76, 0.60, 0.32), device=model.device)
        # The generic particle layer is hidden by default and also contains robot
        # proxies. Render only registered grains, using the display-only offsets.
        wp.launch(gather_surface, dim=len(indices), inputs=[display_state.particle_q, indices],
                  outputs=[self._sand_points], device=model.device)
        viewer.log_points("/scene/sand", self._sand_points, self._sand_radii, self._sand_colors)

    def display_state(self, viewer, state):
        if viewer.world_offsets is None or not self.env.model.particle_count:
            return state

        if not hasattr(self, "_display_points"):
            self._display_points = wp.empty_like(state.particle_q)
        wp.launch(
            _display_particle_positions,
            dim=self.env.model.particle_count,
            inputs=[state.particle_q, self.env.model.particle_world, viewer.world_offsets],
            outputs=[self._display_points],
            device=self.env.model.device,
        )
        display = copy(state)
        display.particle_q = self._display_points
        return display


# Volumetric pad geometry


@dataclass(frozen=True)
class SoftPadConfig:
    """Compressible elastic approximation; these parameters are not calibrated foam data."""

    # Keep the validated 5 cm horizontal spacing and three layers through the
    # thickness, but limit deformable geometry to a local patch of the rigid floor.
    size: tuple[float, float, float] = (1.2, 0.8, 0.06)
    cells: tuple[int, int, int] | None = None
    base_z: float = 0.0015
    density: float = 80.0
    shear_modulus: float = 2.0e3
    lame_lambda: float = 2.0e3
    damping: float = 5.0
    particle_radius: float = 0.0015
    fix_bottom: bool = True
    # Historical checkpoints omitted this field and used 50 kN/m. Current
    # experiment defaults are explicitly selected by env.py's PAD_CONFIG.
    contact_stiffness: float = 5.0e4
    # Explicit cells override density for legacy checkpoints and diagnostic meshes.
    cells_per_meter: float | tuple[float, float, float] | None = None

    def __post_init__(self):
        if len(self.size) != 3 or not all(np.isfinite(v) and v > 0 for v in self.size):
            raise ValueError("pad size must contain three finite positive lengths")
        if self.cells_per_meter is not None:
            resolution = self.cells_per_meter
            if isinstance(resolution, Real):
                resolution = (resolution,) * 3
            if len(resolution) != 3 or not all(
                isinstance(v, Real) and not isinstance(v, bool) and math.isfinite(v) and v > 0 for v in resolution
            ):
                raise ValueError("pad cells_per_meter must be positive and finite: one value or three axis values")
            object.__setattr__(self, "cells_per_meter", tuple(float(v) for v in resolution))
        elif self.cells is None:
            # Keep the historical default topology when old metadata has no resolution fields.
            object.__setattr__(self, "cells", (24, 16, 3))
        if self.cells is not None:
            if len(self.cells) != 3 or not all(isinstance(v, int) and not isinstance(v, bool) and v >= 1 for v in self.cells):
                raise ValueError("pad cells must contain three positive integers")
            object.__setattr__(self, "cells", tuple(self.cells))
        _ = self.resolved_cells  # Validate the size/density combination before building geometry.
        for value in (self.density, self.shear_modulus, self.lame_lambda, self.particle_radius, self.contact_stiffness):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("pad density, stiffnesses and contact radius must be finite and positive")
        if not np.isfinite(self.base_z):
            raise ValueError("pad base height must be finite")
        if not np.isfinite(self.damping) or self.damping < 0:
            raise ValueError("pad damping must be finite and nonnegative")

    @property
    def resolved_cells(self) -> tuple[int, int, int]:
        """Round up to meet the requested resolution, retaining at least one cell per axis."""
        if self.cells is not None:
            return self.cells
        counts = []
        for length, density in zip(self.size, self.cells_per_meter, strict=True):
            count = length * density
            if not math.isfinite(count):
                raise ValueError("pad size times cells_per_meter must be finite")
            # Decimal dimensions such as 1.1 * 100 can lie just above an integer in float64.
            nearest = round(count)
            if math.isclose(count, nearest, rel_tol=1.0e-12, abs_tol=1.0e-12):
                count = nearest
            counts.append(max(1, math.ceil(count)))
        return tuple(counts)

    @property
    def embedded(self) -> bool:
        return self.base_z < 0.0

    @property
    def floor_half_extents(self) -> tuple[float, float]:
        # Cover the 2.5 m episode boundary, including feet extending beyond the root.
        return tuple(max(3.0, length / 2 + 0.5) for length in self.size[:2])

    @property
    def cavity_half_extents(self) -> tuple[float, float]:
        # Leave room for collision skin so an unstressed pad is not precompressed
        # sideways by the surrounding rigid walls.
        return tuple(length / 2 + 2 * self.particle_radius for length in self.size[:2])

    @property
    def cavity_bottom_z(self) -> float:
        return self.base_z - 2 * self.particle_radius

    @property
    def top_z(self) -> float:
        """Return the undeformed upper surface height in world coordinates."""
        return self.base_z + self.size[2]

    @classmethod
    def from_checkpoint(cls, terrain: dict) -> SoftPadConfig:
        """Restore recorded contact compliance, including the older separate metadata layout."""
        values = dict(terrain["pad"])
        values.setdefault("contact_stiffness", terrain.get("soft_contact", {}).get("ke", 5.0e4))
        return cls(**values)


def tetrahedral_box_mesh(size: tuple[float, float, float], cells: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Split every cell along the same body diagonal, preserving shared faces between cells."""
    SoftPadConfig(size=size, cells=cells)

    nx, ny, nz = cells
    xs = np.linspace(-size[0] / 2.0, size[0] / 2.0, nx + 1, dtype=np.float32)
    ys = np.linspace(-size[1] / 2.0, size[1] / 2.0, ny + 1, dtype=np.float32)
    zs = np.linspace(-size[2] / 2.0, size[2] / 2.0, nz + 1, dtype=np.float32)
    vertices = np.asarray([(x, y, z) for z in zs for y in ys for x in xs], dtype=np.float32)

    def index(x: int, y: int, z: int) -> int:
        return z * (ny + 1) * (nx + 1) + y * (nx + 1) + x

    tetrahedra: list[tuple[int, int, int, int]] = []
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                nodes = (
                    index(x, y, z),
                    index(x + 1, y, z),
                    index(x + 1, y + 1, z),
                    index(x, y + 1, z),
                    index(x, y, z + 1),
                    index(x + 1, y, z + 1),
                    index(x + 1, y + 1, z + 1),
                    index(x, y + 1, z + 1),
                )
                pattern = ((0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6))
                tetrahedra.extend(tuple(nodes[i] for i in tet) for tet in pattern)

    tetrahedra_array = np.asarray(tetrahedra, dtype=np.int32)
    points = vertices[tetrahedra_array]
    six_volumes = np.einsum(
        "ij,ij->i",
        points[:, 1] - points[:, 0],
        np.cross(points[:, 2] - points[:, 0], points[:, 3] - points[:, 0]),
    )
    # The solver requires consistently oriented, non-degenerate tetrahedra.
    if np.any(six_volumes <= 0.0):
        raise ValueError("soft pad mesh contains a nonpositive tetrahedron")
    return vertices, tetrahedra_array


def add_soft_pad(builder: newton.ModelBuilder, config: SoftPadConfig | None = None) -> dict:
    """Add one continuous volumetric pad and optionally pin its bottom face.

    The bottom pinning models a sponge bonded to a rigid base. It avoids adding a
    second deformable body while retaining compression and lateral motion on the top.
    """
    config = SoftPadConfig() if config is None else config
    vertices, tetrahedra = tetrahedral_box_mesh(config.size, config.resolved_cells)
    particle_start = len(builder.particle_q)
    triangle_start = len(builder.tri_indices)
    tet_start = len(builder.tet_indices)
    builder.add_soft_mesh(
        pos=wp.vec3(0.0, 0.0, config.base_z + config.size[2] / 2.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0),
        vertices=vertices.tolist(),
        indices=tetrahedra.ravel().tolist(),
        density=config.density,
        k_mu=config.shear_modulus,
        k_lambda=config.lame_lambda,
        k_damp=config.damping,
        particle_radius=config.particle_radius,
        add_surface_mesh_edges=False,
        validate_mesh=True,
        label="microduck_sponge_pad",
    )
    particle_stop = len(builder.particle_q)

    fixed_indices: list[int] = []
    if config.fix_bottom:
        z_min = config.base_z + 1.0e-6
        for index in range(particle_start, particle_stop):
            if builder.particle_q[index][2] <= z_min:
                fixed_indices.append(index)
                builder.particle_mass[index] = 0.0
                builder.particle_flags[index] &= ~newton.ParticleFlags.ACTIVE

    if config.fix_bottom and not fixed_indices:
        raise ValueError("soft pad bottom pinning selected no particles")

    return {
        "particle_range": (particle_start, particle_stop),
        "triangle_range": (triangle_start, len(builder.tri_indices)),
        "tetrahedron_range": (tet_start, len(builder.tet_indices)),
        "fixed_particles": np.asarray(fixed_indices, dtype=np.int32),
        "top_z": config.top_z,
        "config": config,
        "vertices": vertices,
        "tetrahedra": tetrahedra,
    }


# Pad support contacts


@wp.kernel(enable_backward=False)
def lowest_mesh_vertices(
    shapes: wp.array[int],
    starts: wp.array[int],
    vertices: wp.array[wp.vec3],
    shape_body: wp.array[int],
    shape_pose: wp.array[wp.transform],
    body_pose: wp.array[wp.transform],
    world_points: wp.array[wp.vec3],
    local_points: wp.array[wp.vec3],
):
    item = wp.tid()
    shape = shapes[item]
    pose = wp.transform_multiply(body_pose[shape_body[shape]], shape_pose[shape])
    lowest = wp.vec3(0.0, 0.0, 1.0e20)
    local = wp.vec3(0.0)
    for vertex in range(starts[item], starts[item + 1]):
        point = wp.transform_point(pose, vertices[vertex])
        if point[2] < lowest[2]:
            lowest = point
            local = wp.transform_point(shape_pose[shape], vertices[vertex])
    world_points[item] = lowest
    local_points[item] = local


@wp.func
def top_face_weights(a: wp.vec3, b: wp.vec3, c: wp.vec3, point: wp.vec3):
    ab, ac, ap = b - a, c - a, point - a
    determinant = ab[0] * ac[1] - ab[1] * ac[0]
    if wp.abs(determinant) < 1.0e-12:
        return wp.vec3(-1.0)
    u = (ap[0] * ac[1] - ap[1] * ac[0]) / determinant
    v = (ab[0] * ap[1] - ab[1] * ap[0]) / determinant
    return wp.vec3(1.0 - u - v, u, v)


@wp.kernel(enable_backward=False)
def query_top_faces(
    shapes: wp.array[int],
    shape_world: wp.array[int],
    points: wp.array[wp.vec3],
    particles: wp.array[wp.vec3],
    faces: wp.array2d[wp.vec3i],
    margin: float,
    bottom: float,
    heights: wp.array2d[float],
):
    item, face = wp.tid()
    heights[item, face] = -1.0e20
    point = points[item]
    # No artificial lid outside the actual top triangles or below the bonded bottom.
    if point[2] < bottom:
        return
    world = wp.max(shape_world[shapes[item]], 0)
    tri = faces[world, face]
    if tri[0] < 0:
        return
    a, b, c = particles[tri[0]], particles[tri[1]], particles[tri[2]]
    weights = top_face_weights(a, b, c, point)
    if wp.min(weights) < -1.0e-6:
        return
    height = weights[0] * a[2] + weights[1] * b[2] + weights[2] * c[2]
    if point[2] - height <= margin:
        heights[item, face] = height


@wp.kernel(enable_backward=False)
def append_support_contacts(
    shapes: wp.array[int],
    shape_world: wp.array[int],
    points: wp.array[wp.vec3],
    local_points: wp.array[wp.vec3],
    particles: wp.array[wp.vec3],
    faces: wp.array2d[wp.vec3i],
    heights: wp.array2d[float],
    capacity: int,
    count: wp.array[int],
    contact_particle: wp.array[int],
    indices: wp.array[wp.vec3i],
    barycentric: wp.array[wp.vec3],
    contact_shape: wp.array[int],
    body_pos: wp.array[wp.vec3],
    body_vel: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
):
    item = wp.tid()
    best_height = float(-1.0e20)  # noqa: UP018 - Warp needs mutable loop accumulators.
    best_face = int(-1)  # noqa: UP018, RUF046 - Warp needs a mutable loop accumulator.
    # Reduce only scalar heights here; expensive geometry queries run in parallel above.
    # Strict comparison selects one face at a shared edge and the upper face after folding.
    for face in range(heights.shape[1]):
        if heights[item, face] > best_height:
            best_height = heights[item, face]
            best_face = face
    if best_face < 0:
        return
    shape = shapes[item]
    world = wp.max(shape_world[shape], 0)
    tri = faces[world, best_face]
    a, b, c = particles[tri[0]], particles[tri[1]], particles[tri[2]]
    weights = top_face_weights(a, b, c, points[item])
    normal = wp.normalize(wp.cross(b - a, c - a))
    if normal[2] > 0.0:
        normal = -normal
    index = wp.atomic_add(count, 0, 1)
    if index < capacity:
        contact_particle[index] = -1
        indices[index] = tri
        barycentric[index] = weights
        contact_shape[index] = shape
        body_pos[index] = local_points[item]
        body_vel[index] = wp.vec3(0.0)
        normals[index] = normal


class PadVertexContacts:
    """Non-differentiable support contacts for mesh colliders above a bonded pad.

    Newton retains responsibility for analytic shapes, pad sides/bottom, and all
    ordinary SDF contacts. A completely below-pad vertex receives no top contact.
    """

    def __init__(self, env):
        self.env = env
        self._last_generation = None
        model = env.model
        flags = model.shape_flags.numpy()
        bodies = model.shape_body.numpy()
        worlds = model.shape_world.numpy()
        has_pad = env.terrain.has_pad.cpu().numpy()
        scales = model.shape_scale.numpy()
        shapes, starts, points = [], [0], []
        for i, source in enumerate(model.shape_source):
            world = int(worlds[i])
            if world < 0:
                if env.num_envs != 1:
                    continue
                # Standalone interactive scenes use -1 for their single world.
                world = 0
            if bodies[i] < 0 or not has_pad[world] or not flags[i] & int(newton.ShapeFlags.COLLIDE_PARTICLES):
                continue
            if source is None or not hasattr(source, "vertices"):
                continue
            vertices = np.asarray(source.vertices) * scales[i]
            shapes.append(i)
            points.extend(vertices)
            starts.append(len(points))
        self.shapes = wp.array(shapes, dtype=int, device=model.device)
        self.starts = wp.array(starts, dtype=int, device=model.device)
        self.vertices = wp.array(np.asarray(points).reshape(-1, 3), dtype=wp.vec3, device=model.device)
        self.world_points = wp.empty(len(shapes), dtype=wp.vec3, device=model.device)
        self.local_points = wp.empty_like(self.world_points)
        self.heights = wp.empty((len(shapes), env.terrain.top_faces.shape[1]), dtype=float, device=model.device)

    def update(self):
        if not len(self.shapes):
            return
        env = self.env
        m, c = env.model, env.contacts
        generation = (id(c), int(c.contact_generation.numpy()[0]))
        if generation == self._last_generation:
            return
        if c.soft_contact_body_pos.requires_grad:
            raise ValueError("Pad vertex contacts do not support differentiable collision replay")
        self._last_generation = generation
        wp.launch(
            lowest_mesh_vertices,
            dim=len(self.shapes),
            inputs=[
                self.shapes,
                self.starts,
                self.vertices,
                m.shape_body,
                m.shape_transform,
                env.states[0].body_q,
                self.world_points,
                self.local_points,
            ],
            device=m.device,
        )
        wp.launch(
            query_top_faces,
            dim=self.heights.shape,
            inputs=[
                self.shapes,
                m.shape_world,
                self.world_points,
                env.states[0].particle_q,
                env.terrain.top_faces,
                env.cfg["terrain"]["soft_contact"]["margin"],
                env.terrain.config.base_z,
                self.heights,
            ],
            device=m.device,
        )
        wp.launch(
            append_support_contacts,
            dim=len(self.shapes),
            inputs=[
                self.shapes,
                m.shape_world,
                self.world_points,
                self.local_points,
                env.states[0].particle_q,
                env.terrain.top_faces,
                self.heights,
                c.soft_contact_max,
                c.soft_contact_count,
                c.soft_contact_particle,
                c.soft_contact_indices,
                c.soft_contact_barycentric,
                c.soft_contact_shape,
                c.soft_contact_body_pos,
                c.soft_contact_body_vel,
                c.soft_contact_normal,
            ],
            device=m.device,
        )


# Mixed terrain and deformation guard

PAD_EDGE_TIMEOUT_MARGIN = 0.08


@wp.kernel
def _display_particle_positions(
    positions: wp.array[wp.vec3],
    worlds: wp.array[int],
    offsets: wp.array[wp.vec3],
    display: wp.array[wp.vec3],
):
    i = wp.tid()
    point = positions[i]
    if worlds[i] >= 0:
        point += offsets[worlds[i]]
    display[i] = point


@wp.kernel
def _surface_height(
    particles: wp.array[wp.vec3],
    faces: wp.array2d[wp.vec3i],
    queries: wp.array2d[wp.vec3],
    heights: wp.array2d[float],
):
    world, query, face = wp.tid()
    indices = faces[world, face]
    if indices[0] < 0:
        return
    a, b, c = particles[indices[0]], particles[indices[1]], particles[indices[2]]
    p = queries[world, query]
    ab, ac, ap = b - a, c - a, p - a
    determinant = ab[0] * ac[1] - ab[1] * ac[0]
    if wp.abs(determinant) > 1.0e-12:
        u = (ap[0] * ac[1] - ap[1] * ac[0]) / determinant
        v = (ab[0] * ap[1] - ab[1] * ap[0]) / determinant
        if u >= -1.0e-5 and v >= -1.0e-5 and u + v <= 1.00001:
            # Query the deformed surface even if the foot has penetrated it.
            wp.atomic_max(heights, world, query, a[2] + u * ab[2] + v * ac[2])


@wp.kernel
def _volume_ratios(
    particles: wp.array[wp.vec3],
    tets: wp.array2d[int],
    rest: wp.array[float],
    worlds: wp.array[int],
    minimum: wp.array[float],
):
    i = wp.tid()
    tet = wp.vec4i(tets[i, 0], tets[i, 1], tets[i, 2], tets[i, 3])
    a, b, c, d = particles[tet[0]], particles[tet[1]], particles[tet[2]], particles[tet[3]]
    ratio = wp.dot(b - a, wp.cross(c - a, d - a)) / rest[i]
    if not wp.isfinite(ratio):
        ratio = -1.0e10
    wp.atomic_min(minimum, worlds[tet[0]], ratio)


def add_terrain_pad(robot, config):
    """Provision SDFs once on the template before it is replicated into worlds."""
    provision_pad_contacts(robot)
    add_soft_pad(robot, config)


def provision_pad_contacts(robot):
    """The global full-surface pipeline validates every particle-collidable rigid mesh."""
    for shape, flags in enumerate(robot.shape_flags):
        if flags & newton.ShapeFlags.COLLIDE_PARTICLES and robot.shape_type[shape] in (
            newton.GeoType.MESH,
            newton.GeoType.CONVEX_MESH,
        ):
            robot.shape_force_sdf[shape] = True
            robot.shape_sdf_max_resolution[shape] = 64


def add_inset_floor(builder, config, *, has_pad):
    """Finite rigid tiles leave a real hole; no z=0 plane may cover the deformable region."""
    hx, hy = config.floor_half_extents
    cx, cy = config.cavity_half_extents
    bottom = config.cavity_bottom_z
    thickness = max(0.1, -bottom)
    floor = builder.default_shape_cfg.copy()
    floor.is_visible = True

    def box(label, x, y, z, sx, sy, sz):
        shape = builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(x, y, z), wp.quat_identity()),
            hx=sx,
            hy=sy,
            hz=sz,
            cfg=floor,
            label="inset_floor_" + label,
            color=(0.50, 0.53, 0.57),
        )
        filter_authored_collider(builder, shape)

    if not has_pad:
        box("flat", 0, 0, -thickness / 2, hx, hy, thickness / 2)
        return
    # The lower slab supports objects that miss the top mesh, but sits below the
    # bonded bottom's collision skin and cannot prop up the deforming upper layers.
    box("bottom", 0, 0, bottom - thickness / 2, hx, hy, thickness / 2)
    depth = -bottom / 2
    for sign, label in ((-1, "left"), (1, "right")):
        box(label, sign * (hx + cx) / 2, 0, -depth, (hx - cx) / 2, hy, depth)
    for sign, label in ((-1, "front"), (1, "back")):
        box(label, 0, sign * (hy + cy) / 2, -depth, cx, (hy - cy) / 2, depth)


def pad_world_count(num_envs, fraction):
    """Deterministic nearest-integer allocation; a single world at 50% gets a pad."""
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("pad_probability must be in [0, 1]")
    return int(np.floor(num_envs * fraction + 0.5))


def padded_indices(groups):
    """Use -1 only as a lookup sentinel, never as a particle index for state writes."""
    result = np.full((len(groups), max(map(len, groups), default=0)), -1, dtype=np.int64)
    for i, group in enumerate(groups):
        result[i, : len(group)] = group
    return result


class SoftPadGuard:
    """Bound the training domain; this is not an inversion-resistant material or solver."""

    def __init__(self, terrain):
        self.terrain, self.env = terrain, terrain.env
        env = self.env
        self.settings = {
            # Allow positive-volume compression; zero volume, inversion and nonfinite states still fail.
            "min_volume_ratio": 0.0,
            "max_failed_fraction_per_step": 0.05,
            "rate_window_steps": 100,
            "max_failed_transition_fraction": 0.001,
            "min_failures_for_rate_stop": 8,
            "response": "restore_control_step_start_then_failure_reset",
        }
        env.cfg["terrain"]["deformation_guard"] = self.settings.copy()
        self.failed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.invalid = torch.zeros_like(self.failed)
        self.frame_minimum = torch.ones(env.num_envs, device=env.device)
        self.failure_count = self.invalid_count = self.window_failures = self.window_steps = 0
        self.diagnostic_dir = Path("logs/microduck_diagnostics")
        self.reported = set()
        self.backup, self.indices, self.drive_backup = {}, {}, {}

    def begin_step(self):
        env = self.env
        env._sync()
        self.failed.zero_()
        self.invalid.zero_()
        self.frame_minimum.copy_(self.terrain.minimum_volume)
        self.start_volume = self.terrain.minimum_volume.clone()
        # Native stepping overwrites both particle buffers. Keep independent storage,
        # including affine proxies, rather than treating states[1] as a previous state.
        for name in ("particle_q", "particle_qd", "body_q", "body_qd", "joint_q", "joint_qd"):
            if getattr(env.states[0], name) is None:
                continue
            values = wp.to_torch(getattr(env.states[0], name))
            if name not in self.backup:
                self.backup[name] = values.clone()
                kind = name.split("_")[0]
                if kind == "joint":
                    indices = np.arange(len(values)).reshape(env.num_envs, len(values) // env.num_envs)
                else:
                    worlds = getattr(env.model, kind + "_world").numpy()
                    indices = padded_indices([np.flatnonzero(worlds == i) for i in range(env.num_envs)])
                self.indices[name] = torch.tensor(indices, device=env.device)
            else:
                self.backup[name].copy_(values)
        for name in ("_joint_drive_feedback_angle_initialized", "_joint_drive_feedback_previous_angles"):
            values = getattr(env.solver, name, None)
            if values is not None:
                if name not in self.drive_backup:
                    children = wp.to_torch(env.solver._joint_drive_body_indices).long()[:, 1]
                    self.drive_world = wp.to_torch(env.model.body_world).long()[children]
                    self.drive_backup[name] = wp.to_torch(values).clone()
                else:
                    self.drive_backup[name].copy_(wp.to_torch(values))

    def _restore(self):
        env = self.env
        ids = self.failed.nonzero().flatten()
        for state in env.states:
            for name, saved in self.backup.items():
                indices = self.indices[name][ids].flatten()
                indices = indices[indices >= 0]
                wp.to_torch(getattr(state, name))[indices] = saved[indices]
            for kind in ("particle", "body"):
                if kind + "_q" in self.indices:
                    indices = self.indices[kind + "_q"][ids].flatten()
                    indices = indices[indices >= 0]
                    wp.to_torch(getattr(state, kind + "_f"))[indices] = 0.0
        for name, saved in self.drive_backup.items():
            selected = (self.drive_world >= 0) & self.failed[self.drive_world.clamp_min(0)]
            wp.to_torch(getattr(env.solver, name))[selected] = saved[selected]
        self.terrain.minimum_volume[ids] = self.start_volume[ids]
        env._sync()

    def _report(self, ids, substep, kind):
        if kind in self.reported:
            return
        self.reported.add(kind)
        selected = ids[:4]
        report = {
            "kind": kind,
            "control_step": self.env.official.common_step_counter,
            "substep": substep,
            "ids": selected.cpu(),
            "minimum_volume": self.terrain.minimum_volume[selected].cpu(),
            "actions": self.env.actions[selected].cpu(),
            "commands": self.env.commands[selected].cpu(),
            "episode_length": self.env.episode_length_buf[selected].cpu(),
            "terrain": self.env.cfg["terrain"],
            "before": {},
            "after": {},
        }
        for name, saved in self.backup.items():
            indices = self.indices[name][selected]
            # Pad-only failures have equal topology; strip absent-world sentinels
            # before exporting legacy rectangular per-world diagnostics.
            indices = torch.stack([row[row >= 0] for row in indices])
            report["before"][name] = saved[indices].cpu()
            report["after"][name] = wp.to_torch(getattr(self.env.states[0], name))[indices].cpu()
        self.diagnostic_dir.mkdir(parents=True, exist_ok=True)
        path = self.diagnostic_dir / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_pad_{kind}.pt")
        torch.save(report, path)
        print(f"Soft-pad {kind}: terminating worlds {ids[:16].tolist()}; diagnostic={path.resolve()}", flush=True)

    def after_substep(self, substep):
        self.terrain.measure_volume()
        indices = self.terrain.pad_indices
        nonfinite = torch.zeros_like(self.failed)
        if indices.numel():
            velocity = wp.to_torch(self.env.states[0].particle_qd)[indices.clamp_min(0)]
            nonfinite = ((~torch.isfinite(velocity).all(-1)) & (indices >= 0)).any(-1)
        minimum = self.terrain.minimum_volume
        self.frame_minimum.copy_(torch.where(self.failed, self.frame_minimum, torch.minimum(self.frame_minimum, minimum)))
        invalid = ((minimum <= 0) | nonfinite) & ~self.failed
        new = ((minimum <= self.settings["min_volume_ratio"]) | nonfinite) & ~self.failed
        if new.any():
            for kind, mask in (("invalid", invalid), ("compression_limit", new & ~invalid)):
                if mask.any():
                    self._report(mask.nonzero().flatten(), substep, kind)
            self.failed |= new
            self.invalid |= invalid
        if self.failed.any():
            # Keep terminal worlds at their last valid control state for all remaining
            # substeps. They are reset only at the episode boundary, never mid-action.
            self._restore()

    def finish_step(self):
        count = int(self.failed.sum().item())
        self.failure_count += count
        self.invalid_count += int(self.invalid.sum().item())
        self.window_failures += count
        self.window_steps += 1
        settings = self.settings
        exposed = max(1, self.terrain.num_pad_worlds)
        widespread = count > max(4, exposed * settings["max_failed_fraction_per_step"])
        transitions = self.window_steps * exposed
        rate = self.window_failures / transitions
        window_complete = self.window_steps >= settings["rate_window_steps"]
        excessive_rate = (
            window_complete
            and self.window_failures >= settings["min_failures_for_rate_stop"]
            and rate > settings["max_failed_transition_fraction"]
        )
        if widespread or excessive_rate:
            # Preserve the offending window before clearing counters; the first
            # per-kind state dump alone cannot explain an aggregate rate stop.
            report = {
                "reason": "single_step_failures" if widespread else "window_failure_rate",
                "control_step": self.env.official.common_step_counter,
                "pad_world_count": self.terrain.num_pad_worlds,
                "failed_this_step": count,
                "window_steps": self.window_steps,
                "window_failures": self.window_failures,
                "failed_transition_fraction": rate,
                "failure_count": self.failure_count,
                "invalid_count": self.invalid_count,
                "settings": settings.copy(),
            }
            self.diagnostic_dir.mkdir(parents=True, exist_ok=True)
            path = self.diagnostic_dir / "guard_stop.json"
            path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            raise FloatingPointError(
                f"Too frequent soft-pad failures: {report['reason']}; "
                f"failed_this_step={count}/{exposed}, "
                f"window_failures={self.window_failures}/{transitions} ({rate:.4%}), "
                f"rate_limit={settings['max_failed_transition_fraction']:.4%}, "
                f"total_inverted_or_nonfinite={self.invalid_count}; diagnostic={path.resolve()}"
            )
        if window_complete:
            self.window_failures = self.window_steps = 0

    def terminations(self, failure, timeout):
        # Numerical termination must not receive RSL-RL's timeout value bootstrap.
        return failure | self.failed, timeout & ~self.failed


class MixedTerrain:
    """Fixed flat/pad cohorts, with explicit support for historical resampled scenes."""

    kind = "mixed_soft_pad"
    shape_count = 1

    def __init__(self, env, builder, config=None, pad_probability=0.5, *, sampling="independent_per_episode"):
        if not np.isfinite(pad_probability) or not 0.0 <= pad_probability <= 1.0:
            raise ValueError("pad_probability must be in [0, 1]")
        self.env = env
        self.config = config or SoftPadConfig()
        self.pad_probability = pad_probability
        if sampling not in ("fixed_groups_center", "independent_per_episode"):
            raise ValueError("unknown pad sampling mode")
        self.sampling = sampling
        self.origins = torch.zeros(env.num_envs, 3, device=env.device)
        self.spawn_on_pad = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.minimum_volume = torch.ones(env.num_envs, device=env.device)
        if not self.config.embedded:
            floor = builder.default_shape_cfg.copy()
            floor.is_visible = True
            filter_authored_collider(builder, builder.add_ground_plane(cfg=floor, color=(0.50, 0.53, 0.57)))
        self.shape_count = sum(body < 0 for body in builder.shape_body)
        env.cfg["terrain"] = {
            "kind": self.kind,
            "pad": asdict(self.config),
            "pad_cells": list(self.config.resolved_cells),
            "pad_probability": pad_probability,
            "sampling": sampling,
            "floor": "world_local_cavity" if self.config.embedded else "shared_plane",
            "soft_contact": {"ke": self.config.contact_stiffness, "kd": 100.0, "mu": 0.8, "margin": 0.01},
            "pad_edge_timeout_margin": PAD_EDGE_TIMEOUT_MARGIN,
            "boundary_curriculum": "surface_locomotion_before_edge_crossing",
            "vertex_face_contacts": "lowest_mesh_vertex_top_surface_v1",
            "scene_timeout_radius": 2.5,
        }
        self.guard = SoftPadGuard(self)

    def finalize(self):
        env, model = self.env, self.env.model
        # Affine proxies can reorder particles. Resolve membership only from finalized topology.
        rest = model.particle_q.numpy()
        tets = model.tet_indices.numpy() if model.tet_count else np.empty((0, 4), dtype=np.int32)
        particles = np.unique(tets)
        membership = np.zeros(model.particle_count, dtype=np.int32)
        membership[particles] = 1
        self.pad_particle = wp.array(membership, dtype=int, device=str(env.device))
        worlds = model.particle_world.numpy().copy()
        if env.num_envs == 1:
            # Standalone builders label their sole world -1; replicated builders use 0.
            worlds[worlds < 0] = 0
        self.particle_world = wp.array(worlds, dtype=int, device=str(env.device))
        indices = [particles[worlds[particles] == i] for i in range(env.num_envs)]
        self.has_pad = torch.tensor([len(row) > 0 for row in indices], dtype=torch.bool, device=env.device)
        self.pad_indices = torch.tensor(padded_indices(indices), dtype=torch.long, device=env.device)
        if self.sampling == "fixed_groups_center":
            self.spawn_on_pad.copy_(self.has_pad)
        self.num_pad_worlds = sum(len(row) > 0 for row in indices)
        env.cfg["terrain"]["pad_world_count"] = self.num_pad_worlds
        env.cfg["terrain"]["pad_world_ids"] = self.has_pad.nonzero().flatten().tolist()
        self.rest = wp.to_torch(model.particle_q).clone()
        faces = model.tri_indices.numpy() if model.tri_count else np.empty((0, 3), dtype=np.int32)
        top = faces[np.isclose(rest[faces, 2], self.config.top_z).all(axis=1)]
        face_groups = [top[worlds[top[:, 0]] == i] for i in range(env.num_envs)]
        grouped = np.full((env.num_envs, max(map(len, face_groups), default=0), 3), -1, dtype=np.int32)
        for i, group in enumerate(face_groups):
            grouped[i, : len(group)] = group
        self.top_faces = wp.array(grouped, dtype=wp.vec3i, device=str(env.device))
        p = rest[tets]
        volumes = np.einsum("ij,ij->i", p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]))
        if (volumes <= 0).any():
            raise ValueError("Replicated pad has nonpositive rest volume")
        self.rest_volumes = wp.array(volumes, dtype=float, device=str(env.device))
        model.soft_contact_ke, model.soft_contact_kd, model.soft_contact_mu = self.config.contact_stiffness, 100.0, 0.8

    def reset(self, ids):
        if self.sampling == "independent_per_episode":
            self.spawn_on_pad[ids] = torch.rand(len(ids), device=self.env.device) < self.pad_probability
        else:
            self.spawn_on_pad[ids] = self.has_pad[ids]
        self.origins[ids] = 0.0
        if self.sampling == "independent_per_episode":
            self.origins[ids, 0] = torch.where(self.spawn_on_pad[ids], 0.0, -self.config.size[0] / 2 - 0.9)
        self.origins[ids, 2] = self.spawn_on_pad[ids] * self.config.top_z
        if getattr(self.env, "interactive", False):
            self.spawn_on_pad[ids] = self.env.spawn == "pad"
            self.origins[ids, 0] = 0.0 if self.env.spawn == "pad" else -self.config.size[0] / 2 - 0.3
            self.origins[ids, 2] = self.config.top_z if self.env.spawn == "pad" else 0.0
        particles = self.pad_indices[ids].flatten()
        particles = particles[particles >= 0]
        for state in self.env.states:
            if not particles.numel():
                break
            wp.to_torch(state.particle_q)[particles] = self.rest[particles]
            wp.to_torch(state.particle_qd)[particles] = 0.0
            wp.to_torch(state.particle_f)[particles] = 0.0
        self.minimum_volume[ids] = 1.0

    def display_state(self, viewer, state):
        """Apply Newton's rigid-world layout to soft geometry using a render-only buffer."""
        if viewer.world_offsets is None:
            return state
        if not hasattr(self, "_display_points"):
            self._display_points = wp.empty_like(state.particle_q)
        # Newton 1.5 offsets rigid shapes, but logs triangle/particle positions
        # directly. Never translate simulation buffers or rigid bodies here.
        wp.launch(
            _display_particle_positions,
            dim=self.env.model.particle_count,
            inputs=[state.particle_q, self.env.model.particle_world, viewer.world_offsets],
            outputs=[self._display_points],
            device=self.env.model.device,
        )
        display = copy(state)
        display.particle_q = self._display_points
        return display

    def adjust_spawn(self, ids):
        # Upstream samples +/-0.5 m in XY, which would put many pad starts beyond its narrow edge.
        selected = ids[self.spawn_on_pad[ids]]
        spawn = self.env.task_logic.spawn
        if getattr(self.env, "interactive", False):
            selected = ids
            xy = self.origins[selected, :2]
            spawn[selected, 2] += self.env.clearance
        elif self.sampling == "fixed_groups_center":
            xy = torch.zeros(len(selected), 2, device=self.env.device)
        else:
            extent = torch.tensor(self.config.size[:2], device=self.env.device) / 2 - 0.2
            if (extent <= 0).any():
                raise ValueError("Training pads must be wider than 0.4 m in both horizontal directions")
            xy = (torch.rand(len(selected), 2, device=self.env.device) * 2 - 1) * extent
        offset = xy - spawn[selected, :2]
        spawn[selected, :2] = xy
        if getattr(self.env.task_logic, "ball_body", None) is not None:
            # A terrain translation must move the task object with its robot-relative reset pose.
            start = self.env.task_logic.ball_q
            spawn[selected, start : start + 2] += offset
            spawn[ids, start + 2] += self.surface_heights(spawn[:, None, start : start + 3])[ids, 0]

    def surface_heights(self, positions):
        queries = positions.contiguous()
        result = torch.zeros(queries.shape[:-1], device=self.env.device)
        if self.config.embedded:
            extent = torch.tensor(self.config.cavity_half_extents, device=self.env.device)
            in_cavity = self.has_pad[:, None] & (queries[..., :2].abs() < extent).all(-1)
            # Initializing every ray to zero would hide every below-ground depression.
            result.masked_fill_(in_cavity, self.config.cavity_bottom_z)
        if self.top_faces.shape[1] == 0:
            return result
        self.env._sync()
        wp.launch(
            _surface_height,
            dim=(*result.shape, self.top_faces.shape[1]),
            inputs=[
                self.env.states[0].particle_q,
                self.top_faces,
                wp.from_torch(queries, dtype=wp.vec3),
                wp.from_torch(result),
            ],
            device=str(self.env.device),
        )
        self.env._sync()
        return result

    def heights(self, positions, quaternions=None):
        return (positions[..., 2] - self.surface_heights(positions)).clamp(0.0, 1.0)

    def update_support(self, pose):
        self.origins[:, 2] = self.surface_heights(pose[:, None, :3])[:, 0]

    def out_of_bounds(self, positions):
        margin = self.env.cfg["terrain"]["pad_edge_timeout_margin"]
        if margin is None:
            # Playback may cross either side of the pad; retain the finite scene boundary for all worlds.
            radius = self.env.cfg["terrain"]["scene_timeout_radius"]
            return (positions[:, :2] - self.origins[:, :2]).norm(dim=-1) > radius
        # End the episode while the robot still has room to recover on the pad.
        extent = torch.tensor(self.config.size[:2], device=self.env.device) / 2 - margin
        pad_exit = (positions[:, :2].abs() > extent).any(-1)
        flat_exit = (positions[:, :2] - self.origins[:, :2]).norm(dim=-1) > 2.5
        approaching_pad = self.has_pad & (positions[:, :2].abs() < extent + 0.24).all(-1)
        flat_exit |= approaching_pad
        return torch.where(self.spawn_on_pad, pad_exit, flat_exit)

    def check_state(self):
        self.measure_volume()
        if (self.minimum_volume <= 0).any():
            ids = (self.minimum_volume <= 0).nonzero().flatten().tolist()
            raise FloatingPointError(f"Inverted/non-finite soft-pad tetrahedra in worlds {ids[:16]}")

    def measure_volume(self):
        self.minimum_volume.fill_(1.0)
        if self.env.model.tet_count == 0:
            return
        self.env._sync()
        wp.launch(
            _volume_ratios,
            dim=self.env.model.tet_count,
            inputs=[
                self.env.states[0].particle_q,
                self.env.model.tet_indices,
                self.rest_volumes,
                self.particle_world,
                wp.from_torch(self.minimum_volume),
            ],
            device=str(self.env.device),
        )
        self.env._sync()

    def check_contacts(self):
        self.env._sync()
        contacts = self.env.contacts
        if self.num_pad_worlds and getattr(contacts, "_enable_rigid_soft_full_surface_contact", False):
            if not hasattr(self, "_vertex_contacts"):
                self._vertex_contacts = PadVertexContacts(self.env)
            # Both pre-solve and sensor collision passes use this same contact stream.
            self._vertex_contacts.update()
            self.env._sync()
        for kind in ("rigid", "soft"):
            count = int(wp.to_torch(getattr(contacts, f"{kind}_contact_count"))[0].item())
            capacity = getattr(contacts, f"{kind}_contact_max")
            if count > capacity:
                raise RuntimeError(f"{kind} contact capacity exceeded: {count} > {capacity}")

    def log(self, reward, failure, timeout, linear, gravity, commands):
        result = {
            "Terrain/pad_spawn_fraction": self.spawn_on_pad.float().mean(),
            "Terrain/min_volume_ratio": self.minimum_volume.min(),
            "Terrain/attempted_min_volume_ratio": self.guard.frame_minimum.min(),
            "Terrain/deformation_failure_fraction": self.guard.failed.float().mean(),
            "Terrain/inversion_or_nonfinite_fraction": self.guard.invalid.float().mean(),
            "Terrain/deformation_failure_count": float(self.guard.failure_count),
            "Terrain/inversion_or_nonfinite_count": float(self.guard.invalid_count),
        }
        for name, mask in (("pad", self.spawn_on_pad), ("flat", ~self.spawn_on_pad)):
            count = mask.sum().clamp_min(1)
            values = {
                "reward": reward,
                "failure_fraction": failure.float(),
                "timeout_fraction": timeout.float(),
                "upright": -gravity[:, 2],
                "episode_length": self.env.episode_length_buf.float(),
            }
            if getattr(self.env, "family", "velocity") in ("velocity", "velstand"):
                values["velocity_error"] = (linear[:, :2] - commands[:, :2]).norm(dim=-1)
            result.update({f"Terrain/{name}/{key}": (value * mask).sum() / count for key, value in values.items()})
        return result


# Official static terrain


@wp.kernel
def _height_ray(mesh: wp.uint64, positions: wp.array[wp.vec3], heights: wp.array[float]):
    i = wp.tid()
    p = positions[i]
    hit = wp.mesh_query_ray(mesh, p, wp.vec3(0.0, 0.0, -1.0), 1.0)
    heights[i] = -1.0
    if hit.result:
        heights[i] = hit.t
        if hit.normal[2] < 0.0:
            heights[i] = 0.0


class OfficialTerrain:
    def __init__(self, env, builder):
        self.env = env
        self.kind = "rough" if "_rough" in env.task else "slope" if env.family == "rollerslope" else "plane"
        self.origins = torch.zeros(env.num_envs, 3, device=env.device)
        self.mesh = None
        self.levels = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.columns = torch.zeros_like(self.levels)
        if self.kind == "plane":
            filter_authored_collider(builder, builder.add_ground_plane())
            self.shape_count = 1
            return
        source = json.loads((Path(__file__).parent / "configs" / f"terrain_{self.kind}.json").read_text())
        self.grid = torch.tensor(source["origins"], device=env.device)
        if self.kind == "rough":
            self.levels.random_(self.grid.shape[0])
            self.columns.random_(self.grid.shape[1])
        self.origins.copy_(self.grid[self.levels, self.columns])
        vertices, faces = [], []
        corners = np.array([[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]])
        box_faces = np.array(
            [
                [0, 4, 6],
                [0, 6, 2],
                [1, 3, 7],
                [1, 7, 5],
                [0, 1, 5],
                [0, 5, 4],
                [2, 6, 7],
                [2, 7, 3],
                [0, 2, 3],
                [0, 3, 1],
                [4, 5, 7],
                [4, 7, 6],
            ]
        )
        for i, geom in enumerate(source["geoms"]):
            w, x, y, z = geom["quat"]
            xf = wp.transform(geom["pos"], (x, y, z, w))
            cfg = builder.default_shape_cfg.copy()
            cfg.mu = 1.0
            if geom["type"] == 6:
                shape = builder.add_shape_box(
                    -1, xform=xf, hx=geom["size"][0], hy=geom["size"][1], hz=geom["size"][2], cfg=cfg, label=f"terrain_{i}"
                )
                points = corners * np.asarray(geom["size"])
                triangles = box_faces
            else:
                nr, nc = geom["nrow"], geom["ncol"]
                hx, hy, hz, _ = geom["hsize"]
                data = np.asarray(geom["data"]).reshape(nr, nc)
                shape = builder.add_shape_heightfield(
                    xform=xf, heightfield=newton.Heightfield(data, nr, nc, hx, hy, 0.0, hz), cfg=cfg, label=f"terrain_{i}"
                )
                xx, yy = np.meshgrid(np.linspace(-hx, hx, nc), np.linspace(-hy, hy, nr))
                points = np.stack([xx, yy, data * hz], axis=-1).reshape(-1, 3)
                triangles = []
                for row in range(nr - 1):
                    for col in range(nc - 1):
                        a = row * nc + col
                        triangles.extend([[a, a + 1, a + nc + 1], [a, a + nc + 1, a + nc]])
                triangles = np.asarray(triangles)
            filter_authored_collider(builder, shape)
            offset = len(vertices)
            vertices.extend([list(wp.transform_point(xf, wp.vec3(*p))) for p in points])
            faces.extend((triangles + offset).reshape(-1).tolist())
        self.shape_count = len(source["geoms"])
        self.mesh = wp.Mesh(
            points=wp.array(vertices, dtype=wp.vec3, device=str(env.device)),
            indices=wp.array(faces, dtype=wp.int32, device=str(env.device)),
        )

    def heights(self, positions, quaternions=None):
        if self.mesh is None:
            return positions[..., 2].clamp(0.0, 1.0)
        rays = positions.unsqueeze(-2).repeat(1, 1, 3, 1)
        if quaternions is not None:
            w, x, y, z = quaternions.unbind(-1)
            yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
            offset = torch.stack((yaw.cos(), yaw.sin(), torch.zeros_like(yaw)), -1) * 0.04
            rays[:, :, 1] += offset
            rays[:, :, 2] -= offset
        result = torch.empty(rays.shape[:-1], device=self.env.device)
        wp.launch(
            _height_ray,
            dim=result.numel(),
            inputs=[
                self.mesh.id,
                wp.from_torch(rays.reshape(-1, 3).contiguous(), dtype=wp.vec3),
                wp.from_torch(result.flatten()),
            ],
            device=str(self.env.device),
        )
        self.env._sync()
        missing = result < 0
        all_missing = missing.all(-1)
        result = torch.where(missing, torch.ones_like(result), result).min(-1).values
        return torch.where(all_missing, positions[..., 2].clamp(0, 1), result)

    def reset(self, ids):
        if self.kind == "plane" or self.env.official is None:
            return
        active = ids[self.env.episode_length_buf[ids] > 0]
        if not len(active):
            return
        position = self.env._views()[2][active, :3]
        if self.kind == "slope":
            distance = position[:, 0] - self.origins[active, 0]
            up, down = distance > 6.0, distance < 3.0
        else:
            distance = (position[:, :2] - self.origins[active, :2]).norm(dim=-1)
            up = distance > 4.0
            target = self.env.commands[active, :2].norm(dim=-1) * self.env.max_episode_length * self.env.step_dt * 0.5
            down = (distance < target) & ~up
        levels = self.levels[active] + up.long() - down.long()
        levels = torch.where(
            levels >= len(self.grid), torch.randint(len(self.grid), levels.shape, device=levels.device), levels
        ).clamp(min=0)
        self.levels[active] = levels
        self.origins[active] = self.grid[levels, self.columns[active]]

    def out_of_bounds(self, positions):
        if self.kind != "rough":
            return torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        return (positions[:, :2].abs() > 19.7).any(-1)


# Frozen standalone policy playback


def load_policy_checkpoint(path, device, *, allow_recipe_mismatch=False):
    """Validate the saved control contract and restore the frozen RSL-RL inference path."""
    from rsl_rl.modules import ActorCritic, EmpiricalNormalization

    from .checkpoint import checkpoint_policy_variant
    from .observations import CONTRACT, HISTORY_SIZE
    from .recipe import RECIPE, recipe_fingerprint
    from .task_registry import TASKS

    path = Path(path).expanduser().resolve()
    if path.is_dir():
        candidates = [p for p in path.glob("model_*.pt") if p.stem.removeprefix("model_").isdigit()]
        if not candidates:
            raise ValueError(f"No model_<iteration>.pt checkpoint in {path}")
        path = max(candidates, key=lambda p: int(p.stem.removeprefix("model_")))
    if not path.is_file():
        raise ValueError(f"Checkpoint does not exist: {path}")
    metadata = json.loads(path.with_name("config.json").read_text(encoding="utf-8"))
    required = {
        "solver": "synreal",
        "recipe": RECIPE,
        "observation_contract": CONTRACT,
        "actuator": "position_pd",
    }
    for name, expected in required.items():
        if metadata.get(name) != expected:
            raise ValueError(f"Soft-pad playback requires {name}={expected!r}; saved value is {metadata.get(name)!r}")
    if metadata.get("task") not in TASKS:
        raise ValueError(f"Unknown checkpoint task: {metadata.get('task')!r}")
    for name, expected in (("kp", KP), ("kd", KD), ("effort_limit", EFFORT), ("policy_dt", 0.02)):
        value = metadata.get(name)
        if not isinstance(value, (int, float)) or not math.isclose(value, expected, rel_tol=1.0e-6):
            raise ValueError(f"Checkpoint {name}={value!r} does not match the supported control contract ({expected})")
    physics_dt = metadata.get("physics_dt")
    if not isinstance(physics_dt, (int, float)) or not math.isfinite(physics_dt) or physics_dt <= 0:
        raise ValueError("Checkpoint physics_dt must be finite and positive")
    substeps = metadata["policy_dt"] / physics_dt
    if substeps < 1 or not math.isclose(substeps, round(substeps), rel_tol=1.0e-6):
        raise ValueError("Checkpoint physics_dt must divide policy_dt into an integer number of substeps")
    alignment = metadata["official_alignment"]
    critic_dim = alignment.get("critic_dim")
    if alignment.get("actor_dim") != 61 or type(critic_dim) is not int or critic_dim < 61:
        raise ValueError("Soft-pad playback requires a 61-D actor and a valid task-specific critic_dim")
    current_fingerprint = recipe_fingerprint()
    mismatch = alignment.get("recipe_fingerprint") != current_fingerprint
    if mismatch and not allow_recipe_mismatch:
        raise ValueError("Recipe files differ from the checkpoint; pass --allow-recipe-mismatch for diagnostic playback")
    if mismatch:
        print("Recipe fingerprint differs; diagnostic playback uses the current observation recipe.", flush=True)

    variant = checkpoint_policy_variant(metadata)
    policy_cfg = metadata["ppo"]["policy"].copy()
    policy_cfg.pop("class_name")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    policy_class, normalizer_class = ActorCritic, EmpiricalNormalization
    extra = 0
    if variant == "history":
        from .policy import HistoryActorCritic, HistoryNormalization

        policy_class, normalizer_class = HistoryActorCritic, HistoryNormalization
        extra = HISTORY_SIZE
    actor = policy_class(61 + extra, critic_dim + extra, 14, **policy_cfg).to(device)
    actor.load_state_dict(checkpoint["model_state_dict"], strict=True)
    actor.eval()
    actor.requires_grad_(False)
    if metadata["ppo"]["empirical_normalization"]:
        normalizer = normalizer_class(61).to(device)
        normalizer.load_state_dict(checkpoint["obs_norm_state_dict"], strict=True)
    else:
        normalizer = torch.nn.Identity()
    normalizer.eval()
    for module in (actor, normalizer):
        if not all(torch.isfinite(value).all() for value in module.state_dict().values()):
            raise ValueError("Checkpoint contains non-finite policy or normalizer values")
    stats = getattr(normalizer, "current", normalizer)
    if hasattr(stats, "std") and (stats.std < 0).any():
        raise ValueError("Checkpoint normalizer contains a negative standard deviation")
    print(f"Soft-pad checkpoint: {path} (iteration {checkpoint['iter']})", flush=True)
    return SimpleNamespace(
        path=path,
        metadata=metadata,
        actor=actor,
        normalizer=normalizer,
        iteration=int(checkpoint["iter"]),
        recipe_mismatch=mismatch,
        current_recipe_fingerprint=current_fingerprint,
    )


class SoftPadPlayback:
    """Reuse the normal playback actor observations without rigid-ground reward sensors."""

    def __init__(self, env, checkpoint, command):
        from .checkpoint import checkpoint_policy_variant
        from .observations import configure_policy_observations
        from .play import PolicyPlayback

        self.env, self.checkpoint = env, checkpoint
        if env.task != checkpoint.metadata["task"]:
            raise ValueError("Soft-pad environment task must match its checkpoint")
        configure_policy_observations(env, checkpoint_policy_variant(checkpoint.metadata))
        env.official.common_step_counter = (checkpoint.iteration + 1) * checkpoint.metadata["ppo"].get("num_steps_per_env", 24)
        if env.interactive:
            env.reset()
        self.command = list(command)
        self.previous_command = None
        self.key_was_down = {}
        self.observer = PolicyPlayback(env)
        self.steps = 0
        self.last_observation = None

    @torch.no_grad()
    def begin_frame(self):
        from .play import update_playback_observation, update_velocity_command

        env = self.env
        if env.family in ("velocity", "velstand"):
            if env.viewer is not None and env.viewer.is_key_down("x"):
                self.command[:] = [0.0, 0.0, 0.0]
            self.previous_command = update_velocity_command(
                env, env.viewer, self.command, self.key_was_down, self.previous_command
            )
        if env.official.observations.history is None:
            env.official.observations.cache = None
        observation = self.observer.observe()
        update_playback_observation(env, observation)
        self.last_observation = observation.clone()
        actions = self.checkpoint.actor.act_inference(self.checkpoint.normalizer(observation))
        if actions.shape != env.actions.shape or not torch.isfinite(actions).all():
            raise FloatingPointError("Invalid soft-pad policy actions")
        env.actions.copy_(actions)
        self.steps += 1
        return env.actions

    def reset(self):
        self.env.reset()
        self.steps = 0
        self.last_observation = None

    def metrics(self):
        checkpoint = self.checkpoint
        return {
            "checkpoint": str(checkpoint.path),
            "iteration": checkpoint.iteration,
            "task": self.env.task,
            "command": self.env.commands[0, :3].tolist(),
            "policy_steps": self.steps,
            "policy_dt": checkpoint.metadata["policy_dt"],
            "trained_physics_dt": checkpoint.metadata["physics_dt"],
            "recipe_mismatch": checkpoint.recipe_mismatch,
            "saved_recipe_fingerprint": checkpoint.metadata["official_alignment"]["recipe_fingerprint"],
            "current_recipe_fingerprint": checkpoint.current_recipe_fingerprint,
            "max_abs_action": float(self.env.actions.abs().max().item()),
        }


# Standalone pad viewer and CLI


@wp.kernel
def gather_surface(particles: wp.array[wp.vec3], indices: wp.array[int], points: wp.array[wp.vec3]):
    i = wp.tid()
    points[i] = particles[indices[i]]


def login(force: bool = False):
    import style3d_sim as s3d

    if s3d.is_logged_in():
        return
    account = os.getenv("STYLE3D_SIM_ACCOUNT") or os.getenv("STYLE3D_ACCOUNT")
    password = os.getenv("STYLE3D_SIM_PASSWORD") or os.getenv("STYLE3D_PASSWORD")
    if not (account and password):
        if not sys.stdin.isatty():
            raise RuntimeError("SynReal needs login. Run in a terminal, or use --preview for the initial scene.")
        account = getpass.getpass("Style3D account (hidden): ")
        password = getpass.getpass("Style3D password (hidden): ")
    try:
        if not s3d.login(account, password, force):
            raise RuntimeError("SynReal login failed")
    finally:
        del account, password


def surface_texture():
    """A subtle material grid makes local compression easier to inspect."""
    rng = np.random.default_rng(0)
    y, x = np.mgrid[:512, :512]
    noise = rng.normal(0.0, 2.0, (512, 512, 1))
    texture = np.broadcast_to(np.array([239.0, 195.0, 109.0]), (512, 512, 3)).copy() + noise
    grid = ((x % 32) < 1) | ((y % 32) < 1)
    texture[grid] *= 0.88
    return texture.clip(0, 255).astype(np.uint8)


class SoftPadScene:
    def __init__(self, viewer, args):
        # env imports scene geometry; defer the reverse dependency until construction.
        from .env import MicroDuckEnv, environment_config, resolve_physics

        self.viewer, self.args = viewer, args
        profile = environment_config("mixed_soft_pad")
        if args.num_envs != 1:
            raise ValueError("scenes supports one interactive world; use examples.microduck.play --num-envs N")
        self.playback = None
        checkpoint = None
        if args.checkpoint is not None:
            torch.manual_seed(args.seed)
            checkpoint = load_policy_checkpoint(args.checkpoint, args.device, allow_recipe_mismatch=args.allow_recipe_mismatch)
        metadata = checkpoint.metadata if checkpoint is not None else None
        physics = resolve_physics(profile, metadata, use_checkpoint=args.use_checkpoint_environment, home=checkpoint is None)
        if args.substeps is not None:
            physics = physics.with_substeps(args.substeps)
            if checkpoint is not None and physics.physics_dt != checkpoint.metadata["physics_dt"]:
                print(f"Physics step override: replay dt={physics.physics_dt:g} s changes the policy dynamics.", flush=True)
        self.reset_requested = False
        self.reset_key = False
        pad = profile.pad
        if args.use_checkpoint_environment and metadata.get("terrain", {}).get("kind") == "mixed_soft_pad":
            pad = SoftPadConfig.from_checkpoint(metadata["terrain"])
        overrides = {
            name: getattr(args, flag)
            for name, flag in (
                ("size", "size"),
                ("cells", "cells"),
                ("density", "density"),
                ("shear_modulus", "mu"),
                ("lame_lambda", "lame_lambda"),
                ("damping", "damping"),
            )
            if getattr(args, flag) is not None
        }
        if args.cells_per_meter is not None:
            values = args.cells_per_meter
            if len(values) not in (1, 3):
                raise ValueError("--cells-per-meter requires one value or three axis values")
            overrides.update(cells=None, cells_per_meter=values[0] if len(values) == 1 else tuple(values))
        self.config = replace(pad, **overrides)
        if not args.preview:
            if not wp.get_device(args.device).is_cuda:
                raise ValueError("SynReal simulation requires CUDA; --preview also works on CPU")
            login(args.force_login)
        self.env = MicroDuckEnv(
            solver="synreal",
            num_envs=1,
            device=args.device,
            seed=None,
            viewer=viewer,
            task=metadata["task"] if metadata else "velocity",
            enable_self_collisions=metadata.get("self_collisions", False) if metadata else False,
            terrain="mixed_soft_pad",
            pad_config=self.config,
            physics=physics,
            interactive=True,
            controller="policy" if checkpoint is not None else "home",
            preview=args.preview,
            spawn=args.spawn,
            clearance=args.clearance,
            randomize_foot_friction=args.randomize_foot_friction,
        )
        self.env.cfg["environment_source"] = "checkpoint + CLI" if args.use_checkpoint_environment else "env.py + CLI"
        self.env.cfg["environment_profile"] = profile.name
        self.model, self.solver, self.control = self.env.model, self.env.solver, self.env.control
        self.root, self.robot_mass = self.env.root, self.env.robot_mass
        self.qi, self.qdi = self.env.qi.tolist(), self.env.qdi.tolist()
        self.foot_friction = self.env.cfg["foot_friction_randomization"]
        self.initial_state, self.initial_targets = self.env.initial_state, self.env.initial_targets
        self.tets = self.model.tet_indices.numpy()
        self.pad_indices = self.env.terrain.pad_indices[0].cpu().numpy()
        self.rest = self.env.terrain.rest.cpu().numpy().copy()
        self.fixed_indices = self.pad_indices[np.isclose(self.rest[self.pad_indices, 2], self.config.base_z)]
        self.top_indices = self.pad_indices[np.isclose(self.rest[self.pad_indices, 2], self.config.top_z)]
        self.rest_volumes = self.six_volumes(self.rest)
        if checkpoint is not None:
            self.playback = SoftPadPlayback(self.env, checkpoint, args.command)
        self.setup_rendering()
        mode = "INITIAL LAYOUT ONLY (no physics)" if args.preview else "SynReal / HOME joint targets (no walking policy)"
        if self.playback is not None:
            mode = f"SynReal / trained {self.env.task} policy on rigid + soft terrain"
            print(f"Policy dt={self.frame_dt:g} s; scene physics dt={self.dt:g} s (soft-terrain transfer).", flush=True)
            if self.env.family in ("velocity", "velstand"):
                print("I/K: forward/back; J/L: lateral; U/O: turn; X: zero velocity command.", flush=True)
        print(f"MicroDuck sponge pad: {mode}", flush=True)
        print(
            f"Environment: {self.env.cfg['environment_source']}, profile={profile.name}, num_envs=1, "
            f"spawn={args.spawn}, clearance={args.clearance:g}, "
            f"randomize_foot_friction={args.randomize_foot_friction}",
            flush=True,
        )
        print(f"Pad settings: {json.dumps(asdict(self.config), sort_keys=True)}", flush=True)
        print(
            f"Physics: dt={self.dt:g}, substeps={self.substeps}, "
            f"nonlinear={self.env.physics.nonlinear_iterations}, linear={self.env.physics.linear_iterations}",
            flush=True,
        )
        print(
            f"Pad={self.config.size} m, cells={self.config.resolved_cells}, "
            f"vertices={len(self.pad_indices)}, tetrahedra={len(self.tets)}",
            flush=True,
        )
        print("Terrain: rigid flat ground with a local sponge pad.", flush=True)
        print("Bottom bonded to the floor; R resets robot and pad. Space pauses; mouse/WASD moves the camera.", flush=True)

    def six_volumes(self, particles):
        p = particles[self.tets]
        return np.einsum("ij,ij->i", p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]))

    def setup_rendering(self):
        self.viewer.set_model(self.model)
        self.viewer.show_ground = True
        self.viewer.show_particles = False
        self.viewer.show_triangles = False
        self.set_camera(self.args.camera)
        renderer = getattr(self.viewer, "renderer", None)
        if renderer is not None:
            renderer.set_title(
                "MicroDuck | Flat ground + sponge pad | " + ("Initial layout" if self.args.preview else "SynReal")
            )
            renderer.sky_upper = (0.32, 0.39, 0.47)
            renderer.sky_lower = (0.65, 0.70, 0.75)
            renderer.ambient_sky = (0.85, 0.88, 0.93)
            renderer.ambient_ground = (0.42, 0.38, 0.30)
            renderer.exposure = 1.0
            renderer.shadow_radius = 0.01
            renderer.shadow_extents = max(2.0, *self.config.size[:2])
            self.viewer.register_ui_callback(self.gui, position="side")
        triangles = self.model.tri_indices.numpy()
        top_mask = np.isclose(self.rest[triangles, 2], self.config.top_z).all(axis=1)
        self.surfaces = []
        self.surface_pose = wp.array([wp.transform_identity()], dtype=wp.transform)
        self.surface_scale = wp.array([wp.vec3(1.0)], dtype=wp.vec3)
        self.surface_color = wp.array([wp.vec3(1.0)], dtype=wp.vec3)
        # The viewer exposes texture enablement through instance material.w.
        self.surface_material = wp.array([wp.vec4(0.95, 0.0, 0.0, 1.0)], dtype=wp.vec4)
        for name, faces, color in (
            ("pad_top", triangles[top_mask], (1.0, 1.0, 1.0)),
            ("pad_sides", triangles[~top_mask], (0.79, 0.57, 0.27)),
        ):
            indices, local = np.unique(faces, return_inverse=True)
            source = wp.array(indices.astype(np.int32), dtype=int)
            points = wp.empty(len(indices), dtype=wp.vec3)
            faces_wp = wp.array(local.ravel().astype(np.int32), dtype=int)
            uvs = None
            texture = None
            if name == "pad_top":
                # Keep each material grid square at 5 cm when the ground area changes.
                uv = self.rest[indices, :2] / 0.8 + 0.5
                uvs = wp.array(uv.astype(np.float32), dtype=wp.vec2)
                texture = surface_texture()
            self.surfaces.append((name, source, points, faces_wp, color, uvs, texture))

    def set_camera(self, view):
        if view == "overview":
            span = max(1.5, *self.config.size[:2])
            self.viewer.set_camera(pos=wp.vec3(0.8 * span, -0.9 * span, 0.8 * span), pitch=-33.0, yaw=132.0)
        else:
            self.viewer.set_camera(pos=wp.vec3(0.8, -1.0, 0.72), pitch=-26.0, yaw=129.0)

    @property
    def state_0(self):
        return self.env.states[0]

    @property
    def state_1(self):
        return self.env.states[1]

    @property
    def frame_dt(self):
        return self.env.step_dt

    @property
    def dt(self):
        return self.env.physics_dt

    @property
    def substeps(self):
        return self.env.decimation

    @property
    def sim_time(self):
        return self.env.sim_time

    @property
    def steps(self):
        return self.env.steps

    def reset(self):
        if self.playback is not None:
            self.playback.reset()
        else:
            self.env.reset()
        self.reset_requested = False

    def step(self):
        pressed = self.viewer.is_key_down("r")
        if self.reset_requested or (pressed and not self.reset_key):
            self.reset()
        self.reset_key = pressed
        if self.env.preview:
            return
        actions = self.playback.begin_frame() if self.playback is not None else None
        self.env.step(actions)
        if self.args.test or self.steps % 60 == 0:
            self.check_state()

    def check_state(self):
        self.env.check_scene_state()

    def metrics(self):
        particles = self.state_0.particle_q.numpy()
        bodies = self.state_0.body_q.numpy()
        materials = {name: getattr(self.model, f"shape_material_{name}").numpy() for name in ("ke", "kd", "mu")}
        shape_types = self.model.shape_type.numpy()
        rigid_contacts = {}
        for index, label in enumerate(self.model.shape_label):
            name = label.rsplit("/", 1)[-1]
            if shape_types[index] == newton.GeoType.PLANE:
                name = "ground"
            elif name not in ("left_foot_collision", "right_foot_collision"):
                continue
            rigid_contacts[name] = {key: float(values[index]) for key, values in materials.items()}
        return {
            "preview_only": self.args.preview,
            "environment_source": self.env.cfg["environment_source"],
            "controller": "trained Velocity policy"
            if self.playback is not None
            else "HOME joint targets; no learned walking policy",
            "policy": self.playback.metrics() if self.playback is not None else None,
            "spawn": self.args.spawn,
            "terrain": "rigid flat ground with a local sponge pad",
            "physics_frames": self.steps,
            "simulation_seconds": self.sim_time,
            "pad": asdict(self.config),
            "robot_mass_kg": self.robot_mass,
            "pad_vertices": len(self.pad_indices),
            "pad_tetrahedra": len(self.tets),
            "physics_dt": self.dt,
            "substeps": self.substeps,
            "foot_friction_randomization": self.foot_friction,
            "rigid_contact_materials": rigid_contacts,
            "soft_contact_material": {
                "ke": float(self.model.soft_contact_ke),
                "kd": float(self.model.soft_contact_kd),
                "mu": float(self.model.soft_contact_mu),
            },
            "solver_settings": None
            if self.solver is None
            else {
                name: getattr(self.solver.config, name)
                for name in (
                    "joint_constraint_primal_iterations",
                    "affine_body_block_preconditioner",
                    "nonlinear_iterations",
                    "linear_iterations",
                    "air_density",
                )
            },
            "max_top_displacement_down_m": float(np.max(self.rest[self.top_indices, 2] - particles[self.top_indices, 2])),
            "min_tet_volume_ratio": float(np.min(self.six_volumes(particles) / self.rest_volumes)),
            "trunk_height_m": float(bodies[self.root, 2]),
            "trunk_position_m": bodies[self.root, :3].tolist(),
            "trunk_tilt_deg": float(np.degrees(np.arccos(np.clip(1.0 - 2.0 * np.sum(bodies[self.root, 3:5] ** 2), -1.0, 1.0)))),
        }

    def gui(self, ui):
        ui.text("MICRODUCK / FLAT GROUND + SPONGE PAD")
        ui.separator()
        if self.playback is not None:
            ui.text(f"SynReal - {self.env.task} policy {self.playback.checkpoint.iteration}")
            if self.env.family in ("velocity", "velstand"):
                vx, vy, yaw = self.playback.command
                ui.text(f"Command: {vx:.2f}, {vy:.2f} m/s | {yaw:.2f} rad/s")
                ui.text("I/K: forward | J/L: lateral | U/O: turn | X: stop")
        else:
            ui.text("Initial layout - physics OFF" if self.args.preview else "SynReal - HOME pose hold")
        ui.text("Compressible elastic material")
        ui.text("Bottom face bonded to floor")
        ui.text(f"Size: {self.config.size[0]:.2f} x {self.config.size[1]:.2f} x {self.config.size[2]:.2f} m")
        ui.text(f"Robot mass: {self.robot_mass:.3f} kg")
        ui.text(f"Time: {self.sim_time:.2f} s")
        if ui.button("Reset robot and pad [R]"):
            self.reset_requested = True
        if ui.button("Robot view"):
            self.set_camera("close")
        ui.same_line()
        if ui.button("Terrain view"):
            self.set_camera("overview")
        ui.text("Space: pause | Mouse / WASD: camera")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        for name, source, points, faces, color, uvs, texture in self.surfaces:
            wp.launch(gather_surface, dim=len(source), inputs=[self.state_0.particle_q, source, points])
            self.viewer.log_mesh(
                name,
                points,
                faces,
                color=color,
                uvs=uvs,
                texture=texture,
                roughness=0.95,
                hidden=texture is not None,
            )
            if texture is not None:
                self.viewer.log_instances(
                    name + "_surface",
                    name,
                    self.surface_pose,
                    self.surface_scale,
                    self.surface_color,
                    self.surface_material,
                )
        self.viewer.end_frame()


def create_parser():
    from .env import environment_config

    profile = environment_config("mixed_soft_pad")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preview", action="store_true", help="Inspect the initial scene without physics or login")
    parser.add_argument("--checkpoint", type=Path, help="Task .pt checkpoint or a run directory (highest saved iteration)")
    parser.add_argument(
        "--use-checkpoint-environment",
        action="store_true",
        help="Restore recorded physics and pad materials instead of the current env.py settings",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=profile.playback_num_envs,
        help="Interactive scene requires 1; use examples.microduck.play for multiple worlds",
    )
    parser.add_argument("--command", type=float, nargs=3, default=(0.1, 0.0, 0.0), metavar=("VX", "VY", "YAW"))
    parser.add_argument(
        "--spawn", choices=("pad", "ground"), default=profile.spawn, help="Start on the pad or on the ground before it"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--randomize-foot-friction",
        action=argparse.BooleanOptionalAction,
        default=profile.playback_randomize_foot_friction,
        help="Use the same foot friction event as play.py",
    )
    parser.add_argument(
        "--allow-recipe-mismatch", action="store_true", help="Allow diagnostic playback with changed recipe files"
    )
    parser.add_argument("--viewer", choices=("gl", "null"), default="gl")
    parser.add_argument("--headless", action="store_true", help="Render OpenGL without opening a window")
    parser.add_argument("--camera", choices=("close", "overview"), default="overview")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-frames", type=int, default=36000)
    parser.add_argument("--screenshot", type=Path, help="Save the final OpenGL frame and a JSON report beside it")
    parser.add_argument("--size", type=float, nargs=3, metavar=("LENGTH", "WIDTH", "THICKNESS"))
    resolution = parser.add_mutually_exclusive_group()
    resolution.add_argument("--cells", type=int, nargs=3, metavar=("NX", "NY", "NZ"), help="override with fixed cell counts")
    resolution.add_argument(
        "--cells-per-meter",
        type=float,
        nargs="+",
        metavar="DENSITY",
        help="one value for all axes, or X Y Z values; counts scale with --size",
    )
    parser.add_argument("--mu", type=float, help="Shear modulus, Pa; default from env.py PAD_CONFIG")
    parser.add_argument("--lambda", type=float, dest="lame_lambda", help="Lame lambda, Pa")
    parser.add_argument("--density", type=float, help="Rest density, kg/m^3")
    parser.add_argument("--damping", type=float)
    parser.add_argument(
        "--clearance", type=float, default=profile.clearance, help="Additional robot spawn height above the pad, m"
    )
    parser.add_argument("--substeps", type=int, help="Override physics steps per frame; defaults are centralized in env.py")
    parser.add_argument("--force-login", action="store_true", help="Allow replacing another active SynReal session")
    parser.add_argument("--test", action="store_true", help="Check finite state, bonded base and positive tetrahedron volumes")
    return parser


def main(argv=None):
    parser = create_parser()
    args = parser.parse_args(argv)
    if args.num_envs != 1:
        parser.error("scenes requires --num-envs 1; use examples.microduck.play --num-envs N for batch playback")
    if args.use_checkpoint_environment and args.checkpoint is None:
        parser.error("--use-checkpoint-environment requires --checkpoint")
    if args.num_frames < 1 or (args.substeps is not None and args.substeps < 1):
        parser.error("num-frames and substeps must be positive")
    if not np.isfinite(args.clearance) or args.clearance < 0:
        parser.error("clearance must be finite and nonnegative")
    if args.screenshot is not None and args.viewer != "gl":
        parser.error("screenshot requires --viewer gl")
    if args.preview and args.checkpoint is not None:
        parser.error("checkpoint playback requires physics; remove --preview")
    if not np.isfinite(args.command).all():
        parser.error("command values must be finite")
    if args.allow_recipe_mismatch and args.checkpoint is None:
        parser.error("allow-recipe-mismatch requires --checkpoint")
    wp.set_device(args.device)
    from newton.viewer import ViewerGL, ViewerNull

    viewer = ViewerGL(width=1440, height=900, headless=args.headless) if args.viewer == "gl" else ViewerNull(args.num_frames)
    try:
        scene = SoftPadScene(viewer, args)
        screenshot_frame = None
        for _ in range(args.num_frames):
            if not viewer.is_running():
                break
            start = time.monotonic()
            if viewer.should_step():
                scene.step()
            scene.render()
            if args.screenshot is not None:
                # Keep a detached GPU image so closing the window cannot invalidate the last screenshot.
                screenshot_frame = viewer.get_frame(target_image=screenshot_frame, render_ui=False)
            if args.viewer == "gl" and not args.headless:
                time.sleep(max(0.0, scene.frame_dt - (time.monotonic() - start)))
        if args.test:
            scene.check_state()
        report = scene.metrics()
        if args.screenshot is not None and screenshot_frame is not None:
            from PIL import Image

            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(screenshot_frame.numpy()).save(args.screenshot)
            args.screenshot.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"Saved scene: {args.screenshot}", flush=True)
        print(json.dumps(report, indent=2), flush=True)
    finally:
        viewer.close()


if __name__ == "__main__":
    # Use one canonical module for Warp kernels and dataclass types under python -m.
    from examples.microduck.scenes import main as scene_main

    scene_main()
