"""Free basketball and the upstream assisted-balance curriculum.

Task formulas adapted from Vottivott/microduck-playground (Apache-2.0),
Copyright 2026 Hannes von Essen / Vottivott.
commit 052a2494dd6372289221c7a4f24a245b443dd75f. Physics uses the local PD robot.
"""

import math

import newton
import warp as wp

from .policy_spec import build_robot, filter_authored_collider


def build_basketball_robot(builder_type, config):
    robot, qi, qdi = build_robot(builder_type, task="velstand")
    geometry = config["geometry"]
    radius = geometry["radius"]
    robot.joint_q[2] = 2 * radius + geometry["root_height"] + 0.003
    ball_q = len(robot.joint_q)
    ball = robot.add_link(xform=wp.transform((0, 0, radius + 0.001), wp.quat_identity()), label="basketball")
    joint = robot.add_joint_free(child=ball, label="basketball_free")
    robot.add_articulation([joint], label="basketball")
    cfg = robot.default_shape_cfg.copy()
    cfg.density = geometry["mass"] / (4 / 3 * math.pi * radius**3)
    cfg.mu, cfg.mu_torsional, cfg.mu_rolling = geometry["friction"]
    shape = robot.add_shape_sphere(ball, radius=radius, cfg=cfg, color=(0.91, 0.42, 0.12), label="basketball_sphere",
                                  custom_attributes={"mujoco:geom_priority": 1, "mujoco:condim": 4,
                                                     "mujoco:contype": 1, "mujoco:conaffinity": 1})
    filter_authored_collider(robot, shape)
    # Seams are visual only: they must not alter the spherical collision or inertia.
    visual = newton.ModelBuilder.ShapeConfig(density=0, has_shape_collision=False, has_particle_collision=False)
    for axis in range(3):
        for segment in range(32):
            points = []
            for i in (segment, segment + 1):
                point = [0.0, 0.0, 0.0]
                point[(axis + 1) % 3] = radius * math.cos(i * math.tau / 32)
                point[(axis + 2) % 3] = radius * math.sin(i * math.tau / 32)
                points.append(wp.vec3(*point))
            delta = points[1] - points[0]
            orientation = wp.quat_between_vectors(wp.vec3(0, 0, 1), wp.normalize(delta))
            robot.add_shape_capsule(ball, radius=0.0012, half_height=float(wp.length(delta)) / 2,
                                    xform=wp.transform((points[0] + points[1]) * 0.5, orientation), cfg=visual,
                                    color=(0.055, 0.035, 0.02), label=f"basketball_seam_{axis}_{segment}",
                                    custom_attributes={"mujoco:contype": 0, "mujoco:conaffinity": 0})
    return robot, qi, qdi, ball, ball_q


@wp.kernel
def hold_forces(
    poses: wp.array(dtype=wp.transform),
    velocities: wp.array(dtype=wp.spatial_vector),
    com: wp.array(dtype=wp.vec3),
    forces: wp.array(dtype=wp.spatial_vector),
    hold: wp.array(dtype=float),
    bodies_per_world: int,
    root_index: int,
    ball_index: int,
    ball_kp: float,
    ball_kd: float,
    ball_kd_rot: float,
    duck_kp: float,
    duck_kd: float,
    duck_kp_rot: float,
    duck_kd_rot: float,
):
    world = wp.tid()
    root = world * bodies_per_world + root_index
    ball = world * bodies_per_world + ball_index
    bpos = wp.transform_get_translation(poses[ball])
    bvel = wp.spatial_top(velocities[ball])
    bforce = -ball_kp * wp.vec3(bpos[0], bpos[1], 0.0) - ball_kd * bvel
    btorque = -ball_kd_rot * wp.spatial_bottom(velocities[ball])
    forces[ball] += hold[world] * wp.spatial_vector(bforce, btorque)
    quat = wp.transform_get_rotation(poses[root])
    omega = wp.spatial_bottom(velocities[root])
    # The spring targets the link origin; Newton stores velocity at the COM.
    offset = wp.quat_rotate(quat, com[root])
    velocity = wp.spatial_top(velocities[root]) - wp.cross(omega, offset)
    delta = wp.transform_get_translation(poses[root]) - bpos
    force = -duck_kp * delta - duck_kd * velocity
    force[2] = 0.0
    up = wp.quat_rotate(quat, wp.vec3(0.0, 0.0, 1.0))
    torque = duck_kp_rot * wp.cross(up, wp.vec3(0.0, 0.0, 1.0)) - duck_kd_rot * omega
    forces[root] += hold[world] * wp.spatial_vector(force, torque)
