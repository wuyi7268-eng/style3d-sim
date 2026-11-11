"""SynReal adaptation of the retained seat at Vottivott/microduck-playground c5fcc50.

Copyright 2026 Pollen Robotics. Apache-2.0; see assets/swing/LICENSE.
Modified: reuse local robot/PD, and replace MuJoCo tendons with unilateral cord forces.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import torch
import warp as wp

from .policy_spec import ASSET, build_robot

ASSETS = Path(__file__).with_name("assets") / "swing"


def build_swing_robot(builder_type, geometry, joint_reference):
    spec = mujoco.MjSpec.from_file(str(ASSET.with_name("robot_allcollisions.xml")))
    # Serialized temporary MJCF must still resolve the existing robot meshes.
    mesh_root = ASSET.parent / spec.meshdir
    for mesh in spec.meshes:
        if mesh.file:
            mesh.file = str((mesh_root / mesh.file).resolve())
    spec.meshdir = ""
    top = geometry["anchor_height"] + 0.005
    for side in (-1, 1):
        for front in (-1, 1):
            spec.worldbody.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=(0.007,),
                fromto=(front * 0.34, side * 0.34, 0, 0, side * 0.25, top),
                contype=0, conaffinity=0, rgba=(0.28, 0.18, 0.11, 1),
            )
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=(0.009,),
                            fromto=(0, -0.27, top, 0, 0.27, top), contype=0, conaffinity=0,
                            rgba=(0.28, 0.18, 0.11, 1))
    trunk = spec.body("trunk_base")
    payload = trunk.add_body(name="swing_seat_payload")
    parts = [("seat_retained", 0.150), ("retention_strap", 0.004), ("retention_buckle", 0.002)]
    parts += [(f"retention_bumper_{i}", 0.002 / 3) for i in range(3)]
    for name, mass in parts:
        name = "swing_" + name
        spec.add_mesh(name=name, file=str(ASSETS / (name + ".stl")))
        # Upstream retention is rigidly attached; meshes contribute mass/inertia but no contact.
        payload.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, mass=mass,
                         contype=0, conaffinity=0, group=2, rgba=(0.09, 0.10, 0.11, 1.0))
    with TemporaryDirectory(prefix="microduck-swing-") as directory:
        path = Path(directory) / "robot.xml"
        path.write_text(spec.to_xml())
        robot, qi, qdi = build_robot(builder_type, asset=path, task="velstand")
    robot.joint_q[2] = geometry["anchor_height"] - geometry["hang_length"] - geometry["attachment_z"]
    for index, value in zip(qi, joint_reference, strict=True):
        robot.joint_q[index] = value
        robot.joint_target_q[index] = value
    return robot, qi, qdi


def cord_tension(length, geometry):
    """Both elastic regions are tension-only; the upper limit is a compliant penalty."""
    return (geometry["stiffness"] * (length - geometry["length"]).clamp_min(0)
            + geometry["limit_stiffness"] * (length - geometry["limit"]).clamp_min(0))


@wp.kernel
def apply_cords(
    poses: wp.array(dtype=wp.transform),
    com: wp.array(dtype=wp.vec3),
    forces: wp.array(dtype=wp.spatial_vector),
    bodies_per_world: int,
    root: int,
    y: float,
    z: float,
    anchor_height: float,
    rest: float,
    stiffness: float,
    limit: float,
    limit_stiffness: float,
):
    body = wp.tid() * bodies_per_world + root
    pose = poses[body]
    center = wp.transform_point(pose, com[body])
    force, torque = wp.vec3(0.0), wp.vec3(0.0)
    for side in range(2):
        offset = y * (2.0 * float(side) - 1.0)
        attach = wp.transform_point(pose, wp.vec3(0.0, offset, z))
        delta = wp.vec3(0.0, offset, anchor_height) - attach
        length = wp.length(delta)
        tension = stiffness * wp.max(length - rest, 0.0) + limit_stiffness * wp.max(length - limit, 0.0)
        pull = delta * (tension / wp.max(length, 1.0e-8))
        force += pull
        torque += wp.cross(attach - center, pull)
    # Newton's spatial convention is linear force followed by torque about the COM.
    forces[body] += wp.spatial_vector(force, torque)


def mechanism_state(position, quaternion, linear, angular, geometry, horizon=0.08):
    from .adapter import rotate

    local = position.new_tensor([[0, geometry["attachment_y"], geometry["attachment_z"]],
                                 [0, -geometry["attachment_y"], geometry["attachment_z"]]])
    offsets = rotate(quaternion[:, None, :].expand(-1, 2, -1), local.expand(len(position), -1, -1))
    attach = position[:, None] + offsets
    velocity = linear[:, None] + torch.cross(angular[:, None].expand_as(offsets), offsets, dim=-1)
    anchors = local.clone()
    anchors[:, 2] = geometry["anchor_height"]
    rope = attach.mean(1) - anchors.mean(0)
    mid_velocity = velocity.mean(1)
    lengths = (attach - anchors).norm(dim=-1)
    span = attach[:, 0] - attach[:, 1]
    predicted = span + horizon * (velocity[:, 0] - velocity[:, 1])
    body_y = span / span.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return {
        "angle": torch.atan2(rope[:, 0], -rope[:, 2]),
        "rate": (-rope[:, 2] * mid_velocity[:, 0] + rope[:, 0] * mid_velocity[:, 2])
                / (rope[:, 0].square() + rope[:, 2].square()).clamp_min(1e-8),
        "lateral": rope[:, 1], "lengths": lengths,
        "slack": (geometry["length"] - lengths - 0.0005).clamp_min(0).amax(-1),
        "imbalance": (lengths[:, 0] - lengths[:, 1]).abs(),
        "alignment": 1 - body_y[:, 1].clamp(-1, 1),
        "predicted_alignment": 1 - (predicted[:, 1] / predicted.norm(dim=-1).clamp_min(1e-8)).clamp(-1, 1),
        "cue": torch.stack((torch.zeros_like(rope[:, 0]), body_y[:, 0], body_y[:, 2]), -1),
        "linear": linear, "angular": angular, "attach": attach, "anchors": anchors,
    }
