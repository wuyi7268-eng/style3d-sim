"""Robot asset, servo order and position-PD task constants."""

from pathlib import Path

import newton

from .task_registry import ROLLER_TASKS, base_task

ASSET = (
    Path(__file__).resolve().parents[1] / "microduck/microduck_rl-develop/src/mjlab_microduck/robot/microduck/robot_walk.xml"
)
JOINT_NAMES = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)
HOME = (0.0, -0.0873, -0.4579, -0.0049, 0.4530, 0.3491, 0.3491, 0.0, 0.0, 0.0, 0.0873, 0.4579, 0.0049, -0.4530)
# The XML gain (0.55) cannot hold HOME under gravity; use the gain measured by the hold test.
KP, KD, EFFORT = 2.0, 0.053, 0.96


def joint_coordinates(robot):
    """Recover scalar MJCF names from Newton's combined same-body hinge labels."""
    coordinates = {}
    for joint, label in enumerate(robot.joint_label):
        name = label.rsplit("/", 1)[-1]
        names = name.split("_passive_", 1)
        if len(names) == 2:
            names[1] = "passive_" + names[1]
        for offset, component in enumerate(names):
            coordinates[component] = (robot.joint_q_start[joint] + offset, robot.joint_qd_start[joint] + offset)
    return coordinates


def build_robot(builder_type=newton.ModelBuilder, *, enable_self_collisions=False, task="velocity"):
    robot = builder_type(up_axis=newton.Axis.Z)
    # Preserve ground/ball filtering independently of robot self-collision detection.
    newton.solvers.SolverMuJoCo.register_custom_attributes(robot)
    # Both backends import the full collision asset and its authored MuJoCo masks/excludes.
    # Visual duplicates stay non-colliding; enabling them would double the contact geometry.
    family = base_task(task)
    if family in ROLLER_TASKS:
        asset = ASSET.with_name("robot_groundcontact_rollers.xml")
    elif enable_self_collisions or family in ("velstand", "ballkick"):
        asset = ASSET.with_name("robot_allcollisions.xml")
    else:
        asset = ASSET if family == "velocity" else ASSET.with_name("robot_groundcontact.xml")
    if task.endswith("_backlash"):
        asset = asset.with_name(asset.stem + "_backlash.xml")
    robot.add_mjcf(str(asset), floating=True, collapse_fixed_joints=True, enable_self_collisions=enable_self_collisions)
    # Use the same convex collision representation on both backends.
    robot.approximate_meshes("convex_hull")
    by_name = joint_coordinates(robot)
    if not set(JOINT_NAMES).issubset(by_name):
        raise ValueError("MicroDuck asset does not contain all 14 servo joints")
    q_indices, qd_indices = [], []
    for name, home in zip(JOINT_NAMES, HOME, strict=True):
        q, qd = by_name[name]
        q_indices.append(q)
        qd_indices.append(qd)
        robot.joint_q[q] = home
        robot.joint_target_q[q] = home
        robot.joint_target_ke[qd] = KP
        robot.joint_target_kd[qd] = KD
        robot.joint_effort_limit[qd] = EFFORT
        robot.joint_target_mode[qd] = int(newton.JointTargetMode.POSITION)
    # The exported walk model has a free trunk rooted at 12 cm, not robot total height.
    robot.joint_q[:7] = [0.0, 0.0, 0.12, 0.0, 0.0, 0.0, 1.0]
    return robot, q_indices, qd_indices


def filter_authored_collider(builder, collider):
    """Extend imported MJCF masks to a collider added after the XML import."""
    attributes = builder.custom_attributes
    if "mujoco:contype" not in attributes:
        return
    type_attr, affinity_attr = attributes["mujoco:contype"], attributes["mujoco:conaffinity"]

    def masks(shape):
        values = []
        for attr in (type_attr, affinity_attr):
            value = int(attr.values.get(shape, attr.default))
            # Newton's unset integer sentinel is not a collision mask. Authored
            # ground/ball shapes use MuJoCo's default category and affinity (1).
            if value == -(1 << 63):
                value = 1
                attr.values[shape] = value
            values.append(value)
        return tuple(values)

    ctype, affinity = masks(collider)
    for shape in range(collider):
        other_type, other_affinity = masks(shape)
        if not ((ctype & other_affinity) or (other_type & affinity)):
            builder.add_shape_collision_filter_pair(shape, collider)
