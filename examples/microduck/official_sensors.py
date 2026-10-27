"""Contact flags and rigid-body sensors for the flat-terrain official recipe.

These sensors consume geometric contacts, never the unpopulated contact-force buffer.
"""

from types import SimpleNamespace

import torch
import warp as wp

from examples.locomotion.math_utils import quat_rotate_inverse

from .tasks import rotate


class TerrainHeightSensor:
    """Flat infinite plane specialization of mjlab's terrain-height sensor."""

    num_frames = 2

    def __init__(self):
        self.data = SimpleNamespace()


def whole_body_angular_momentum(pose, velocity, local_com, mass, inertia):
    """World-frame subtree angular momentum about its aggregate center of mass."""
    quat = pose[..., 3:]
    com = pose[..., :3] + rotate(quat, local_com)
    center = (com * mass[..., None]).sum(1) / mass.sum(1, keepdim=True)
    omega_local = quat_rotate_inverse(quat, velocity[..., 3:])
    spin = rotate(quat, (inertia @ omega_local[..., None]).squeeze(-1))
    orbital = torch.cross(com - center[:, None], mass[..., None] * velocity[..., :3], dim=-1)
    return (spin + orbital).sum(1)


class ContactSensor:
    def __init__(self, num_envs, num_primary, device):
        self.data = SimpleNamespace(found=torch.zeros(num_envs, num_primary, device=device), force_history=None)
        for name in ("current_air_time", "last_air_time", "current_contact_time", "last_contact_time"):
            setattr(self.data, name, torch.zeros_like(self.data.found))

    def reset(self, ids):
        for value in vars(self.data).values():
            if isinstance(value, torch.Tensor):
                value[ids] = 0

    def update(self, found, dt):
        data = self.data
        contact = found > 0
        data.found.copy_(found)
        data.last_air_time.copy_(
            torch.where(contact & (data.current_air_time > 0), data.current_air_time + dt, data.last_air_time)
        )
        data.last_contact_time.copy_(
            torch.where(~contact & (data.current_contact_time > 0), data.current_contact_time + dt, data.last_contact_time)
        )
        data.current_air_time.copy_(torch.where(contact, 0.0, data.current_air_time + dt))
        data.current_contact_time.copy_(torch.where(contact, data.current_contact_time + dt, 0.0))

    def compute_first_contact(self, dt, abs_tol=1.0e-8):
        time = self.data.current_contact_time
        return (time > 0) & (time < dt + abs_tol)


@wp.kernel
def _contact_flags(
    count: wp.array[int],
    shape0: wp.array[int],
    shape1: wp.array[int],
    point0: wp.array[wp.vec3],
    point1: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    margin0: wp.array[float],
    margin1: wp.array[float],
    shape_body: wp.array[int],
    shape_world: wp.array[int],
    shape_role: wp.array[int],
    body_q: wp.array[wp.transform],
    feet: wp.array2d[float],
    self_contact: wp.array2d[float],
    robot_ground: wp.array2d[float],
    head_ground: wp.array2d[float],
):
    i = wp.tid()
    if i >= count[0]:
        return
    s0, s1 = shape0[i], shape1[i]
    if s0 < 0 or s1 < 0:
        return
    b0, b1 = shape_body[s0], shape_body[s1]
    p0_local, p1_local = point0[i], point1[i]
    if b0 >= 0:
        p0_world = wp.transform_point(body_q[b0], p0_local)
    else:
        p0_world = p0_local
    if b1 >= 0:
        p1_world = wp.transform_point(body_q[b1], p1_local)
    else:
        p1_world = p1_local
    # Collision generation includes speculative contacts; do not count a positive gap as touchdown.
    gap = wp.dot(p1_world - p0_world, normal[i]) - margin0[i] - margin1[i]
    if gap > 0.0:
        return
    w0, w1 = shape_world[s0], shape_world[s1]
    r0, r1 = shape_role[s0], shape_role[s1]
    if w0 >= 0 and r1 == -1 and r0 != -2:
        wp.atomic_add(robot_ground, w0, 0, 1.0)
        if r0 == -3:
            wp.atomic_add(head_ground, w0, 0, 1.0)
        if r0 == 1 or r0 == 2:
            wp.atomic_add(feet, w0, r0 - 1, 1.0)
    if w1 >= 0 and r0 == -1 and r1 != -2:
        wp.atomic_add(robot_ground, w1, 0, 1.0)
        if r1 == -3:
            wp.atomic_add(head_ground, w1, 0, 1.0)
        if r1 == 1 or r1 == 2:
            wp.atomic_add(feet, w1, r1 - 1, 1.0)
    # Role -2 excludes the separately articulated ball; -3 marks the head body.
    if w0 >= 0 and w0 == w1 and r0 != -2 and r1 != -2 and r0 != -1 and r1 != -1:
        wp.atomic_add(self_contact, w0, 0, 1.0)


class OfficialSensors:
    def __init__(self, env, robot):
        self.env = env
        self.robot_body_count = env.task_logic.ball_body if env.task.startswith("ballkick") else len(robot.body_mass)
        self.feet = ContactSensor(env.num_envs, 2, env.device)
        self.support = ContactSensor(env.num_envs, 1, env.device)
        self.self_contact = ContactSensor(env.num_envs, 1, env.device)
        self.robot_ground = ContactSensor(env.num_envs, 1, env.device)
        self.head_ground = ContactSensor(env.num_envs, 1, env.device)
        self.height = TerrainHeightSensor()
        self.feet_found = torch.zeros_like(self.feet.data.found)
        self.self_found = torch.zeros_like(self.self_contact.data.found)
        self._feet_wp = wp.from_torch(self.feet_found)
        self._self_wp = wp.from_torch(self.self_found)
        body_labels = [label.rsplit("/", 1)[-1] for label in robot.body_label]
        ankle_ids = {}
        for side in ("left", "right"):
            matches = [i for i, name in enumerate(body_labels) if name in (f"ankle_{side[0]}_v1", f"{side}_ankle")]
            ankle_ids[side] = set(matches)
        roles = []
        for shape, label in enumerate(robot.shape_label):
            name = label.rsplit("/", 1)[-1]
            body = robot.shape_body[shape]
            role = -2 if body < 0 or body >= self.robot_body_count else 0
            if body >= 0 and body < self.robot_body_count:
                body_name = body_labels[body]
                if body_name == "jaw_soft":
                    role = -3
                elif body in ankle_ids["left"] or name == "left_foot_collision" or "/ankle_l_v1/" in robot.body_label[body]:
                    role = 1
                elif body in ankle_ids["right"] or name == "right_foot_collision" or "/ankle_r_v1/" in robot.body_label[body]:
                    role = 2
            roles.append(role)
        # One shared world collider is visible to each replicated physics world.
        roles = roles * env.num_envs + [-1] * env.terrain.shape_count
        if len(roles) != env.model.shape_count:
            raise ValueError("Official contact sensors require replicated robot shapes and shared terrain shapes")
        self.roles = wp.array(roles, dtype=wp.int32, device=str(env.device))

    def reset(self, ids):
        for sensor in (self.feet, self.support, self.self_contact, self.robot_ground, self.head_ground):
            sensor.reset(ids)

    def update(self):
        env = self.env
        contacts = env.contacts
        self.feet_found.zero_()
        self.self_found.zero_()
        robot_found = torch.zeros_like(self.robot_ground.data.found)
        head_found = torch.zeros_like(self.head_ground.data.found)
        env._sync()
        wp.launch(
            _contact_flags,
            dim=contacts.rigid_contact_max,
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
                env.model.shape_body,
                env.model.shape_world,
                self.roles,
                env.states[0].body_q,
                self._feet_wp,
                self._self_wp,
                wp.from_torch(robot_found),
                wp.from_torch(head_found),
            ],
            device=str(env.device),
        )
        env._sync()
        # mjlab's found field counts matches before reduction, including netforce sensors.
        self.feet.update(self.feet_found, env.physics_dt)
        self.support.update(self.feet.data.found[:, :1], env.physics_dt)
        self.self_contact.update(self.self_found, env.physics_dt)
        self.robot_ground.update(robot_found, env.physics_dt)
        self.head_ground.update(head_found, env.physics_dt)
