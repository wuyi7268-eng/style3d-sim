"""Substep accelerometer for the cloth-only policy, sampled at the torso IMU site."""

import torch
import warp as wp

from examples.locomotion.math_utils import quat_rotate_inverse

from .adapter import rotate


def site_velocity(com_velocity, angular_velocity, body_quaternion, local_com, local_site):
    """Point velocity includes the lever arm from the body's center of mass to the IMU."""
    offset = rotate(body_quaternion, (local_site - local_com).expand_as(com_velocity))
    return com_velocity + torch.cross(angular_velocity, offset, dim=-1)


class ClothAccelerometer:
    def __init__(self, count, device, spec):
        self.spec = spec
        self.acceleration = torch.zeros(count, 3, device=device)
        self._sum = torch.zeros_like(self.acceleration)
        self._samples = 0
        self.local_site = self.acceleration.new_tensor(spec["site_position_body_m"])
        self.site_quaternion = self.acceleration.new_tensor(spec["site_quaternion_xyzw"]).expand(count, -1)
        self._previous_velocity = torch.zeros_like(self.acceleration)

    def reset(self, ids):
        # Reset is not a physical impulse; never difference velocities across a teleport.
        self.acceleration[ids] = 0
        self._sum[ids] = 0
        self._previous_velocity[ids] = 0

    def advance(self):
        self._sum.zero_()
        self._samples = 0
        self.acceleration.zero_()

    def sample(self, before, after, body_quaternion, gravity_world, dt, imu_rotation=None):
        """Average specific force in sensor axes, then scale and clip for policy input."""
        force_world = (after - before) / dt - gravity_world
        force_body = quat_rotate_inverse(body_quaternion, force_world)
        force_sensor = quat_rotate_inverse(self.site_quaternion, force_body)
        if imu_rotation is not None:
            force_sensor = rotate(imu_rotation, force_sensor)
        if not torch.isfinite(force_sensor).all():
            raise FloatingPointError("Non-finite cloth IMU specific force")
        self._sum.add_(force_sensor)
        self._samples += 1
        self.acceleration.copy_((self._sum / (self._samples * self.spec["scale_m_s2"])).clamp(
            -self.spec["clip_g"], self.spec["clip_g"]
        ))
        if not torch.isfinite(self.acceleration).all():
            raise FloatingPointError("Non-finite cloth IMU acceleration")

    def _state(self, env):
        env._sync()
        _, _, pose, velocity = env._views()
        quat = torch.nn.functional.normalize(pose[:, 3:7], dim=-1)
        com = wp.to_torch(env.model.body_com).reshape(env.num_envs, -1, 3)[:, 0]
        world_velocity = site_velocity(velocity[:, :3], velocity[:, 3:], quat, com, self.local_site)
        return world_velocity, quat

    def begin_substep(self, env):
        velocity, _ = self._state(env)
        self._previous_velocity.copy_(velocity)

    def end_substep(self, env):
        velocity, quat = self._state(env)
        worlds = wp.to_torch(env.model.body_world).reshape(env.num_envs, -1)[:, 0].long()
        gravity = wp.to_torch(env.model.gravity)[worlds]
        self.sample(self._previous_velocity, velocity, quat, gravity, env.physics_dt,
                    env.official.observations.imu_rotation)
