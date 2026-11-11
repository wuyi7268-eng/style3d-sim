"""Optional playback steering through policy commands, never through external forces."""

import math


def wrap_angle(value):
    return math.atan2(math.sin(value), math.cos(value))


class HeadingHold:
    """Follow the initial forward line using ground-truth pose; not a learned policy capability."""

    def __init__(self, pose, max_yaw=1.0):
        if not math.isfinite(max_yaw) or max_yaw <= 0:
            raise ValueError("max_yaw must be finite and positive")
        self.max_yaw = max_yaw
        self.origin = pose[:2]
        self.heading = self.yaw(pose)
        self.integral = 0.0

    @staticmethod
    def yaw(pose):
        x, y, z, w = pose[3:7]
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def update(self, pose, forward, dt):
        dx, dy = pose[0] - self.origin[0], pose[1] - self.origin[1]
        cross_track = -math.sin(self.heading) * dx + math.cos(self.heading) * dy
        correction = max(-0.35, min(0.35, math.atan2(-0.8 * cross_track, max(abs(forward), 0.1))))
        error = wrap_angle(self.heading + correction - self.yaw(pose))
        proposed = max(-1.5, min(1.5, self.integral + error * dt))
        output = 2.0 * error + 0.6 * proposed
        # Do not integrate further into saturation. The default bound matches the training command range.
        if abs(output) <= self.max_yaw or output * error < 0:
            self.integral = proposed
        return max(-self.max_yaw, min(self.max_yaw, 2.0 * error + 0.6 * self.integral))


class ProgressHold(HeadingHold):
    """Regulate measured progress and reject yaw drift through policy inputs only."""

    def __init__(self, pose, max_yaw=2.0, target_speed=0.10, nominal_forward=0.46):
        super().__init__(pose, max_yaw)
        if not math.isfinite(target_speed) or target_speed <= 0:
            raise ValueError("target_speed must be finite and positive")
        self.target_speed = target_speed
        if not math.isfinite(nominal_forward) or nominal_forward <= 0.06:
            raise ValueError("nominal_forward must be finite and exceed 0.06 m/s")
        self.nominal_forward = nominal_forward
        self.minimum_forward, self.maximum_forward = nominal_forward - 0.06, nominal_forward + 0.09
        self.previous_pose = list(pose)
        self.forward_speed = self.yaw_speed = 0.0
        self.speed_integral = 0.0
        self.forward_command = nominal_forward

    def update_commands(self, pose, dt):
        direction = (math.cos(self.heading), math.sin(self.heading))
        speed = sum((pose[i] - self.previous_pose[i]) * direction[i] for i in range(2)) / dt
        yaw_rate = wrap_angle(self.yaw(pose) - self.yaw(self.previous_pose)) / dt
        alpha = 1.0 - math.exp(-dt / 0.2)
        self.forward_speed += alpha * (speed - self.forward_speed)
        self.yaw_speed += alpha * (yaw_rate - self.yaw_speed)
        self.previous_pose = list(pose)
        dx, dy = pose[0] - self.origin[0], pose[1] - self.origin[1]
        cross_track = -direction[1] * dx + direction[0] * dy
        correction = max(-0.25, min(0.25, -0.5 * cross_track))
        error = wrap_angle(self.heading + correction - self.yaw(pose))
        proposed = max(-2.0, min(2.0, self.integral + error * dt))
        yaw = 5.0 * error + 1.2 * proposed - 0.7 * self.yaw_speed
        if abs(yaw) <= self.max_yaw or yaw * error < 0:
            self.integral = proposed
        yaw = max(-self.max_yaw, min(self.max_yaw, 5.0 * error + 1.2 * self.integral - 0.7 * self.yaw_speed))
        speed_error = self.target_speed - self.forward_speed
        proposed_speed = max(-0.4, min(0.4, self.speed_integral + speed_error * dt))
        forward = self.nominal_forward + 0.5 * speed_error + 0.15 * proposed_speed
        low, high = self.minimum_forward, self.maximum_forward
        if low <= forward <= high or (forward > high and speed_error < 0) or (forward < low and speed_error > 0):
            self.speed_integral = proposed_speed
        forward = max(low, min(high, forward))
        # Slew limiting prevents the command loop from injecting step changes into the policy history.
        change = max(-0.10 * dt, min(0.10 * dt, forward - self.forward_command))
        self.forward_command += change
        return self.forward_command, yaw
