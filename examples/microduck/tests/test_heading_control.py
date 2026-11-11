"""Steering signs, heading wrapping, and command bounds for optional visual playback."""

import math

import pytest

from examples.microduck.heading_control import HeadingHold, ProgressHold


def pose(x=0, y=0, yaw=0):
    return [x, y, 0.2, 0, 0, math.sin(yaw / 2), math.cos(yaw / 2)]


def test_heading_and_lateral_errors_steer_toward_initial_line():
    assert HeadingHold(pose()).update(pose(), 0.46, 0.02) == 0
    assert HeadingHold(pose()).update(pose(yaw=0.2), 0.46, 0.02) < 0
    assert HeadingHold(pose()).update(pose(y=0.1), 0.46, 0.02) < 0
    assert HeadingHold(pose()).update(pose(y=-0.1), 0.46, 0.02) > 0


def test_heading_wrap_takes_short_path():
    controller = HeadingHold(pose(yaw=math.pi - 0.01))
    assert -0.1 < controller.update(pose(yaw=-math.pi + 0.01), 0.46, 0.02) < 0


def test_saturation_does_not_wind_up_and_reset_clears_history():
    controller = HeadingHold(pose())
    for _ in range(1000):
        assert controller.update(pose(yaw=-2), 0.46, 0.02) == 1
    assert controller.integral == 0
    assert controller.update(pose(), 0.46, 0.02) == pytest.approx(0)
    assert HeadingHold(pose(2, 1, 0.5)).update(pose(2, 1, 0.5), 0.46, 0.02) == 0


def test_progress_feedback_increases_request_when_stalled_and_limits_changes():
    control = ProgressHold(pose())
    previous = control.forward_command
    for _ in range(200):
        forward, yaw = control.update_commands(pose(), 0.02)
        assert 0.4 <= forward <= 0.55
        assert abs(forward - previous) <= 0.002 + 1e-12
        assert yaw == 0
        previous = forward
    assert forward > 0.46


def test_progress_feedback_reduces_request_when_speed_exceeds_target():
    control = ProgressHold(pose())
    for step in range(1, 201):
        forward, yaw = control.update_commands(pose(x=step * 0.004), 0.02)
    assert forward < 0.46
    assert yaw == 0


def test_yaw_rate_feedback_opposes_drift_before_large_heading_error():
    control = ProgressHold(pose())
    forward, yaw = control.update_commands(pose(yaw=-0.02), 0.02)
    assert 0 < yaw <= 2
    assert forward >= 0.46
