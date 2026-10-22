"""Finite-but-exploded states must be caught before squared rewards overflow."""

import torch

from examples.microduck.stability import unstable_worlds


def test_detects_finite_explosion_and_nonfinite_per_environment():
    pose = torch.zeros(5, 3, 7)
    pose[:, :, 6] = 1
    velocity = torch.zeros(5, 3, 6)
    velocity[1, 1, 0] = 1e7
    velocity[2, 2, 4] = 1e9
    pose[3, 0, 2] = float("nan")
    pose[4, 2, 0] = 1e6
    assert unstable_worlds(pose, velocity).tolist() == [False, True, True, True, True]


def test_fallen_pose_and_large_but_plausible_recovery_motion_are_not_rejected():
    pose = torch.zeros(2, 4, 7)
    pose[:, :, 3] = 1
    pose[:, :, 2] = 0.03
    velocity = torch.ones(2, 4, 6) * 10
    velocity[:, :, 3:] = 100
    assert not unstable_worlds(pose, velocity).any()
