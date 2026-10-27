"""Asset mapping and opt-in real-GPU checks for both MicroDuck physics backends."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import newton

sys.path.insert(0, str(Path(__file__).parents[3]))
from examples.microduck.policy_spec import ASSET, HOME, JOINT_NAMES, build_robot


@pytest.mark.parametrize("task", ["velocity", "velstand", "groundpick", "ballkick"])
def test_full_self_collision_matches_mjcf_masks_and_exclusions(task):
    reference = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverMuJoCo.register_custom_attributes(reference)
    reference.add_mjcf(
        str(ASSET.with_name("robot_allcollisions.xml")),
        floating=True,
        collapse_fixed_joints=True,
        enable_self_collisions=True,
    )
    robot, _, _ = build_robot(task=task, enable_self_collisions=True)
    np.testing.assert_array_equal(robot.shape_flags, reference.shape_flags)
    assert robot.shape_collision_filter_pairs == reference.shape_collision_filter_pairs
    for name in ("mujoco:contype", "mujoco:conaffinity"):
        actual, expected = robot.custom_attributes[name], reference.custom_attributes[name]
        assert actual.default == expected.default
        assert actual.values == expected.values


def test_servo_mapping_on_actual_asset():
    robot, qi, qdi = build_robot()
    assert len(set(qi)) == len(set(qdi)) == len(JOINT_NAMES) == 14
    assert len(robot.joint_q) == 21
    assert len(robot.joint_qd) == 20
    np.testing.assert_allclose(np.asarray(robot.joint_q)[qi], HOME)
    assert all(robot.joint_effort_limit[i] == pytest.approx(0.96) for i in qdi)


def test_self_collision_allows_left_right_foot_contact():
    for enabled in (False, True):
        robot, _, _ = build_robot(enable_self_collisions=enabled)
        left = next(i for i, name in enumerate(robot.shape_label) if "left_foot_collision" in name)
        right = next(i for i, name in enumerate(robot.shape_label) if "right_foot_collision" in name)
        pair = (min(left, right), max(left, right))
        assert (pair in robot.shape_collision_filter_pairs) is not enabled


def test_cli_defaults_to_official_recipe():
    from examples.microduck.common import setup

    args, _, _ = setup("train", [])
    assert args.recipe == "official_adapted_v1"
    with pytest.raises(SystemExit):
        setup("train", ["--recipe", "legacy"])


def test_legacy_checkpoint_is_rejected(tmp_path):
    import json

    from examples.microduck.common import setup

    checkpoint = tmp_path / "model_0.pt"
    checkpoint.touch()
    checkpoint.with_name("config.json").write_text(json.dumps({"observation_contract": "shared_position_pd_v1"}))
    with pytest.raises(SystemExit):
        setup("play", ["--checkpoint", str(checkpoint)])
