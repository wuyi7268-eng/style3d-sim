"""CPU-only checks that visual verdicts cannot qualify unrelated simulation evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.summarize_cable_profiles import digest, load_reviews, validated_run


@pytest.fixture
def evidence(tmp_path: Path):
    identity = {field: field for field in ("native_sha256", "adapter_sha256", "helper_sha256", "benchmark_sha256")}
    directory = tmp_path / "scene" / "synreal"
    directory.mkdir(parents=True)
    config = {"nonlinear_iterations": 1, "linear_iterations": 2, "inertia_mode": "mode"}
    metadata = {
        **identity, "source_sha256": "source", "config": config, "completed": True, "finite": True,
        "frames_requested": 30, "frames_completed": 30, "body_labels": ["rod_link_0"],
        "example_args": [], "substeps": 8, "sample_stride": 30, "median_step_ms": 1.0,
        "passed": False, "assertion_failures": ["Authored hook deliberately fails"],
    }
    poses = np.zeros((2, 1, 7))
    poses[..., 6] = 1.0
    for name in ("upstream", "baseline"):
        (directory / f"{name}.json").write_text(json.dumps(metadata), encoding="utf-8")
        np.savez(directory / f"{name}.npz", frames=[0, 30], poses=poses)
    job = {
        "scene": "scene", "method": "synreal", "source_sha256": "source", "frames": 30,
        "expected_non_budget_config": {"inertia_mode": "mode"},
        "reference_files": {suffix: {"sha256": digest(directory / f"upstream{suffix}")}
                            for suffix in (".json", ".npz")},
    }
    return tmp_path, directory, job, identity


def test_metrics_and_hook_flags_do_not_create_a_visual_verdict(evidence):
    root, _, job, identity = evidence
    row = validated_run(root, job, "baseline", [1, 2], identity, {})
    assert row["visually_accepted"] is None and row["status"] == "unreviewed"
    assert row["authored_hooks_passed"] is False and len(row["assertion_failures"]) == 1


def test_stale_visual_review_is_rejected(evidence):
    root, directory, job, identity = evidence
    row = validated_run(root, job, "baseline", [1, 2], identity, {})
    review = {key: row[key] for key in ("metadata_sha256", "trajectory_sha256",
                                       "reference_metadata_sha256", "reference_trajectory_sha256")}
    review.update(config=row["config"], qualifies_user_visual_standard=True, visual_status="accepted")
    review["trajectory_sha256"] = "stale"
    with pytest.raises(ValueError, match="Visual review trajectory_sha256 mismatch"):
        validated_run(root, job, "baseline", [1, 2], identity,
                      {("scene", "synreal", "baseline"): (directory / "review.json", review)})


def test_nonfinite_trajectory_cannot_rely_on_metadata_finite_flag(evidence):
    root, directory, job, identity = evidence
    with np.load(directory / "baseline.npz") as trajectory:
        poses = trajectory["poses"].copy()
    poses[-1, 0, 0] = np.nan
    np.savez(directory / "baseline.npz", frames=[0, 30], poses=poses)
    with pytest.raises(ValueError, match="finiteness"):
        validated_run(root, job, "baseline", [1, 2], identity, {})


def test_visual_review_requires_an_explicit_boolean(tmp_path: Path):
    path = tmp_path / "review.json"
    path.write_text(json.dumps({"entries": [{"scene": "scene", "method": "synreal",
                                            "role": "baseline", "qualifies_user_visual_standard": "true"}]}))
    with pytest.raises(ValueError, match="boolean visual verdict"):
        load_reviews([path])
