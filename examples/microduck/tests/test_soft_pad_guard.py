"""Numerically invalid worlds must terminate without contaminating neighbors or PPO."""

import json
from types import SimpleNamespace

import pytest
import torch
import warp as wp

from examples.microduck.scenes import SoftPadGuard
from examples.microduck.recipe import SNAPSHOT, OfficialRecipe
from examples.microduck.tests.test_mixed_terrain import terrain_fixture


def fixture(tmp_path):
    terrain, env = terrain_fixture()
    env.terrain = terrain
    env.solver = SimpleNamespace()
    env.actions = torch.zeros(4, 14)
    env.commands = torch.zeros(4, 13)
    env.episode_length_buf = torch.zeros(4, dtype=torch.long)
    env.official = SimpleNamespace(common_step_counter=0)
    guard = SoftPadGuard(terrain)
    guard.diagnostic_dir = tmp_path
    terrain.guard = guard
    return terrain, env, guard


@pytest.mark.parametrize("damage", ["compression", "collapse", "inversion", "nan_position", "nan_velocity"])
def test_guard_restores_both_buffers_from_independent_snapshot_only_for_failed_world(tmp_path, damage):
    terrain, env, guard = fixture(tmp_path)
    if damage == "compression":
        # A positive threshold remains available for explicit restricted-domain experiments.
        guard.settings["min_volume_ratio"] = 0.25
    guard.begin_step()
    saved = wp.to_torch(env.states[0].particle_q).clone()
    selected = terrain.pad_indices[2]
    for state in env.states:
        # SynReal overwrites BOTH buffers: neither can serve as the rollback source.
        q = wp.to_torch(state.particle_q)
        q[terrain.pad_indices[1], 0] += 0.001
        if damage in ("compression", "collapse", "inversion"):
            q[selected, 2] *= {"compression": 0.2, "collapse": 0.0, "inversion": -1.0}[damage]
        else:
            values = q if damage == "nan_position" else wp.to_torch(state.particle_qd)
            values[selected[0], 2] = float("nan")
    survivors = [wp.to_torch(s.particle_q)[terrain.pad_indices[[0, 1, 3]]].clone() for s in env.states]
    guard.after_substep(0)
    assert guard.failed.tolist() == [False, False, True, False]
    assert guard.invalid[2].item() == (damage != "compression")
    for state, before in zip(env.states, survivors, strict=True):
        torch.testing.assert_close(wp.to_torch(state.particle_q)[selected], saved[selected], atol=0, rtol=0)
        torch.testing.assert_close(wp.to_torch(state.particle_q)[terrain.pad_indices[[0, 1, 3]]], before, atol=0, rtol=0)
        assert torch.isfinite(wp.to_torch(state.particle_qd)).all()
    # Repeated substeps stay quarantined and do not count as new episode failures.
    wp.to_torch(env.states[0].particle_q)[selected, 0] += 0.01
    guard.after_substep(1)
    torch.testing.assert_close(wp.to_torch(env.states[0].particle_q)[selected], saved[selected], atol=0, rtol=0)
    guard.finish_step()
    assert guard.failure_count == 1
    terrain.check_state()
    assert len(list(tmp_path.glob("*.pt"))) == 1


@pytest.mark.parametrize("volume_ratio", [0.2, 0.01])
def test_guard_allows_positive_compression_without_reset_or_failure_count(tmp_path, volume_ratio):
    terrain, env, guard = fixture(tmp_path)
    guard.begin_step()
    selected = terrain.pad_indices[2]
    for state in env.states:
        wp.to_torch(state.particle_q)[selected, 2] *= volume_ratio
    expected = [wp.to_torch(state.particle_q).clone() for state in env.states]
    guard.after_substep(0)
    guard.finish_step()
    assert terrain.minimum_volume[2].item() == pytest.approx(volume_ratio)
    assert not guard.failed.any() and not guard.invalid.any()
    assert guard.failure_count == 0 and guard.window_failures == 0
    failure, _ = guard.terminations(torch.zeros(4, dtype=torch.bool), torch.zeros(4, dtype=torch.bool))
    assert not failure.any()
    for state, positions in zip(env.states, expected, strict=True):
        torch.testing.assert_close(wp.to_torch(state.particle_q), positions, atol=0, rtol=0)
    assert not list(tmp_path.glob("*.pt"))


def test_guard_keeps_valid_dynamics_and_clears_frame_mask(tmp_path):
    _, env, guard = fixture(tmp_path)
    guard.failed[0] = True
    guard.begin_step()
    for state in env.states:
        wp.to_torch(state.particle_q)[:, 0] += 0.002
    before = [wp.to_torch(s.particle_q).clone() for s in env.states]
    guard.after_substep(0)
    assert not guard.failed.any()
    for state, expected in zip(env.states, before, strict=True):
        torch.testing.assert_close(wp.to_torch(state.particle_q), expected, atol=0, rtol=0)


def test_guard_stops_persistent_failure_instead_of_silently_training(tmp_path):
    _, _, guard = fixture(tmp_path)
    for _ in range(99):
        guard.failed[:] = True
        guard.finish_step()
    with pytest.raises(FloatingPointError, match="frequent soft-pad failures"):
        guard.finish_step()


def test_rate_stop_reports_completed_window_before_counters_are_cleared(tmp_path):
    terrain, _, guard = fixture(tmp_path)
    terrain.num_pad_worlds = 32
    for step in range(99):
        guard.failed.zero_()
        guard.failed[0] = step < 8
        guard.finish_step()
    guard.failed.zero_()
    with pytest.raises(FloatingPointError, match="window_failures=8/3200"):
        guard.finish_step()
    report = json.loads((tmp_path / "guard_stop.json").read_text())
    assert report["reason"] == "window_failure_rate"
    assert report["window_steps"] == 100 and report["window_failures"] == 8
    assert report["failed_transition_fraction"] == pytest.approx(0.0025)
    assert report["invalid_count"] == 0 and report["failed_this_step"] == 0
    assert report["settings"]["min_volume_ratio"] == 0.0


def test_single_step_stop_reports_separate_reason(tmp_path):
    terrain, _, guard = fixture(tmp_path)
    terrain.num_pad_worlds = 32
    guard.failed = torch.ones(5, dtype=torch.bool)
    with pytest.raises(FloatingPointError, match="single_step_failures"):
        guard.finish_step()
    report = json.loads((tmp_path / "guard_stop.json").read_text())
    assert report["failed_this_step"] == 5 and report["window_steps"] == 1


def test_invalid_transition_has_only_failure_cost_and_no_timeout_bootstrap(tmp_path):
    _, env, guard = fixture(tmp_path)
    guard.failed[2] = True
    failure, timeout = guard.terminations(torch.zeros(4, dtype=torch.bool), torch.ones(4, dtype=torch.bool))
    assert failure.tolist() == [False, False, True, False]
    assert timeout.tolist() == [True, True, False, True]
    term = json.loads(SNAPSHOT.read_text())["tasks"]["velocity"]["rewards"]["soft_pad_failure"]
    recipe = OfficialRecipe.__new__(OfficialRecipe)
    recipe.env = env
    recipe.num_envs, recipe.device, recipe.step_dt = 4, "cpu", 0.02
    recipe.extras = {"log": {}}
    recipe.common_step_counter = 0
    recipe.refresh = lambda *args: None
    recipe.supported = {"tracking": "tracking", "soft_pad_failure": "soft_pad_failure"}
    recipe.spec = {"rewards": {"tracking": {"weight": 10}, "soft_pad_failure": term}, "curricula": {}}
    recipe.params = {"tracking": {}, "soft_pad_failure": {}}
    recipe.base_reward = lambda func, params: guard.failed.float() if func == "soft_pad_failure" else torch.ones(4)
    env.mixed_terrain = True
    reward, terms = recipe.reward(*([None] * 8))
    assert reward[2].item() == pytest.approx(term["weight"] * recipe.step_dt)
    assert reward[2] < 0 and terms["tracking"][2] == 0
    torch.testing.assert_close(reward[[0, 1, 3]], torch.full((3,), 0.2))
