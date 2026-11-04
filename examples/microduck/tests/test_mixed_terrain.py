"""Deformed support geometry, mixed sampling and full-surface contact semantics."""

from types import SimpleNamespace

import newton
import numpy as np
import pytest
import torch
import warp as wp

from examples.microduck.common import setup
from examples.microduck.scenes import MixedTerrain
from examples.microduck.sensors import _soft_contact_flags
from examples.microduck.scenes import SoftPadConfig, add_soft_pad


def terrain_fixture(count=4, config=None):
    config = SoftPadConfig(cells=(4, 4, 2)) if config is None else config
    template = newton.ModelBuilder()
    add_soft_pad(template, config)
    builder = newton.ModelBuilder()
    builder.replicate(template, count)
    env = SimpleNamespace(num_envs=count, device=torch.device("cpu"), cfg={}, _sync=lambda: None)
    terrain = MixedTerrain(env, builder, config)
    env.model = builder.finalize(device="cpu")
    terrain.finalize()
    env.states = [env.model.state(), env.model.state()]
    env.task_logic = SimpleNamespace(spawn=torch.zeros(count, 21))
    return terrain, env


def test_soft_contact_stiffness_reaches_solver_model_and_saved_metadata():
    config = SoftPadConfig(cells=(4, 4, 2), contact_stiffness=5000.0)
    terrain, env = terrain_fixture(config=config)
    assert env.model.soft_contact_ke == config.contact_stiffness
    assert env.cfg["terrain"]["soft_contact"]["ke"] == config.contact_stiffness
    assert env.cfg["terrain"]["pad"]["contact_stiffness"] == config.contact_stiffness
    assert terrain.guard.settings["min_volume_ratio"] == 0.0
    assert terrain.guard.settings["max_failed_transition_fraction"] == 0.001


def test_historical_pad_contact_stiffness_is_restored_explicitly():
    legacy = {"pad": {"density": 80.0}, "soft_contact": {"ke": 50000.0}}
    assert SoftPadConfig.from_checkpoint(legacy).contact_stiffness == 50000.0
    assert SoftPadConfig.from_checkpoint({"pad": {}}).contact_stiffness == 50000.0
    configured = {"pad": {"contact_stiffness": 5000.0}, "soft_contact": {"ke": 5000.0}}
    assert SoftPadConfig.from_checkpoint(configured).contact_stiffness == 5000.0


@pytest.mark.parametrize("stiffness", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_contact_stiffness_is_rejected(stiffness):
    with pytest.raises(ValueError, match="positive"):
        SoftPadConfig(contact_stiffness=stiffness)


def test_deformed_surface_heights_are_world_local_and_include_ground():
    terrain, env = terrain_fixture()
    particles = wp.to_torch(env.states[0].particle_q)
    world = wp.to_torch(env.model.particle_world)
    particles[world == 1, 2] -= 0.015
    positions = torch.tensor([[[0.0, 0.0, 0.2], [2.0, 0.0, 0.2]]] * 4)
    actual = terrain.heights(positions)
    assert actual[0, 0].item() == pytest.approx(0.2 - terrain.config.top_z)
    assert actual[1, 0].item() == pytest.approx(actual[0, 0].item() + 0.015)
    torch.testing.assert_close(actual[:, 1], torch.full((4,), 0.2))
    terrain.update_support(positions[:, 0])
    torch.testing.assert_close(terrain.origins[:, 2], 0.2 - actual[:, 0])


def test_partial_reset_restores_both_pad_buffers_without_touching_neighbors():
    terrain, env = terrain_fixture()
    for state in env.states:
        wp.to_torch(state.particle_q).add_(0.007)
        wp.to_torch(state.particle_qd).fill_(0.1)
        wp.to_torch(state.particle_f).fill_(0.2)
    before = [{n: wp.to_torch(getattr(s, n)).clone() for n in ("particle_q", "particle_qd", "particle_f")} for s in env.states]
    terrain.reset(torch.tensor([1, 3]))
    selected, other = terrain.pad_indices[[1, 3]].flatten(), terrain.pad_indices[[0, 2]].flatten()
    for state, previous in zip(env.states, before, strict=True):
        for name in previous:
            value = wp.to_torch(getattr(state, name))
            torch.testing.assert_close(value[other], previous[name][other], atol=0, rtol=0)
        torch.testing.assert_close(wp.to_torch(state.particle_q)[selected], terrain.rest[selected], atol=0, rtol=0)
        assert not wp.to_torch(state.particle_qd)[selected].any()
        assert not wp.to_torch(state.particle_f)[selected].any()


def test_reset_resamples_both_surfaces_and_pad_starts_are_inside_safe_region():
    terrain, env = terrain_fixture()
    torch.manual_seed(42)
    seen = torch.zeros(4, 2, dtype=torch.bool)
    for _ in range(32):
        ids = torch.arange(4)
        terrain.reset(ids)
        terrain.adjust_spawn(ids)
        seen[ids, terrain.spawn_on_pad.long()] = True
        extent = torch.tensor(terrain.config.size[:2]) / 2 - 0.2
        assert (env.task_logic.spawn[terrain.spawn_on_pad, :2].abs() <= extent).all()
        assert (terrain.origins[~terrain.spawn_on_pad, 0] < -terrain.config.size[0] / 2 - 0.5).all()
    assert seen.all()


def test_soft_volume_check_detects_inversion_before_learning():
    terrain, env = terrain_fixture()
    terrain.check_state()
    assert (terrain.minimum_volume > 0.999).all()
    particles = wp.to_torch(env.states[0].particle_q)
    selected = terrain.pad_indices[2]
    particles[selected, 2] *= -1
    with pytest.raises(FloatingPointError, match=r"worlds \[2\]"):
        terrain.check_state()


def test_standalone_world_support_queries_and_volume_checks():
    # The shared terrain must also accept the unreplicated viewer template, whose
    # particles carry world=-1 rather than the replicated training world's zero.
    config = SoftPadConfig(cells=(4, 4, 2))
    builder = newton.ModelBuilder()
    add_soft_pad(builder, config)
    env = SimpleNamespace(num_envs=1, device=torch.device("cpu"), cfg={}, _sync=lambda: None)
    terrain = MixedTerrain(env, builder, config)
    env.model = builder.finalize(device="cpu")
    terrain.finalize()
    env.states = [env.model.state(), env.model.state()]
    terrain.check_state()
    torch.testing.assert_close(terrain.surface_heights(torch.tensor([[[0.0, 0.0, 0.2]]])),
                               torch.tensor([[config.top_z]]))
    wp.to_torch(env.states[0].particle_q)[terrain.pad_indices[0], 2] *= -1
    with pytest.raises(FloatingPointError, match=r"worlds \[0\]"):
        terrain.check_state()


def test_soft_contacts_count_vertex_edge_face_and_reject_speculative_or_cross_world():
    def array(values, dtype):
        return wp.array(values, dtype=dtype, device="cpu")

    # Normals point out of the rigid underside toward the pad. Contact 3 has a
    # positive gap; contact 4 belongs to another world; contact 5 is an ABD proxy.
    feet = wp.zeros((2, 2), device="cpu")
    robot, head = wp.zeros((2, 1), device="cpu"), wp.zeros((2, 1), device="cpu")
    wp.launch(
        _soft_contact_flags,
        dim=8,
        inputs=[
            array([6], int),
            array([[0, -1, -1], [0, 1, -1], [0, 1, 2], [3, -1, -1], [4, -1, -1], [5, -1, -1]], wp.vec3i),
            array([[1, 0, 0], [0.5, 0.5, 0], [0.2, 0.3, 0.5], [1, 0, 0], [1, 0, 0], [1, 0, 0]], wp.vec3),
            array([0, 0, 1, 0, 0, 0], int),
            array([[0, 0, 0.06]] * 6, wp.vec3),
            array([[0, 0, -1]] * 6, wp.vec3),
            array([0, 1], int),
            array([0, 0], int),
            array([0, 0], float),
            array([1, 2], int),
            array([0, 0, 0, 0, 1, 0], int),
            array([0.0015] * 6, float),
            array([1, 1, 1, 1, 1, 0], int),
            array([[0, 0, 0.059], [0.1, 0, 0.059], [0, 0.1, 0.059], [0, 0, 0.04], [0, 0, 0.06], [0, 0, 0.06]], wp.vec3),
            array([wp.transform_identity()] * 2, wp.transform),
            wp.zeros(6, dtype=wp.vec3, device="cpu"),
            wp.zeros(2, dtype=wp.spatial_vector, device="cpu"),
            wp.zeros(2, dtype=wp.vec3, device="cpu"),
            feet,
            robot,
            head,
            wp.zeros((2, 2), device="cpu"),
            wp.zeros((2, 2), device="cpu"),
        ],
        device="cpu",
    )
    np.testing.assert_array_equal(feet.numpy(), [[2, 1], [0, 0]])
    np.testing.assert_array_equal(robot.numpy(), [[3], [0]])
    assert not head.numpy().any()


def test_mixed_cli_requires_supported_backend_and_transfer_is_explicit():
    with pytest.raises(SystemExit):
        setup("train", ["--terrain", "mixed_soft_pad", "--solver", "mujoco"])
    with pytest.raises(SystemExit):
        setup("train", ["--init-from", "x", "--resume", "y"])
    with pytest.raises(SystemExit):
        setup("train", ["--pad-probability", "nan"])
