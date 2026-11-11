"""Deformed support queries and reset isolation for the trampoline scene."""

from dataclasses import replace
from types import SimpleNamespace

import newton
import numpy as np
import pytest
import torch
import warp as wp

from examples.microduck.scenes import SCENE_CONFIG, SceneTerrain, add_scene


def test_deformed_support_and_selected_reset_exclude_robot_proxies():
    import style3d_sim as s3d
    from examples.microduck.policy_spec import build_robot

    template, _, _ = build_robot(s3d.ModelBuilder)
    # CPU tests exercise topology without constructing CUDA mesh SDF textures.
    from unittest.mock import patch

    with patch("examples.microduck.scenes.provision_pad_contacts"):
        metadata = add_scene(template, "trampoline")
    builder = s3d.ModelBuilder()
    newton.solvers.SolverStyle3D.register_custom_attributes(builder)
    builder.replicate(template, 2)
    model = builder.finalize(device="cpu")
    env = SimpleNamespace(model=model, num_envs=2, device=torch.device("cpu"), cfg={}, _sync=lambda: None)
    terrain = SceneTerrain(env, "trampoline", metadata)
    terrain.finalize()
    env.states = [model.state(), model.state()]
    indices = terrain.cloth_indices.numpy()
    assert len(indices) == 2 * SCENE_CONFIG.trampoline_resolution**2
    assert not np.intersect1d(indices, model.synreal_affine_body_indices.numpy()).size
    selected = indices[terrain.cloth_worlds.numpy() == 0]
    other = np.setdiff1d(np.arange(model.particle_count), selected)
    for state in env.states:
        q = wp.to_torch(state.particle_q)
        q[terrain.cloth_indices, 2] -= 0.05
        wp.to_torch(state.particle_qd).fill_(0.2)
    # A former square corner must have neither cloth support nor a valid training position.
    points = torch.tensor([[[0.013, 0.017, 0.6], [0.35, 0.35, 0.6]]] * 2)
    heights = terrain.surface_heights(points)
    torch.testing.assert_close(heights, torch.tensor([[0.35, 0.0]] * 2))
    before = [state.particle_q.numpy().copy() for state in env.states]
    terrain.reset(torch.tensor([0]))
    for state, original in zip(env.states, before, strict=True):
        np.testing.assert_array_equal(state.particle_q.numpy()[selected], model.particle_q.numpy()[selected])
        np.testing.assert_array_equal(state.particle_qd.numpy()[selected], 0)
        np.testing.assert_array_equal(state.particle_q.numpy()[other], original[other])
        np.testing.assert_allclose(state.particle_qd.numpy()[other], 0.2)
    torch.testing.assert_close(terrain.surface_heights(points), torch.tensor([[0.4, 0.0], [0.35, 0.0]]))
    edge = SCENE_CONFIG.trampoline_length / 2
    assert terrain.out_of_bounds(torch.tensor([[0.0, 0.0, 0.5], [edge, 0.0, 0.5]])).tolist() == [False, True]
    assert terrain.out_of_bounds(torch.tensor([[0.2, 0.2, 0.5], [0.3, 0.3, 0.5]])).tolist() == [False, True]


@pytest.mark.parametrize("field,value", [("resolution", 2), ("length", 0), ("density", float("nan")),
                                          ("damping", -1)])
def test_invalid_trampoline_parameters(field, value):
    config = replace(SCENE_CONFIG, **{f"trampoline_{field}": value})
    with pytest.raises(ValueError):
        add_scene(newton.ModelBuilder(), "trampoline", config)
