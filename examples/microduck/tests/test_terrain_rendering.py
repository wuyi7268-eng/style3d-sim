"""Replicated pads and rigid bodies must share display offsets without changing physics."""

from types import SimpleNamespace

import newton
import numpy as np
import pytest
import torch
import warp as wp
from newton.viewer import ViewerNull

from examples.microduck.env import MicroDuckEnv
from examples.microduck.scenes import MixedTerrain, SoftPadConfig, add_soft_pad


class CaptureViewer(ViewerNull):
    def log_points(self, name, points, radii=None, colors=None, hidden=False):
        if name == "/scene/sand":
            self.sand = (points.numpy().copy(), radii.numpy().copy(), hidden)
        elif name == "/scene/water":
            self.water = (points.numpy().copy(), hidden)

    def log_mesh(self, name, points, indices, *args, **kwargs):
        if name == "/model/triangles":
            self.surface = points.numpy().copy()

    def log_instances(self, name, mesh, xforms, *args, **kwargs):
        if xforms is not None and "sphere" in mesh:
            self.markers = xforms.numpy().copy()


@pytest.mark.parametrize("count", [1, 4, 30])
def test_pad_and_robot_display_alignment_preserves_physical_state(count):
    config = SoftPadConfig(cells=(2, 2, 1))
    template = newton.ModelBuilder()
    add_soft_pad(template, config)
    body = template.add_body(xform=wp.transform(wp.vec3(0, 0, config.top_z + 0.1), wp.quat_identity()))
    template.add_shape_sphere(body, radius=0.05)
    builder = newton.ModelBuilder()
    builder.replicate(template, count)
    env = SimpleNamespace(num_envs=count, device=torch.device("cpu"), cfg={}, _sync=lambda: None,
                          mixed_terrain=True, sim_time=1.25)
    env.terrain = MixedTerrain(env, builder, config)
    env.model = builder.finalize(device="cpu")
    env.terrain.finalize()
    env.states = [env.model.state(), env.model.state()]
    # Different deformation per world exposes accidentally repeated or stale pad geometry.
    worlds = env.model.particle_world.numpy()
    particles = env.states[0].particle_q.numpy()
    particles[:, 2] += worlds * 0.001
    env.states[0].particle_q.assign(particles)
    before = [{name: getattr(state, name).numpy().copy() for name in ("particle_q", "body_q")}
              for state in env.states]
    viewer = env.viewer = CaptureViewer()
    viewer.set_model(env.model)
    try:
        for spacing in ((3.2, 1.8, 0.0), (4.2, 2.8, 0.0)):
            viewer.set_world_offsets(spacing)
            MicroDuckEnv.render(env)
            offsets = viewer.world_offsets.numpy()
            np.testing.assert_allclose(viewer.surface, particles + offsets[worlds], atol=1.0e-6)
            np.testing.assert_allclose(viewer.markers[:, :3], before[0]["body_q"][:, :3] + offsets, atol=1.0e-6)
            for state, expected in zip(env.states, before, strict=True):
                for name, values in expected.items():
                    np.testing.assert_array_equal(getattr(state, name).numpy(), values)
            assert env.sim_time == 1.25
    finally:
        viewer.close()


@pytest.mark.parametrize("count", [1, 3])
@pytest.mark.parametrize("kind", ["sand", "pool"])
def test_scene_particles_visible_without_proxies_and_preserves_physics(count, kind):
    import style3d_sim as s3d
    from examples.microduck.policy_spec import build_robot
    from examples.microduck.scenes import SceneTerrain, add_scene

    template, _, _ = build_robot(s3d.ModelBuilder)
    metadata = add_scene(template, kind)
    builder = s3d.ModelBuilder()
    builder.replicate(template, count)
    env = SimpleNamespace(num_envs=count, device=torch.device("cpu"), cfg={}, mixed_terrain=False,
                          scene_terrain=True, sim_time=1.25)
    env.model = builder.finalize(device="cpu")
    env.terrain = SceneTerrain(env, kind, metadata)
    env.terrain.finalize()
    env.states = [env.model.state(), env.model.state()]
    material = "granular" if kind == "sand" else "fluid"
    indices = getattr(env.model, f"synreal_{material}_particle_indices").numpy()
    worlds = env.model.particle_world.numpy()[indices]
    assert len(indices) < env.model.particle_count  # Robot proxies must not enter the visible sand layer.
    viewer = env.viewer = CaptureViewer()
    viewer.set_model(env.model)
    viewer.show_particles = False
    try:
        for spacing in ((1.8, 1.6, 0.0), (2.8, 2.6, 0.0)):
            particles = env.states[0].particle_q.numpy()
            particles[indices, 2] += 0.01 * (worlds + 1)
            env.states[0].particle_q.assign(particles)
            before = [{name: getattr(state, name).numpy().copy() for name in ("particle_q", "body_q")}
                      for state in env.states]
            viewer.set_world_offsets(spacing)
            MicroDuckEnv.render(env)
            if kind == "sand":
                points, radii, hidden = viewer.sand
                np.testing.assert_allclose(radii, env.model.particle_radius.numpy()[indices])
            else:
                points, hidden = viewer.water
            assert not hidden
            assert not viewer.show_particles
            count_key = "grain_count_per_world" if kind == "sand" else "fluid_count_per_world"
            assert len(points) == metadata[count_key] * count
            np.testing.assert_allclose(points, particles[indices] + viewer.world_offsets.numpy()[worlds], atol=1e-6)
            for state, expected in zip(env.states, before, strict=True):
                for name, values in expected.items():
                    np.testing.assert_array_equal(getattr(state, name).numpy(), values)
    finally:
        viewer.close()
