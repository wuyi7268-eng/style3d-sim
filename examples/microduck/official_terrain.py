"""Shared static terrain geometry exported from the bundled official generator."""

import json
from pathlib import Path

import newton
import numpy as np
import torch
import warp as wp

from .policy_spec import filter_authored_collider


@wp.kernel
def _height_ray(mesh: wp.uint64, positions: wp.array[wp.vec3], heights: wp.array[float]):
    i = wp.tid()
    p = positions[i]
    hit = wp.mesh_query_ray(mesh, p, wp.vec3(0.0, 0.0, -1.0), 1.0)
    heights[i] = -1.0
    if hit.result:
        heights[i] = hit.t
        if hit.normal[2] < 0.0:
            heights[i] = 0.0


class OfficialTerrain:
    def __init__(self, env, builder):
        self.env = env
        self.kind = "rough" if "_rough" in env.task else "slope" if env.family == "rollerslope" else "plane"
        self.origins = torch.zeros(env.num_envs, 3, device=env.device)
        self.mesh = None
        self.levels = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.columns = torch.zeros_like(self.levels)
        if self.kind == "plane":
            filter_authored_collider(builder, builder.add_ground_plane())
            self.shape_count = 1
            return
        source = json.loads((Path(__file__).parent / "configs" / f"terrain_{self.kind}.json").read_text())
        self.grid = torch.tensor(source["origins"], device=env.device)
        if self.kind == "rough":
            self.levels.random_(self.grid.shape[0])
            self.columns.random_(self.grid.shape[1])
        self.origins.copy_(self.grid[self.levels, self.columns])
        vertices, faces = [], []
        corners = np.array([[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]])
        box_faces = np.array(
            [
                [0, 4, 6],
                [0, 6, 2],
                [1, 3, 7],
                [1, 7, 5],
                [0, 1, 5],
                [0, 5, 4],
                [2, 6, 7],
                [2, 7, 3],
                [0, 2, 3],
                [0, 3, 1],
                [4, 5, 7],
                [4, 7, 6],
            ]
        )
        for i, geom in enumerate(source["geoms"]):
            w, x, y, z = geom["quat"]
            xf = wp.transform(geom["pos"], (x, y, z, w))
            cfg = builder.default_shape_cfg.copy()
            cfg.mu = 1.0
            if geom["type"] == 6:
                shape = builder.add_shape_box(
                    -1, xform=xf, hx=geom["size"][0], hy=geom["size"][1], hz=geom["size"][2], cfg=cfg, label=f"terrain_{i}"
                )
                points = corners * np.asarray(geom["size"])
                triangles = box_faces
            else:
                nr, nc = geom["nrow"], geom["ncol"]
                hx, hy, hz, _ = geom["hsize"]
                data = np.asarray(geom["data"]).reshape(nr, nc)
                shape = builder.add_shape_heightfield(
                    xform=xf, heightfield=newton.Heightfield(data, nr, nc, hx, hy, 0.0, hz), cfg=cfg, label=f"terrain_{i}"
                )
                xx, yy = np.meshgrid(np.linspace(-hx, hx, nc), np.linspace(-hy, hy, nr))
                points = np.stack([xx, yy, data * hz], axis=-1).reshape(-1, 3)
                triangles = []
                for row in range(nr - 1):
                    for col in range(nc - 1):
                        a = row * nc + col
                        triangles.extend([[a, a + 1, a + nc + 1], [a, a + nc + 1, a + nc]])
                triangles = np.asarray(triangles)
            filter_authored_collider(builder, shape)
            offset = len(vertices)
            vertices.extend([list(wp.transform_point(xf, wp.vec3(*p))) for p in points])
            faces.extend((triangles + offset).reshape(-1).tolist())
        self.shape_count = len(source["geoms"])
        self.mesh = wp.Mesh(
            points=wp.array(vertices, dtype=wp.vec3, device=str(env.device)),
            indices=wp.array(faces, dtype=wp.int32, device=str(env.device)),
        )

    def heights(self, positions, quaternions=None):
        if self.mesh is None:
            return positions[..., 2].clamp(0.0, 1.0)
        rays = positions.unsqueeze(-2).repeat(1, 1, 3, 1)
        if quaternions is not None:
            w, x, y, z = quaternions.unbind(-1)
            yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
            offset = torch.stack((yaw.cos(), yaw.sin(), torch.zeros_like(yaw)), -1) * 0.04
            rays[:, :, 1] += offset
            rays[:, :, 2] -= offset
        result = torch.empty(rays.shape[:-1], device=self.env.device)
        wp.launch(
            _height_ray,
            dim=result.numel(),
            inputs=[
                self.mesh.id,
                wp.from_torch(rays.reshape(-1, 3).contiguous(), dtype=wp.vec3),
                wp.from_torch(result.flatten()),
            ],
            device=str(self.env.device),
        )
        self.env._sync()
        missing = result < 0
        all_missing = missing.all(-1)
        result = torch.where(missing, torch.ones_like(result), result).min(-1).values
        return torch.where(all_missing, positions[..., 2].clamp(0, 1), result)

    def reset(self, ids):
        if self.kind == "plane" or self.env.official is None:
            return
        active = ids[self.env.episode_length_buf[ids] > 0]
        if not len(active):
            return
        position = self.env._views()[2][active, :3]
        if self.kind == "slope":
            distance = position[:, 0] - self.origins[active, 0]
            up, down = distance > 6.0, distance < 3.0
        else:
            distance = (position[:, :2] - self.origins[active, :2]).norm(dim=-1)
            up = distance > 4.0
            target = self.env.commands[active, :2].norm(dim=-1) * self.env.max_episode_length * self.env.step_dt * 0.5
            down = (distance < target) & ~up
        levels = self.levels[active] + up.long() - down.long()
        levels = torch.where(
            levels >= len(self.grid), torch.randint(len(self.grid), levels.shape, device=levels.device), levels
        ).clamp(min=0)
        self.levels[active] = levels
        self.origins[active] = self.grid[levels, self.columns[active]]

    def out_of_bounds(self, positions):
        if self.kind != "rough":
            return torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        return (positions[:, :2].abs() > 19.7).any(-1)
