"""Capture the actual checkpoint playback states without advancing or editing the solver."""

from __future__ import annotations

import re
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path

import newton
import numpy as np
from pxr import Gf, Usd, UsdGeom, Vt

from demos.utils.render_provenance import atomic_json, sha256


def pose_ops(prim, pose, frame=None):
    xform = UsdGeom.Xformable(prim)
    ops = xform.GetOrderedXformOps()
    if not ops:
        ops = [xform.AddTranslateOp(), xform.AddOrientOp()]
    time = Usd.TimeCode.Default() if frame is None else Usd.TimeCode(frame)
    ops[0].Set(Gf.Vec3d(*map(float, pose[:3])), time)
    ops[1].Set(Gf.Quatf(float(pose[6]), Gf.Vec3f(*map(float, pose[3:6]))), time)


class PlaybackRecording:
    def __init__(self, env, args):
        self.env, self.args = env, args
        self.folder = args.record_dir.expanduser().resolve()
        self.folder.mkdir(parents=True, exist_ok=False)
        self.path = self.folder / "simulation.usdc"
        self.stage = Usd.Stage.CreateNew(str(self.path))
        self.fps = 1.0 / env.step_dt
        self.stage.SetTimeCodesPerSecond(self.fps)
        self.stage.SetFramesPerSecond(self.fps)
        self.stage.SetStartTimeCode(0)
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        self.stage.SetDefaultPrim(UsdGeom.Xform.Define(self.stage, "/World").GetPrim())
        self.bodies, self.timeline = [], []
        model = env.model
        parents, flags = model.shape_body.numpy(), model.shape_flags.numpy()
        types, scales = model.shape_type.numpy(), model.shape_scale.numpy()
        transforms, colors = model.shape_transform.numpy(), model.shape_color.numpy()
        paths = {}
        for index in sorted(set(int(p) for p, f in zip(parents, flags) if p >= 0 and f & int(newton.ShapeFlags.VISIBLE))):
            name = re.sub(r"[^a-zA-Z0-9_]", "_", model.body_label[index].split("/")[-1])
            path = f"/World/Robot/body_{index:03d}_{name}"
            prim = UsdGeom.Xform.Define(self.stage, path)
            paths[index] = path
            self.bodies.append((index, prim))
        for index, source in enumerate(model.shape_source):
            if not flags[index] & int(newton.ShapeFlags.VISIBLE):
                continue
            parent = int(parents[index])
            path = f"{paths.get(parent, '/World/Ground')}/shape_{index:03d}"
            kind = "robot" if parent >= 0 else "floor"
            if parent < 0 and "trampoline_" in model.shape_label[index] and "floor" not in model.shape_label[index]:
                kind = "frame"
            if types[index] in (newton.GeoType.MESH, newton.GeoType.CONVEX_MESH):
                points, faces = np.asarray(source.vertices) * scales[index], source.indices
            elif types[index] == newton.GeoType.PLANE:
                # The physical ground is infinite; this finite patch is only its visible representation.
                points = np.array([[-20, -20, 0], [20, -20, 0], [20, 20, 0], [-20, 20, 0]])
                faces = [0, 1, 2, 0, 2, 3]
            elif types[index] == newton.GeoType.BOX:
                # Shape scales are half extents; preserve the cloth's actual corner mounts and posts.
                points = np.array([
                    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
                ]) * scales[index]
                faces = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4,
                         1, 2, 6, 1, 6, 5, 2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7]
            elif types[index] == newton.GeoType.CAPSULE:
                from newton._src.utils.mesh import create_mesh_capsule

                points, faces, _, _ = create_mesh_capsule(
                    float(scales[index, 0]), float(scales[index, 1]), up_axis=2, segments=24,
                )
            else:
                raise ValueError(f"Unsupported visible shape {index}: {types[index]}")
            mesh = self.mesh(path, points, faces, colors[index], kind)
            pose_ops(mesh, transforms[index])
        # Soft pads are tetrahedral meshes, while cloth scenes expose only surface
        # triangles. Select the terrain topology from the active scene instead of
        # assuming every deformable support has tetrahedra.
        terrain_kind = env.terrain.kind if hasattr(env.terrain, "kind") else env.terrain.cfg.get("kind")
        triangles = model.tri_indices.numpy()
        if terrain_kind == "trampoline":
            triangles = triangles[np.isin(triangles, env.terrain.cloth_indices.cpu().numpy()).all(axis=1)]
        elif terrain_kind == "cloth":
            triangles = np.asarray(env.terrain.surface_triangles[0].detach().cpu().numpy(), dtype=np.int32)
        else:
            pad_particles = np.unique(model.tet_indices.numpy())
            triangles = triangles[np.isin(triangles, pad_particles).all(axis=1)]
        if not len(triangles):
            raise ValueError("The recorded environment has no deformable terrain surface")
        self.particles, faces = np.unique(triangles, return_inverse=True)
        self.pad = self.mesh("/World/Pad/soft_pad", model.particle_q.numpy()[self.particles], faces,
                             (0.16, 0.42, 0.49), "soft_pad")
        self.source_hashes = {
            str(p): sha256(p) for p in [
                Path("examples/microduck/play.py"), Path("examples/microduck/common.py"),
                Path("examples/microduck/env.py"), Path("examples/microduck/scenes.py"), Path(__file__),
                Path("examples/microduck/heading_control.py"),
                Path("examples/microduck/jump.py"), Path("examples/microduck/rebound.py"),
            ]
        }
        self.checkpoint_hash = sha256(args.checkpoint)
        print(f"Recording actual playback to {self.folder} at {self.fps:g} Hz", flush=True)

    def mesh(self, path, points, faces, color, kind):
        mesh = UsdGeom.Mesh.Define(self.stage, path)
        faces = np.asarray(faces, dtype=np.int32).ravel()
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(points, dtype=np.float32)))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces) // 3, 3, dtype=np.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*map(float, color))])
        mesh.GetPrim().SetCustomDataByKey("kind", kind)
        return mesh

    def sample(self):
        self.env._sync()
        state = self.env.states[0]
        bodies = state.body_q.numpy()
        particles = state.particle_q.numpy()[self.particles]
        if not np.isfinite(bodies).all() or not np.isfinite(particles).all():
            raise FloatingPointError("Cannot record nonfinite simulation geometry")
        frame = len(self.timeline)
        for index, prim in self.bodies:
            pose_ops(prim, bodies[index], frame)
        self.pad.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(particles), frame)
        root = self.env._views()[2][0].cpu().numpy()
        self.timeline.append({
            "frame": frame, "simulation_time": float(self.env.sim_time), "root_pose": root.tolist(),
            "command": self.env.commands[0, :3].cpu().tolist(), "resets": self.env.total_resets,
            "pad_min_z": float(particles[:, 2].min()), "pad_max_z": float(particles[:, 2].max()),
        })
        if (frame + 1) % 100 == 0:
            print(f"Recorded {frame + 1} frames ({(frame + 1) / self.fps:.1f} seconds)", flush=True)

    def reached_pad_edge(self):
        position = self.timeline[-1]["root_pose"]
        center = self.env.terrain.origins[0, :2].cpu().tolist()
        if getattr(self.env.terrain, "kind", None) == "cloth":
            config = self.env.terrain.cfg["config"]
            size = (config["cloth_length"], config["cloth_width"])
        else:
            size = self.env.terrain.config.size
        return any(abs(position[axis] - center[axis]) >= size[axis] / 2 for axis in range(2))

    def close(self, *, completed):
        if self.stage is None:
            return
        self.stage.SetEndTimeCode(max(0, len(self.timeline) - 1))
        self.stage.GetRootLayer().Save()
        self.stage = None
        atomic_json(self.folder / "trajectory.json", self.timeline)
        atomic_json(self.folder / "environment.json", self.env.cfg)
        manifest = {
            "schema_version": 1, "run_id": str(uuid.uuid4()), "completed": completed,
            "checkpoint": {"path": str(self.args.checkpoint.resolve()), "sha256": self.checkpoint_hash},
            "checkpoint_config_sha256": sha256(self.args.checkpoint.with_name("config.json")),
            "arguments": {k: str(v) if isinstance(v, Path) else asdict(v) if is_dataclass(v) else v
                          for k, v in vars(self.args).items()},
            "source_sha256": self.source_hashes,
            "timeline": {"first_frame": 0, "last_frame": len(self.timeline) - 1, "fps": self.fps,
                         "frame_count": len(self.timeline), "sampling": "post-policy-step, no paused frames"},
            "artifacts": {key: {"path": name, "sha256": sha256(self.folder / name)} for key, name in (
                ("usd", "simulation.usdc"), ("trajectory", "trajectory.json"), ("environment", "environment.json"),
            )},
        }
        atomic_json(self.folder / "run_manifest.json", manifest)
        print(f"Saved {len(self.timeline)} recorded frames to {self.path}", flush=True)
