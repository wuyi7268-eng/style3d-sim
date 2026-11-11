"""Build an editable Blender studio scene from one sealed MicroDuck playback."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector
from pxr import Usd, UsdGeom

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from render_provenance import load_run  # noqa: E402
from usd_cache import record_usd_sources  # noqa: E402


def aim(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def material(name, color, roughness):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    shader = mat.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Roughness"].default_value = roughness
    mat.diffuse_color = (*color, 1)
    return mat


def mechanical_normals(obj):
    """Preserve mechanical creases and keep thin triangles from bending broad panel shading."""
    obj.data.set_sharp_from_angle(angle=math.radians(35))
    modifier = obj.modifiers.new("Panel normals | preserve hard edges", "WEIGHTED_NORMAL")
    modifier.keep_sharp = True
    modifier.mode = "FACE_AREA_WITH_ANGLE"
    modifier.weight = 50


def pad_grid(obj, mat, relief=False):
    """Attach subtle metric markings to rest-space UVs so they follow the cached deformation."""
    uv = obj.data.uv_layers.new(name="Pad rest coordinates")
    for loop in obj.data.loops:
        position = obj.data.vertices[loop.vertex_index].co
        uv.data[loop.index].uv = (position.x, position.y)
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    coords = nodes.new("ShaderNodeTexCoord")
    separate = nodes.new("ShaderNodeSeparateXYZ")
    links.new(coords.outputs["UV"], separate.inputs[0])
    lines = []
    for axis in ("X", "Y"):
        scale = nodes.new("ShaderNodeMath")
        scale.operation = "MULTIPLY"
        scale.inputs[1].default_value = 20.0
        links.new(separate.outputs[axis], scale.inputs[0])
        fraction = nodes.new("ShaderNodeMath")
        fraction.operation = "FRACT"
        links.new(scale.outputs[0], fraction.inputs[0])
        line = nodes.new("ShaderNodeMath")
        line.operation = "LESS_THAN"
        line.inputs[1].default_value = 0.035 if relief else 0.018
        links.new(fraction.outputs[0], line.inputs[0])
        lines.append(line)
    union = nodes.new("ShaderNodeMath")
    union.operation = "MAXIMUM"
    for index, line in enumerate(lines):
        links.new(line.outputs[0], union.inputs[index])
    mix = nodes.new("ShaderNodeMixRGB")
    mix.inputs[1].default_value = (0.055, 0.24, 0.28, 1)
    mix.inputs[2].default_value = (0.032, 0.14, 0.17, 1)
    links.new(union.outputs[0], mix.inputs[0])
    links.new(mix.outputs[0], nodes.get("Principled BSDF").inputs["Base Color"])
    if relief:
        # Short-range occlusion emphasizes actual foot contact without displacing the cached surface.
        occlusion = nodes.new("ShaderNodeAmbientOcclusion")
        occlusion.inputs["Distance"].default_value = 0.045
        occlusion.samples = 16
        links.new(mix.outputs[0], occlusion.inputs["Color"])
        links.new(occlusion.outputs["Color"], nodes.get("Principled BSDF").inputs["Base Color"])
        nodes.get("Principled BSDF").inputs["Roughness"].default_value = 0.48


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--relief-lighting", action="store_true",
                        help="Lower camera, grazing key light and contact shading; preserve recorded geometry")
    parser.add_argument("--mechanical-normals", action="store_true",
                        help="Preserve robot hard edges and apply area/angle weighted normals without changing geometry")
    parser.add_argument("--front-view", action="store_true", help="Place the camera ahead of the initial robot heading")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    folder = args.record_dir.resolve()
    run = load_run(folder / "run_manifest.json")
    trajectory = json.loads((folder / "trajectory.json").read_text())
    environment = json.loads((folder / "environment.json").read_text())
    cloth = environment.get("terrain", {}).get("kind") == "cloth"
    trampoline = environment.get("terrain", {}).get("kind") == "trampoline"
    if not trajectory:
        raise ValueError("Cannot build a scene from an empty recording")
    usd = folder / run["artifacts"]["usd"]["path"]
    stage = Usd.Stage.Open(str(usd))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.usd_import(filepath=str(usd), import_materials=False, set_frame_range=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.05
    scene.cycles.max_bounces = 4
    scene.render.resolution_x, scene.render.resolution_y = 1280, 720
    scene.render.resolution_percentage = 100
    scene.render.fps = round(run["timeline"]["fps"])
    scene.render.fps_base = scene.render.fps / run["timeline"]["fps"]
    scene.frame_start, scene.frame_end = run["timeline"]["first_frame"], run["timeline"]["last_frame"]
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = "AgX"
    scene.world = bpy.data.worlds.new("Studio world")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.30, 0.36, 0.45, 1)
    scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.12 if args.relief_lighting else 0.35
    materials = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        obj = bpy.data.objects[prim.GetName()]
        kind = prim.GetCustomDataByKey("kind")
        color = tuple(UsdGeom.Mesh(prim).GetDisplayColorAttr().Get()[0])
        if kind == "floor":
            color = (0.16, 0.19, 0.23)
        key = (kind, color)
        if key not in materials:
            materials[key] = material(f"{kind}_{len(materials):02d}", color, 0.76 if kind != "robot" else 0.32)
        obj.data.materials.clear()
        obj.data.materials.append(materials[key])
        if kind == "soft_pad":
            pad_grid(obj, materials[key], args.relief_lighting)
        # Keep the FEM mesh unsmoothed so contact indentation is visible without invented displacement.
        if kind == "robot":
            for polygon in obj.data.polygons:
                polygon.use_smooth = True
            if args.mechanical_normals:
                mechanical_normals(obj)
        if kind == "frame" and len(obj.data.polygons) > 100:
            for polygon in obj.data.polygons:
                polygon.use_smooth = True
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = "Camera | follow duck"
    camera.data.lens = 48
    camera.data.clip_start, camera.data.clip_end = 0.01, 100
    scene.camera = camera
    x, y, z, w = trajectory[0]["root_pose"][3:7]
    initial_heading = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    # Translation-only follow keeps world heading and the pad edge readable throughout the replay.
    for entry in trajectory:
        root = Vector(entry["root_pose"][:3])
        target = Vector((root.x, root.y, root.z + (0.005 if args.relief_lighting else 0.025)))
        offset = (0.48, -0.68, 0.25) if args.relief_lighting else (0.50, -0.68, 0.36)
        if cloth:
            # Include the suspended surface and corner supports so its boundary conditions remain visible.
            target.z -= 0.12
            offset = (0.95, -1.35, 0.68)
        if trampoline:
            target.z = root.z + 0.045
            offset = (0.65, -0.90, 0.44)
        if args.front_view:
            offset = (1.45, 0.35, 0.95) if cloth else (0.55, 0.65, offset[2])
            c, s = math.cos(initial_heading), math.sin(initial_heading)
            offset = (c * offset[0] - s * offset[1], s * offset[0] + c * offset[1], offset[2])
        camera.location = target + Vector(offset)
        aim(camera, target)
        camera.keyframe_insert(data_path="location", frame=entry["frame"])
        camera.keyframe_insert(data_path="rotation_euler", frame=entry["frame"])
    lighting = (
        ("Key", (1.5, -2.0, 1.5), 150, 1.2, (1.0, 0.88, 0.74)),
        ("Fill", (-2.0, -0.5, 2.0), 65, 2.5, (0.66, 0.80, 1.0)),
        ("Rim", (0.0, 2.0, 2.5), 220, 2.0, (1.0, 1.0, 1.0)),
    )
    if args.relief_lighting:
        # Grazing light turns real surface slopes into readable highlights and shadows.
        lighting = (
            ("Key", (-0.8, -1.3, 0.55), 90, 0.40, (1.0, 0.91, 0.80)),
            ("Fill", (1.5, -0.2, 1.8), 25, 2.0, (0.72, 0.84, 1.0)),
            ("Rim", (0.0, 1.5, 1.4), 95, 1.2, (1.0, 1.0, 1.0)),
        )
    for name, location, energy, size, color in lighting:
        data = bpy.data.lights.new(name, "AREA")
        data.energy, data.shape, data.size, data.color = energy, "DISK", size, color
        light = bpy.data.objects.new(name, data)
        scene.collection.objects.link(light)
        support_height = 0.93 if cloth else 0.4 if trampoline else 0.0
        light.location = Vector(location) + Vector((0, 0, support_height))
        aim(light, (0, 0, 0.15 + support_height))
        for entry in trajectory:
            root = entry["root_pose"]
            light.location = Vector(location) + Vector((root[0], root[1], support_height))
            light.keyframe_insert(data_path="location", frame=entry["frame"])
    scene["run_id"] = run["run_id"]
    scene["description"] = "Recorded SynReal checkpoint playback; camera and lighting only, no resimulation."
    scene["relief_lighting"] = args.relief_lighting
    scene["mechanical_normals"] = args.mechanical_normals
    blend = folder / "scene.blend"
    # Save once to establish the base directory used by Blender's relative cache paths.
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    for cache in bpy.data.cache_files:
        cache.filepath = bpy.path.relpath(cache.filepath)
    record_usd_sources(scene)
    scene.frame_set(scene.frame_start)
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    print(f"Saved {blend}", flush=True)


if __name__ == "__main__":
    main()
