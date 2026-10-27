"""Export fixed terrain realizations using the bundled official Python environment."""

import json
from pathlib import Path

import mujoco
from mjlab.tasks.registry import load_env_cfg
from mjlab.terrains.terrain_generator import TerrainGenerator

root = Path(__file__).resolve().parent / "configs"
for name, task in [("rough", "Mjlab-Velocity-Rough-MicroDuck"), ("slope", "Mjlab-RollerSlope-Flat-MicroDuck")]:
    cfg = load_env_cfg(task).scene.terrain.terrain_generator
    cfg.seed = 42
    spec = mujoco.MjSpec()
    generator = TerrainGenerator(cfg)
    generator.compile(spec)
    m = spec.compile()
    geoms = []
    for i in range(m.ngeom):
        g = {
            "type": int(m.geom_type[i]),
            "pos": m.geom_pos[i].tolist(),
            "quat": m.geom_quat[i].tolist(),
            "size": m.geom_size[i].tolist(),
        }
        if g["type"] == 1:
            j = m.geom_dataid[i]
            nr = int(m.hfield_nrow[j])
            nc = int(m.hfield_ncol[j])
            start = m.hfield_adr[j]
            g.update(nrow=nr, ncol=nc, hsize=m.hfield_size[j].tolist(), data=m.hfield_data[start : start + nr * nc].tolist())
        elif g["type"] != 6:
            raise ValueError(g["type"])
        geoms.append(g)
    (root / f"terrain_{name}.json").write_text(
        json.dumps({"seed": 42, "origins": generator.terrain_origins.tolist(), "geoms": geoms}, separators=(",", ":")) + "\n"
    )
    print(name, len(geoms))
