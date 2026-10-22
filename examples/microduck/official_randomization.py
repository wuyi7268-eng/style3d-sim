"""Startup model randomization applied before either native solver is initialized."""

import json
from pathlib import Path

import torch
import warp as wp


def randomize_foot_friction(model, task, num_envs, device):
    spec = json.loads((Path(__file__).parent / "configs/official_snapshot.json").read_text())["tasks"][task]
    params = spec["events"]["foot_friction"]["params"]
    if params["operation"] != "abs" or not params["shared_random"]:
        raise NotImplementedError("Foot friction adapter expects shared absolute sampling")
    names = params["asset_cfg"]["geom_names"]
    indices = [i for i, label in enumerate(model.shape_label) if label.rsplit("/", 1)[-1] in names]
    if len(indices) != num_envs * len(names):
        raise ValueError("Could not resolve all foot collision shapes for friction randomization")
    indices = torch.tensor(indices, device=device)
    worlds = wp.to_torch(model.shape_world)[indices].long()
    values = torch.full((num_envs,), 1.0, device=device)
    wp.to_torch(model.shape_material_mu)[indices] = values[worlds]
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    # Identical foot materials permit SynReal's shared-world layout when its other constraints hold.
    return {"enabled": False, "range": [1.0, 1.0], "per_env": False, "shared_between_feet": True}
