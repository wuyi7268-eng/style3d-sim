"""Startup model randomization applied before either native solver is initialized."""

import json
from pathlib import Path

import torch
import warp as wp


def randomize_foot_friction(model, task, num_envs, device, *, enabled=False):
    spec = json.loads((Path(__file__).parent / "configs/official_snapshot.json").read_text())["tasks"][task]
    if "foot_friction" not in spec["events"]:
        return {"enabled": False, "reason": "official roller task has no foot_friction event"}
    params = spec["events"]["foot_friction"]["params"]
    if params["operation"] != "abs" or not params["shared_random"]:
        raise NotImplementedError("Foot friction adapter expects shared absolute sampling")
    names = params["asset_cfg"]["geom_names"]
    indices = [i for i, label in enumerate(model.shape_label) if label.rsplit("/", 1)[-1] in names]
    if len(indices) != num_envs * len(names):
        raise ValueError("Could not resolve all foot collision shapes for friction randomization")
    indices = torch.tensor(indices, device=device)
    worlds = wp.to_torch(model.shape_world)[indices].long()
    if enabled:
        values = torch.empty(num_envs, device=device).uniform_(*params["ranges"])
    else:
        values = torch.ones(num_envs, device=device)
    wp.to_torch(model.shape_material_mu)[indices] = values[worlds]
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "enabled": enabled,
        "range": params["ranges"] if enabled else [1.0, 1.0],
        "per_env": enabled,
        "shared_between_feet": True,
    }
