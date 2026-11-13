"""Resolve independent task recipes without inheriting live flat-ground settings."""

import json
from pathlib import Path

ROOT = Path(__file__).parent
SNAPSHOT = ROOT / "configs/official_snapshot.json"
BRIDGE_RECIPE = ROOT / "configs/elastic_bridge_velocity.json"
BRIDGE_TRAINING = ROOT / "configs/elastic_bridge_training.json"
RUNNING_RECIPE = ROOT / "configs/running.json"


def is_bridge_velocity(task, terrain):
    return task == "velocity" and terrain == "elastic_bridge"


def load_task_spec(task="velocity", terrain="default"):
    """Return fresh mutable data for one task; bridge never reads the flat recipe."""
    if task == "running":
        if terrain not in ("default", "plane"):
            raise ValueError("Running supports flat ground only")
        document = json.loads(RUNNING_RECIPE.read_text())
        return document["spec"], document["source_sha256"], document["profile"]
    if is_bridge_velocity(task, terrain):
        document = json.loads(BRIDGE_RECIPE.read_text())
        spec = document["spec"]
        spec["rl"] = json.loads(BRIDGE_TRAINING.read_text())["rl"]
        return spec, document["source_sha256"], document["profile"]
    snapshot = json.loads(SNAPSHOT.read_text())
    spec = snapshot["tasks"][task]
    for section, terms in spec.get("terrain_overrides", {}).get(terrain, {}).items():
        spec[section].update(terms)
    return spec, snapshot["source_sha256"], f"{task}:{terrain}"
