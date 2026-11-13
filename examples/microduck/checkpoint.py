"""Explicit transfer initialization, distinct from strict same-recipe resume."""

import hashlib
import json
import math
from pathlib import Path

from .observations import CONTRACT, HISTORY_SPEC
from .policy_spec import EFFORT, KD, KP
from .recipe import RECIPE


def checkpoint_policy_variant(metadata):
    """Legacy metadata denotes the standard network; new history checkpoints are explicit."""
    variant = metadata.get("policy_variant", "standard")
    from .cloth_policy import POLICY_CLASS, SUPPORTED_SPECS, VARIANT, supports_cloth_policy

    classes = {"standard": "ActorCritic", "history": "HistoryActorCritic", VARIANT: POLICY_CLASS}
    if variant == VARIANT and (
        metadata.get("history") not in SUPPORTED_SPECS
        or not supports_cloth_policy(metadata.get("task"), metadata.get("terrain", {}).get("kind"))
    ):
        raise ValueError("Unsupported cloth history scope or observation/feedback contract")
    if variant not in classes:
        raise ValueError(f"Unsupported policy_variant: {variant}")
    policy_class = metadata.get("ppo", {}).get("policy", {}).get("class_name", "ActorCritic")
    if policy_class != classes[variant]:
        raise ValueError(f"policy_variant={variant} does not match policy class {policy_class}")
    if variant == "history" and metadata.get("history") != HISTORY_SPEC:
        raise ValueError("Unsupported history observation/encoder contract")
    return variant


def resolve_checkpoint(path):
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        candidates = [p for p in path.glob("model_*.pt") if p.stem.removeprefix("model_").isdigit()]
        if not candidates:
            raise ValueError(f"No numbered checkpoints in {path}")
        path = max(candidates, key=lambda p: int(p.stem.removeprefix("model_")))
    if not path.is_file():
        raise ValueError(f"Checkpoint does not exist: {path}")
    return path


def transfer_metadata(path, *, task="velocity"):
    if task not in ("velocity", "velstand"):
        raise ValueError("Transfer supports same-task Velocity or VelStand checkpoints")
    metadata = json.loads(path.with_name("config.json").read_text(encoding="utf-8"))
    for name, expected in {
        "solver": "synreal",
        "task": task,
        "recipe": RECIPE,
        "observation_contract": CONTRACT,
        "actuator": "position_pd",
        "self_collisions": False,
    }.items():
        if metadata.get(name) != expected:
            raise ValueError(f"Transfer requires {name}={expected!r}; checkpoint has {metadata.get(name)!r}")
    for name, expected in (("kp", KP), ("kd", KD), ("effort_limit", EFFORT), ("policy_dt", 0.02)):
        value = metadata.get(name)
        if not isinstance(value, (int, float)) or not math.isclose(value, expected, rel_tol=1.0e-6):
            raise ValueError(f"Transfer checkpoint has incompatible {name}: {value}")
    from .env import PhysicsConfig

    physics = PhysicsConfig.from_checkpoint(metadata)
    if (not math.isclose(physics.physics_dt * physics.decimation, 0.02, rel_tol=1.0e-6)
            or not math.isclose(metadata.get("physics_dt", 0.0), physics.physics_dt, rel_tol=1.0e-6)):
        raise ValueError("Transfer checkpoint has incompatible physics/control timing")
    alignment = metadata.get("official_alignment", {})
    if (alignment.get("actor_dim"), alignment.get("critic_dim")) != (61, 70):
        raise ValueError("Transfer requires the 61-D Actor / 70-D Critic contract")
    checkpoint_policy_variant(metadata)
    return metadata


def initialize_transfer(runner, path, *, inherit_training_stage=False):
    """Restore learned weights, exploration and both normalizers without stale Adam moments."""
    import torch

    checkpoint = torch.load(path, map_location=runner.device, weights_only=True)
    for key in ("model_state_dict", "obs_norm_state_dict", "privileged_obs_norm_state_dict"):
        if key in checkpoint and not all(torch.isfinite(value).all() for value in checkpoint[key].values()):
            raise ValueError(f"Non-finite tensors in checkpoint {key}")
    from .cloth_policy import ClothHistoryActorCritic, migrate_normalizer, migrate_weights

    cloth = isinstance(runner.alg.policy, ClothHistoryActorCritic)
    converted = False
    feedback_reinitialized = False
    if cloth:
        metadata_path = Path(path).with_name("config.json")
        source_history = json.loads(metadata_path.read_text()).get("history") if metadata_path.exists() else None
        if source_history is not None and source_history.get("frame_dim") == 48:
            source_history = None  # Legacy flat history has no leg feedback channels.
        feedback_reinitialized = source_history is not None and source_history["version"] < 3
        converted = migrate_weights(runner.alg.policy, checkpoint["model_state_dict"], source_history=source_history)
    else:
        runner.alg.policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if cloth and runner.empirical_normalization:
        migrate_normalizer(runner.obs_normalizer, checkpoint["obs_norm_state_dict"])
        migrate_normalizer(runner.privileged_obs_normalizer, checkpoint["privileged_obs_norm_state_dict"])
    elif runner.empirical_normalization:
        runner.obs_normalizer.load_state_dict(checkpoint["obs_norm_state_dict"], strict=True)
        runner.privileged_obs_normalizer.load_state_dict(checkpoint["privileged_obs_norm_state_dict"], strict=True)
    # RSL-RL stores the last completed iteration index, not the next iteration.
    runner.current_learning_iteration = int(checkpoint["iter"]) + 1 if inherit_training_stage else 0
    return {
        "checkpoint": str(path),
        "source_iteration": int(checkpoint["iter"]),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "optimizer_restored": False,
        "network_expanded": converted,
        "feedback_reinitialized": feedback_reinitialized,
        "normalizers_restored": runner.empirical_normalization,
        "training_stage_inherited": inherit_training_stage,
        "start_iteration": runner.current_learning_iteration,
    }
