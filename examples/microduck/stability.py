"""Fail fast on implausible simulator states before they contaminate PPO rollouts."""

from datetime import datetime
from pathlib import Path

import torch
import warp as wp

# Diagnostic limits, deliberately far beyond MicroDuck's intended motions.
# These are not reward clipping, termination curricula, or a solver-stability fix.
MAX_POSITION = 1000.0
MAX_LINEAR_SPEED = 100.0
MAX_ANGULAR_SPEED = 1000.0


def unstable_worlds(pose, velocity):
    finite = torch.isfinite(pose).all(dim=(-2, -1)) & torch.isfinite(velocity).all(dim=(-2, -1))
    excessive = pose[..., :3].abs().amax(dim=(-2, -1)) > MAX_POSITION
    excessive |= velocity[..., :3].abs().amax(dim=(-2, -1)) > MAX_LINEAR_SPEED
    excessive |= velocity[..., 3:].abs().amax(dim=(-2, -1)) > MAX_ANGULAR_SPEED
    return ~finite | excessive


def check_physics(env, substep):
    env._sync()
    pose = wp.to_torch(env.states[0].body_q).reshape(env.num_envs, -1, 7)
    velocity = wp.to_torch(env.states[0].body_qd).reshape(env.num_envs, -1, 6)
    bad = unstable_worlds(pose, velocity)
    if not bad.any():
        return
    ids = bad.nonzero().flatten()
    selected = ids[:16]
    report = {
        "solver": env.backend,
        "task": env.task,
        "control_step": env.official.common_step_counter,
        "substep": substep,
        "physics_dt": env.physics_dt,
        "shared_world_count": getattr(env.solver, "_shared_world_count", None),
        "bad_env_ids": ids.cpu(),
        "sample_env_ids": selected.cpu(),
        "body_q": pose[selected].cpu(),
        "body_qd": velocity[selected].cpu(),
        "previous_body_q": wp.to_torch(env.states[1].body_q).reshape(env.num_envs, -1, 7)[selected].cpu(),
        "previous_body_qd": wp.to_torch(env.states[1].body_qd).reshape(env.num_envs, -1, 6)[selected].cpu(),
        "joint_targets": wp.to_torch(env.control.joint_target_q).reshape(env.num_envs, -1)[selected].cpu(),
        "actions": env.actions[selected].cpu(),
        "commands": env.commands[selected].cpu(),
        "episode_length": env.episode_length_buf[selected].cpu(),
        "limits": {"position": MAX_POSITION, "linear_speed": MAX_LINEAR_SPEED, "angular_speed": MAX_ANGULAR_SPEED},
    }
    directory = Path("logs/microduck_diagnostics")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_unstable.pt")
    torch.save(report, path)
    raise FloatingPointError(
        f"{env.backend}: unstable physics in {len(ids)} environments before reward/PPO; "
        f"control_step={report['control_step']}, substep={substep}; diagnostic={path.resolve()}"
    )
