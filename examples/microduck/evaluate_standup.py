"""Evaluate a local StandUp checkpoint separately on each upstream reset distribution."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from .common import create_env, positive_int, setup
from .events import scheduled_params
from .task_registry import base_task

START_POSES = ("standing", "sitting", "face_down", "face_up")


def select_start_pose(recipe, pose):
    """Keep the official pose noise/height, replacing only its categorical sampling weights."""
    if pose not in START_POSES:
        raise ValueError(f"Unknown stand-up start pose: {pose}")
    params = scheduled_params(recipe, "set_ground_state", "events")
    for name in START_POSES:
        params[f"{name}_prob"] = float(name == pose)
    recipe.spec["events"]["set_ground_state"]["params"] = params
    # Otherwise the reset-time curriculum would overwrite the evaluation bucket.
    recipe.spec["curricula"] = {
        name: term for name, term in recipe.spec["curricula"].items()
        if not (term["func"].endswith(".event_param_curriculum")
                and term["params"]["event_name"] == "set_ground_state")
    }


def standing_mask(height, gravity, foot_contact):
    """Require upright trunk height and both supporting feet, not a transient head-supported pose."""
    return (height >= 0.10) & (-gravity[:, 2] >= math.cos(math.radians(20))) & (foot_contact > 0).all(-1)


def evaluate(env, policy, poses=START_POSES, episodes=3, hold_seconds=0.5):
    if env.family != "standup":
        raise ValueError("Stand-up evaluation requires the standup task")
    if not math.isfinite(hold_seconds) or hold_seconds <= 0:
        raise ValueError("hold_seconds must be finite and positive")
    horizon = env.max_episode_length
    hold_steps = math.ceil(hold_seconds / env.step_dt)
    if hold_steps > horizon:
        raise ValueError("The standing hold cannot exceed the episode duration")
    env.official.observations.spec["actor"]["enable_corruption"] = False
    # Evaluate neutral head/body/twist commands. Calibration offsets and sensor delays remain as trained.
    env.official.update_commands = lambda ids=None: env.commands.zero_()
    env.commands.zero_()
    # Sample the final physical frame before auto-reset can replace it with a standing spawn.
    env.max_episode_length = horizon + 1
    report = {}
    try:
        with torch.inference_mode():
            for pose in poses:
                select_start_pose(env.official, pose)
                successes, settled, times = 0, 0, []
                for _ in range(episodes):
                    env.reset()
                    obs, _ = env.get_observations()
                    streak = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
                    reached = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
                    active = torch.ones_like(reached)
                    for frame in range(horizon):
                        if env.viewer is not None and not env.viewer.is_running():
                            raise RuntimeError("Evaluation interrupted before all trials completed")
                        obs, _, done, _ = env.step(policy(obs), apply_interval_events=False)
                        active &= ~done
                        _, _, root, _, _, gravity = env._kinematics()
                        height = root[:, 2] - env.terrain.origins[:, 2]
                        standing = active & standing_mask(height, gravity, env.official.sensors.feet.data.found)
                        streak = torch.where(standing, streak + 1, 0)
                        first = (streak >= hold_steps) & ~reached
                        times.extend([(frame + 1) * env.step_dt] * int(first.sum()))
                        reached |= first
                    successes += int(reached.sum())
                    settled += int((streak >= hold_steps).sum())
                trials = env.num_envs * episodes
                report[pose] = {
                    "trials": trials,
                    "recovered": successes,
                    "recovery_rate": successes / trials,
                    "standing_at_end": settled,
                    "standing_at_end_rate": settled / trials,
                    "mean_time_to_held_stand_s": sum(times) / len(times) if times else None,
                }
                print(f"StandUp {pose}: {json.dumps(report[pose])}", flush=True)
    finally:
        env.max_episode_length = horizon
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--start-pose", choices=("all", *START_POSES), default="all")
    parser.add_argument("--episodes", type=positive_int, default=3, help="batches of six-second trials per start pose")
    parser.add_argument("--hold-seconds", type=float, default=0.5)
    parser.add_argument("--output", type=Path, help="write the evaluation report as JSON")
    options, remaining = parser.parse_known_args(argv)
    if "--help" in remaining or "-h" in remaining:
        parser.print_help()
    args, _, ppo = setup("play", remaining)
    if args.checkpoint is None or base_task(args.task) != "standup":
        parser.error("use --checkpoint <standup/model_N.pt>")
    if not math.isfinite(options.hold_seconds) or not 0 < options.hold_seconds <= 6:
        parser.error("--hold-seconds must be in (0, 6]")
    env = create_env(args)
    try:
        from .ppo import create_runner

        runner = create_runner(env, ppo, log_dir=None, device=args.device)
        runner.load(str(args.checkpoint), load_optimizer=False)
        env.official.common_step_counter = (runner.current_learning_iteration + 1) * ppo["num_steps_per_env"]
        poses = START_POSES if options.start_pose == "all" else (options.start_pose,)
        result = evaluate(env, runner.get_inference_policy(device=args.device), poses, options.episodes, options.hold_seconds)
        report = {
            "checkpoint": str(args.checkpoint.resolve()),
            "solver": args.solver,
            "seed": args.seed,
            "environment": env.cfg,
            "criteria": {"min_trunk_height_m": 0.10, "max_tilt_deg": 20, "both_feet": True,
                         "hold_seconds": options.hold_seconds, "trial_seconds": env.max_episode_length * env.step_dt},
            "evaluation": {"commands": "zero", "actor_noise": False, "interval_pushes": False,
                           "pose_distribution": "upstream bucket including authored tilt and joint noise"},
            "results": result,
        }
        if options.output:
            options.output.parent.mkdir(parents=True, exist_ok=True)
            options.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"Evaluation report: {options.output}", flush=True)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
