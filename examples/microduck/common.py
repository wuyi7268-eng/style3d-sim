"""Shared CLI validation, authentication and environment construction."""

import argparse
import getpass
import json
import os
from pathlib import Path


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def setup(mode, argv=None):
    parser = argparse.ArgumentParser(description=f"MicroDuck {mode}: MuJoCo / SynReal")
    parser.add_argument("--training-config", type=Path, help="JSON file containing CLI defaults and optional PPO overrides")
    parser.add_argument("--task", choices=["velocity", "velstand", "groundpick", "ballkick"], default=None)
    parser.add_argument("--recipe", choices=["official_adapted_v1"], default=None)
    parser.add_argument("--solver", choices=["mujoco", "synreal"], default="mujoco")
    parser.add_argument("--num-envs", type=positive_int, default=64 if mode == "train" else 1)
    parser.add_argument("--iterations", type=positive_int, default=5)
    parser.set_defaults(mode=mode)
    parser.add_argument("--checkpoint", type=Path, help="official-adapted checkpoint to load in play mode")
    parser.add_argument("--resume", type=Path, help="resume training, including optimizer and normalization state")
    parser.add_argument("--num-frames", type=positive_int, default=150)
    parser.add_argument("--viewer", choices=["null", "gl"], default="gl" if mode == "play" else "null")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="smoke")
    parser.add_argument("--log-root", type=Path, default=Path("logs/rsl_rl"))
    parser.add_argument("--force-login", action="store_true", help="allow replacing another active SynReal session")
    parser.add_argument(
        "--self-collisions",
        action="store_true",
        help="enable robot self-collision in training or playback",
    )
    if mode == "play":
        parser.add_argument("--policy", choices=["walk"], help="official walk.onnx policy; default when no checkpoint is given")
        parser.add_argument(
            "--allow-cross-backend",
            action="store_true",
            help="allow replaying a checkpoint with a different physics backend",
        )
        parser.add_argument(
            "--allow-recipe-mismatch",
            action="store_true",
            help="allow playback when current recipe files differ from the checkpoint",
        )
        parser.add_argument("--disable-cuda-graph", action="store_true", help="disable playback CUDA capture for comparison")
        parser.add_argument("--full-step", action="store_true", help="use the full training step for playback comparison")
        parser.add_argument("--benchmark", action="store_true", help="measure playback without real-time pacing")
        parser.add_argument(
            "--diagnostics",
            action="store_true",
            help="print policy command sensitivity and measured base velocity during replay",
        )
        parser.add_argument(
            "--command",
            type=float,
            nargs=3,
            default=(0.0, 0.0, 0.0),
            metavar=("VX", "VY", "YAW"),
            help="initial velocity command; each key press adds one velocity increment",
        )
    preliminary, _ = parser.parse_known_args(argv)
    training_config = {}
    if preliminary.training_config:
        training_config = json.loads(preliminary.training_config.read_text(encoding="utf-8"))
        defaults = training_config.get("training", {})
        allowed = {action.dest for action in parser._actions} - {"help", "training_config", "checkpoint", "resume", "mode"}
        if set(defaults) - allowed:
            parser.error(f"unknown training defaults: {sorted(set(defaults) - allowed)}")
        parser.set_defaults(**defaults)
    args, extra = parser.parse_known_args(argv)
    if args.resume is not None and args.mode != "train":
        parser.error("--resume requires training mode")
    if extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    if mode == "play":
        if args.allow_cross_backend and args.checkpoint is None:
            parser.error("--allow-cross-backend requires --checkpoint")
        if args.checkpoint and args.policy:
            parser.error("select either --checkpoint or --policy walk")
        if args.checkpoint is None:
            args.policy = "walk"
            if args.num_envs != 1:
                parser.error("official walk replay supports --num-envs 1")
    if args.checkpoint is not None and args.mode != "play":
        parser.error("--checkpoint is only supported by play")
    play_cfg = None
    saved_checkpoint = args.checkpoint if args.mode == "play" else args.resume
    if saved_checkpoint is not None:
        if not saved_checkpoint.is_file():
            parser.error(f"checkpoint does not exist: {saved_checkpoint}")
        metadata = json.loads(saved_checkpoint.with_name("config.json").read_text(encoding="utf-8"))
        if metadata.get("observation_contract") != "official_asymmetric_pd_v2":
            parser.error("legacy or unsupported checkpoint; only official_asymmetric_pd_v2 is supported")
        checkpoint_solver = metadata.get("solver")
        if checkpoint_solver != args.solver and not (mode == "play" and args.allow_cross_backend):
            parser.error(
                "select the checkpoint's training solver with --solver"
                + (" or enable --allow-cross-backend for playback" if mode == "play" else "")
            )
        if checkpoint_solver != args.solver:
            print(
                f"Cross-backend playback enabled: checkpoint={checkpoint_solver}, runtime={args.solver}",
                flush=True,
            )
        saved_task = metadata.get("task", "velocity")
        if args.task is not None and args.task != saved_task:
            parser.error(f"checkpoint task is {saved_task}; cannot load it as {args.task}")
        saved_recipe = metadata.get("recipe")
        if saved_recipe != "official_adapted_v1":
            parser.error("only official_adapted_v1 checkpoints are supported")
        if args.recipe is not None and args.recipe != saved_recipe:
            parser.error(f"checkpoint recipe is {saved_recipe}; changing rewards during resume is not supported")
        args.recipe = saved_recipe
        if saved_recipe == "official_adapted_v1":
            from .official_recipe import recipe_fingerprint

            if metadata.get("official_alignment", {}).get("recipe_fingerprint") != recipe_fingerprint():
                if mode != "play" or not args.allow_recipe_mismatch:
                    parser.error(
                        "official recipe files differ from the checkpoint; use its original recipe files "
                        "or pass --allow-recipe-mismatch for diagnostic playback"
                    )
                print(
                    "Recipe fingerprint differs from the checkpoint; replay uses the current recipe files.",
                    flush=True,
                )
        args.task = saved_task
        play_cfg = metadata["ppo"]
    args.recipe = args.recipe or "official_adapted_v1"
    args.task = args.task or "velocity"
    if mode == "play" and args.policy == "walk" and args.task != "velocity":
        parser.error("--task requires a trained --checkpoint for non-velocity playback")
    return args, training_config, play_cfg


def create_env(args):
    if args.solver == "synreal":
        import style3d_sim as s3d

        if not s3d.is_logged_in():
            account = os.environ.get("STYLE3D_ACCOUNT") or getpass.getpass("Style3D account (hidden): ")
            password = os.environ.get("STYLE3D_PASSWORD") or getpass.getpass("Style3D password (hidden): ")
            try:
                if not s3d.login(account, password, args.force_login):
                    raise RuntimeError("SynReal authentication failed")
            finally:
                del account, password
    from examples.microduck.env import MicroDuckEnv

    viewer = None
    if args.viewer == "gl":
        from newton.viewer import ViewerGL

        viewer = ViewerGL()
    env = MicroDuckEnv(
        args.solver,
        args.num_envs,
        args.device,
        args.seed,
        viewer,
        task=args.task,
        recipe=args.recipe,
        enable_self_collisions=getattr(args, "self_collisions", False),
    )
    print(
        f"MicroDuck: solver={args.solver}, task={args.task}, recipe={args.recipe}, "
        f"envs={args.num_envs}, self_collisions={env.cfg['self_collisions']}, "
        f"obs={env.num_obs}, critic={env.num_privileged_obs}, actions=14",
        flush=True,
    )
    if env.official is not None:
        print("Official port differences: " + "; ".join(env.official.alignment["differences"]), flush=True)
        print("Omitted reward terms: " + ", ".join(env.official.omitted), flush=True)
    return env
