"""Shared CLI validation, authentication and environment construction."""

import argparse
import getpass
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

from .task_registry import TASKS


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def setup(mode, argv=None):
    from .env import (
        ENVIRONMENT_CONFIGS,
        PhysicsConfig,
        environment_config,
        physics_matches,
        resolve_physics,
    )
    from .scenes import SCENE_CONFIG, SCENE_KINDS, SoftPadConfig

    parser = argparse.ArgumentParser(description=f"MicroDuck {mode}: MuJoCo / SynReal")
    parser.add_argument("--training-config", type=Path, help="JSON file containing CLI defaults and optional PPO overrides")
    parser.add_argument("--task", choices=TASKS, default=None)
    parser.add_argument("--recipe", choices=["official_adapted_v1"], default=None)
    parser.add_argument("--policy-variant", choices=["standard", "history"], default=None,
                        help="network for new training; resume/play infer it from the checkpoint")
    parser.add_argument("--solver", choices=["mujoco", "synreal"], default="mujoco")
    parser.add_argument("--environment-profile", choices=tuple(ENVIRONMENT_CONFIGS), default=None)
    parser.add_argument("--num-envs", type=positive_int)
    parser.add_argument("--physics-substeps", type=positive_int,
                        help="subdivide each policy step without changing the policy control period")
    parser.add_argument("--joint-constraint-primal-iterations", type=positive_int,
                        help="SynReal joint-constraint solve iterations; recorded in checkpoint physics")
    parser.add_argument("--use-checkpoint-environment", action="store_true",
                        help="Explicitly restore saved physics and terrain instead of the current env.py settings")
    parser.add_argument("--iterations", type=positive_int, default=5)
    parser.set_defaults(mode=mode)
    parser.add_argument("--checkpoint", type=Path, help="official-adapted checkpoint to load in play mode")
    parser.add_argument("--resume", type=Path, help="resume training, including optimizer and normalization state")
    parser.add_argument("--init-from", type=Path,
                        help="transfer a same-task SynReal Velocity/VelStand student; fresh PPO without teacher BC")
    parser.add_argument("--velocity-teacher", type=Path, help="frozen history Velocity checkpoint; also initializes VelStand")
    parser.add_argument("--standup-teacher", type=Path, help="frozen history StandUp checkpoint for VelStand BC")
    parser.add_argument("--inherit-training-stage", action="store_true",
                        help="with --init-from, inherit iteration and curriculum progress; keep a fresh optimizer")
    parser.add_argument("--terrain", choices=["default", "mixed_soft_pad", *SCENE_KINDS], default=None)
    parser.add_argument("--pad-probability", type=float, default=None,
                        help="fraction of fixed pad worlds (rounded to nearest count, ties up); pad worlds reset at center")
    parser.add_argument("--num-frames", type=positive_int, default=150)
    parser.add_argument("--command", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("VX", "VY", "YAW"),
                        help="initial velocity command for playback or the deterministic training diagnostic")
    if mode == "train":
        parser.add_argument("--diagnostic-no-exploration", action="store_true",
                            help="observe the training step with mean actions and frozen weights/normalizers; no PPO updates")
    parser.add_argument("--viewer", choices=["null", "gl"], default="gl" if mode == "play" else "null")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="smoke")
    parser.add_argument("--log-root", type=Path, default=Path("logs/rsl_rl"))
    parser.add_argument("--force-login", action="store_true", help="allow replacing another active SynReal session")
    parser.add_argument(
        "--self-collisions",
        action="store_true",
        help="enable full robot collision geometry and self-collision with shared MJCF masks on both solvers",
    )
    parser.add_argument(
        "--randomize-foot-friction",
        action=argparse.BooleanOptionalAction,
        help="sample one foot-ground friction value per environment at startup",
    )
    if mode == "play":
        parser.add_argument("--heading-hold", action="store_true",
                            help="steer a single Velocity checkpoint along its initial heading using pose feedback")
        parser.add_argument("--heading-max-yaw", type=float, default=1.0,
                            help="heading-hold yaw limit in rad/s; values above 1 exceed the saved training command range")
        parser.add_argument("--adaptive-forward", action="store_true",
                            help="with heading hold, adjust forward/yaw policy commands using measured progress")
        parser.add_argument("--target-forward-speed", type=float, default=0.10,
                            help="desired measured forward speed for adaptive playback, m/s; not a tracking guarantee")
        parser.add_argument("--record-dir", type=Path,
                            help="save this checkpoint replay as animated USD for offline Blender rendering")
        parser.add_argument("--record-until-pad-edge", action="store_true",
                            help="stop recording when the robot root reaches the soft-pad boundary")
        parser.add_argument(
            "--fall-reset",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="legacy option, ignored: playback only resets manually with R",
        )
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
        parser.add_argument("--benchmark", action="store_true", help="measure playback without real-time pacing")
        parser.add_argument(
            "--keep-open",
            action="store_true",
            help="keep the GL viewer open after the requested playback frames; close the window to exit",
        )
        parser.add_argument(
            "--diagnostics",
            action="store_true",
            help="print policy command sensitivity and measured base velocity during replay",
        )
    preliminary, _ = parser.parse_known_args(argv)
    training_config = {}
    if preliminary.training_config:
        training_config = json.loads(preliminary.training_config.read_text(encoding="utf-8"))
        defaults = training_config.get("training", {})
        allowed = {action.dest for action in parser._actions} - {
            "help", "training_config", "checkpoint", "resume", "init_from", "mode", "use_checkpoint_environment",
            "num_envs", "terrain", "pad_probability", "randomize_foot_friction",
        }
        if set(defaults) - allowed:
            parser.error(f"unknown or relocated training defaults: {sorted(set(defaults) - allowed)}; "
                         "edit environment settings in env.py or pass explicit CLI options")
        parser.set_defaults(**defaults)
    preliminary, _ = parser.parse_known_args(argv)
    saved_checkpoint = preliminary.checkpoint if mode == "play" else preliminary.resume
    metadata = None
    if saved_checkpoint is not None:
        if not saved_checkpoint.is_file():
            parser.error(f"checkpoint does not exist: {saved_checkpoint}")
        metadata = json.loads(saved_checkpoint.with_name("config.json").read_text(encoding="utf-8"))
    saved_terrain = (metadata or {}).get("terrain", {}).get("kind", "default")
    inferred_profile = (saved_terrain if saved_terrain in ENVIRONMENT_CONFIGS else
                        preliminary.terrain if preliminary.terrain in ENVIRONMENT_CONFIGS else "official")
    profile_name = preliminary.environment_profile or inferred_profile
    try:
        profile = environment_config(profile_name)
    except ValueError as exc:
        parser.error(str(exc))
    environment_defaults = profile.cli_defaults(mode)
    pad_config = asdict(profile.pad)
    pad_sampling = profile.pad_sampling
    if preliminary.use_checkpoint_environment:
        if metadata is None:
            parser.error("--use-checkpoint-environment requires --checkpoint or --resume")
        environment_defaults.update(
            terrain=saved_terrain,
            randomize_foot_friction=metadata.get("randomize_foot_friction", profile.randomize_foot_friction),
        )
        if saved_terrain == "mixed_soft_pad":
            environment_defaults["pad_probability"] = metadata["terrain"]["pad_probability"]
            pad_sampling = metadata["terrain"].get("sampling", "independent_per_episode")
            pad_config = asdict(SoftPadConfig.from_checkpoint(metadata["terrain"]))
    parser.set_defaults(**environment_defaults, environment_profile=profile_name)
    args, extra = parser.parse_known_args(argv)
    if args.resume is not None and args.mode != "train":
        parser.error("--resume requires training mode")
    if args.init_from is not None and (mode != "train" or args.resume is not None):
        parser.error("--init-from requires training mode and cannot be combined with --resume")
    if args.inherit_training_stage and (mode != "train" or args.init_from is None):
        parser.error("--inherit-training-stage requires --init-from in training mode")
    if extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    if metadata is not None:
        from .checkpoint import checkpoint_policy_variant

        try:
            saved_variant = checkpoint_policy_variant(metadata)
        except ValueError as exc:
            parser.error(str(exc))
        if args.policy_variant is not None and args.policy_variant != saved_variant:
            parser.error(f"checkpoint uses {saved_variant}; cannot load it as {args.policy_variant}")
        args.policy_variant = saved_variant
    profile = environment_config(args.environment_profile)
    args.physics = resolve_physics(profile, metadata, use_checkpoint=args.use_checkpoint_environment)
    if args.physics_substeps is not None:
        args.physics = args.physics.with_substeps(args.physics_substeps)
    if args.joint_constraint_primal_iterations is not None:
        args.physics = replace(args.physics, joint_constraint_primal_iterations=args.joint_constraint_primal_iterations)
    args.pad_config = pad_config
    args.pad_sampling = pad_sampling
    args.environment_source = "checkpoint + CLI" if args.use_checkpoint_environment else "env.py + CLI"
    if mode == "play":
        if args.allow_cross_backend and args.checkpoint is None:
            parser.error("--allow-cross-backend requires --checkpoint")
        if args.checkpoint and args.policy:
            parser.error("select either --checkpoint or --policy walk")
        if args.checkpoint is None:
            if args.terrain in SCENE_KINDS:
                parser.error("Optional scenes require a trained --checkpoint for playback")
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
            from .recipe import recipe_fingerprint

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
        if mode == "train":
            changed = not physics_matches(args.physics, PhysicsConfig.from_checkpoint(metadata))
            changed |= args.terrain != saved_terrain
            if "randomize_foot_friction" in metadata:
                changed |= args.randomize_foot_friction != metadata["randomize_foot_friction"]
            if saved_terrain == "mixed_soft_pad":
                changed |= args.pad_probability != metadata["terrain"]["pad_probability"]
                changed |= args.pad_sampling != metadata["terrain"].get("sampling", "independent_per_episode")
                changed |= (json.dumps(asdict(SoftPadConfig(**args.pad_config)), sort_keys=True)
                            != json.dumps(asdict(SoftPadConfig.from_checkpoint(metadata["terrain"])), sort_keys=True))
            elif saved_terrain in SCENE_KINDS:
                # JSON saves tuples as lists; compare serialized values without relaxing material checks.
                changed |= (json.dumps(metadata.get("terrain", {}).get("config"), sort_keys=True)
                            != json.dumps(asdict(SCENE_CONFIG), sort_keys=True))
            if changed:
                parser.error("Resume environment differs from the checkpoint. Use --use-checkpoint-environment "
                             "for saved dynamics, or --init-from for a changed environment.")
        play_cfg = metadata["ppo"]
    args.recipe = args.recipe or "official_adapted_v1"
    args.task = args.task or "velocity"
    args.terrain = args.terrain or "default"
    args.pad_probability = 0.5 if args.pad_probability is None else args.pad_probability
    if not 0.0 <= args.pad_probability <= 1.0:
        parser.error("--pad-probability must be between 0 and 1")
    if args.terrain != "default" and args.solver != "synreal":
        parser.error("Non-default terrain scenes require SynReal")
    if args.init_from:
        from .checkpoint import resolve_checkpoint, transfer_metadata

        if args.solver != "synreal" or args.task not in ("velocity", "velstand") or args.self_collisions:
            parser.error("--init-from supports SynReal Velocity/VelStand without self collisions")
        args.init_from = resolve_checkpoint(args.init_from)
        source_metadata = transfer_metadata(args.init_from, task=args.task)
        from .checkpoint import checkpoint_policy_variant

        source_variant = checkpoint_policy_variant(source_metadata)
        if args.policy_variant is not None and args.policy_variant != source_variant:
            parser.error("--init-from requires the same policy variant; network conversion is not supported")
        args.policy_variant = source_variant
    args.policy_variant = args.policy_variant or "standard"
    args.teacher_bc = None
    if args.velocity_teacher or args.standup_teacher:
        if not (args.velocity_teacher and args.standup_teacher):
            parser.error("provide both --velocity-teacher and --standup-teacher")
        if (mode != "train" or args.resume or args.init_from or args.task != "velstand"
                or args.solver != "synreal" or args.terrain != "default" or args.self_collisions):
            parser.error("teacher initialization requires fresh SynReal flat velstand without self collisions")
        if args.policy_variant != "history":
            parser.error("teacher initialization requires --policy-variant history")
        from .distillation import teacher_config

        try:
            args.teacher_bc = teacher_config(args.velocity_teacher, args.standup_teacher)
        except (ValueError, KeyError, OSError) as exc:
            parser.error(str(exc))
    if mode == "play" and args.policy == "walk" and args.policy_variant != "standard":
        parser.error("--policy-variant history requires a trained --checkpoint")
    if mode == "play" and args.policy == "walk" and args.task != "velocity":
        parser.error("--task requires a trained --checkpoint for non-velocity playback")
    if (mode == "train" and args.diagnostic_no_exploration
            and (not (args.init_from or args.resume) or args.task != "velocity")):
        parser.error("--diagnostic-no-exploration requires a velocity checkpoint via --init-from or --resume")
    if mode == "play":
        if args.fall_reset:
            print("--fall-reset is ignored: playback resets only when R is pressed.", flush=True)
        args.fall_reset = False
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
        randomize_foot_friction=getattr(args, "randomize_foot_friction", False),
        terrain=getattr(args, "terrain", "default"),
        pad_probability=getattr(args, "pad_probability", 0.5),
        pad_config=getattr(args, "pad_config", None),
        pad_sampling=getattr(args, "pad_sampling", None),
        physics=getattr(args, "physics", None),
        reset_on_fall=args.mode != "play",
    )
    env.cfg["environment_profile"] = args.environment_profile
    env.cfg["environment_source"] = args.environment_source
    if args.mode == "play":
        print("Playback resets: manual only (R resets all environments; --fall-reset is ignored)", flush=True)
    from .observations import configure_policy_observations

    configure_policy_observations(env, getattr(args, "policy_variant", "standard"))
    print(f"Environment: {args.environment_source}, profile={args.environment_profile}, "
          f"terrain={args.terrain}, num_envs={args.num_envs}, pad_probability={args.pad_probability:g}, "
          f"randomize_foot_friction={args.randomize_foot_friction}", flush=True)
    if env.mixed_terrain:
        print(f"Pad settings: {json.dumps(args.pad_config, sort_keys=True)}", flush=True)
        cells = env.terrain.config.resolved_cells
        print(f"Pad mesh: cells={cells}, tetrahedra_per_pad={6 * cells[0] * cells[1] * cells[2]}", flush=True)
    print(
        f"MicroDuck: solver={args.solver}, task={args.task}, recipe={args.recipe}, "
        f"envs={args.num_envs}, self_collisions={env.cfg['self_collisions']}, "
        f"obs={env.num_obs}, critic={env.num_privileged_obs}, actions=14",
        flush=True,
    )
    if args.solver == "synreal":
        print(f"Physics: dt={env.physics_dt:g}, substeps={env.decimation}, "
              f"nonlinear={env.physics.nonlinear_iterations}, linear={env.physics.linear_iterations}, "
              f"joint_primal={env.physics.joint_constraint_primal_iterations}", flush=True)
    if env.official is not None:
        print("Official port differences: " + "; ".join(env.official.alignment["differences"]), flush=True)
        print("Omitted reward terms: " + ", ".join(env.official.omitted), flush=True)
    return env
