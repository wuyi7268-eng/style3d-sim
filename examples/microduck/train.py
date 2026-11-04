"""MicroDuck train entry point for MuJoCo and SynReal."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
from datetime import datetime

from examples.microduck.common import create_env, setup


def run_deterministic_diagnostic(env, runner, args, checkpoint_rollout_steps):
    """Exercise env.step directly while matching playback's policy inputs and reset stage."""
    import torch

    from examples.microduck.play import update_playback_observation, update_velocity_command

    env.official.observations.spec["actor"]["enable_corruption"] = False
    runner.load(str(args.init_from or args.resume), load_optimizer=False)
    env.official.common_step_counter = (runner.current_learning_iteration + 1) * checkpoint_rollout_steps
    env.reset()
    # eval_mode freezes running normalization statistics as well as the policy.
    policy = runner.get_inference_policy(device=args.device)
    obs, _ = env.get_observations()
    command, key_was_down, previous_command = list(args.command), {}, None
    frames = 0
    print("TRAIN DIAGNOSTIC: mean actions; weights/normalizers frozen; PPO updates=0; no checkpoints saved.", flush=True)
    print("Matching playback: actor noise and interval events disabled; head command and reset stage aligned.", flush=True)
    with torch.inference_mode():
        for _ in range(args.num_frames):
            if env.viewer is not None and not env.viewer.is_running():
                break
            previous_command = update_velocity_command(env, env.viewer, command, key_was_down, previous_command)
            update_playback_observation(env, obs)
            # Keep the training environment step, including rewards, sensors,
            # terminations and resets. Do not enter the playback step wrapper.
            obs, _, _, _ = env.step(policy(obs), apply_interval_events=False, apply_viewer_forces=True)
            frames += 1
            if frames == 1 or frames % 100 == 0:
                print(f"Train diagnostic: frames={frames}, resets={env.total_resets}, PPO updates=0", flush=True)
    print(f"Train diagnostic complete: frames={frames}, resets={env.total_resets}, PPO updates=0", flush=True)


def main(argv=None):
    args, training_config, play_cfg = setup("train", argv)
    env = create_env(args)
    try:
        from examples.microduck.ppo import (
            apply_official_ppo,
            create_runner,
            ppo_config,
        )

        cfg = ppo_config(args.num_envs, args.policy_variant)
        cfg = apply_official_ppo(cfg, env.official.spec, args.num_envs)
        if args.teacher_bc:
            cfg["policy"] = args.teacher_bc["walk"]["policy"].copy()
            cfg["teacher_bc"] = args.teacher_bc
        if args.init_from:
            from examples.microduck.checkpoint import transfer_metadata

            source_metadata = transfer_metadata(args.init_from, task=args.task)
            cfg["policy"] = source_metadata["ppo"]["policy"].copy()
            cfg["empirical_normalization"] = source_metadata["ppo"]["empirical_normalization"]
        # Resume the exact policy/optimizer configuration recorded with the checkpoint.
        if args.resume:
            cfg = play_cfg
        else:
            for key, value in training_config.get("ppo", {}).items():
                if key not in cfg:
                    raise ValueError(f"unknown PPO setting: {key}")
                if isinstance(value, dict) and isinstance(cfg[key], dict):
                    cfg[key].update(value)
                else:
                    cfg[key] = value
        expected_class = "HistoryActorCritic" if args.policy_variant == "history" else "ActorCritic"
        if cfg["policy"]["class_name"] != expected_class:
            raise ValueError("PPO policy class must match --policy-variant")
        if cfg.get("teacher_bc"):
            from examples.microduck.distillation import validate_environment

            validate_environment(cfg["teacher_bc"], env)
            env.official.alignment["differences"] = [
                difference.replace("; no expert BC", "; local frozen history teachers with PPO + BC")
                for difference in env.official.alignment["differences"]
            ]
            env.official.alignment["expert_bc"] = "synreal_history_dual_teacher_v2"
        if args.inherit_training_stage and cfg["num_steps_per_env"] != source_metadata["ppo"]["num_steps_per_env"]:
            # Curricula and later resumes derive control steps from iteration * rollout length.
            raise ValueError("--inherit-training-stage requires the checkpoint's num_steps_per_env")
        if args.diagnostic_no_exploration:
            # Match playback's runner construction too: policy initialization may
            # consume RNG before the checkpoint is loaded and the scene is reset.
            diagnostic_cfg = source_metadata["ppo"] if args.init_from else cfg
            runner = create_runner(env, diagnostic_cfg, log_dir=None, device=args.device)
            run_deterministic_diagnostic(env, runner, args, diagnostic_cfg["num_steps_per_env"])
            return 0
        log_dir = (
            args.log_root / f"microduck_{args.solver}" / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f") + "_" + args.run_name)
        )
        log_dir.mkdir(parents=True)
        if env.mixed_terrain:
            env.terrain.guard.diagnostic_dir = log_dir / "diagnostics"
        metadata = {
            **env.cfg,
            "seed": args.seed,
            "num_envs": args.num_envs,
            "iterations": args.iterations,
            "planned_samples": args.num_envs * cfg["num_steps_per_env"] * args.iterations,
            "resume_from": str(args.resume) if args.resume else None,
            "ppo": cfg,
            "observation_contract": env.cfg["observation_contract"],
            "deployment_compatible": False,
        }
        # RSL-RL pops class names and injects runtime objects into its configuration.
        # Keep the serializable recipe intact for checkpoint playback and resume.
        runner = create_runner(env, cfg, str(log_dir), args.device)
        if args.teacher_bc:
            from examples.microduck.distillation import warm_start

            metadata["velstand_warm_start"] = warm_start(runner, cfg["teacher_bc"])
            info = metadata["velstand_warm_start"]
            print(f"VelStand: Velocity history warm start; PPO optimizer={info['warm_start_optimizer']}; "
                  f"PPO LR={info['effective_ppo_learning_rate']:.6g}; fresh BC optimizer and curricula.", flush=True)
        if args.init_from:
            from examples.microduck.checkpoint import initialize_transfer

            metadata["transfer_from"] = initialize_transfer(
                runner, args.init_from, inherit_training_stage=args.inherit_training_stage
            )
            metadata["transfer_from"]["source_recipe_fingerprint"] = source_metadata["official_alignment"]["recipe_fingerprint"]
            print(f"Transfer initialization: {args.init_from}; optimizer reset; both normalizers restored", flush=True)
        (log_dir / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if args.resume:
            runner.load(str(args.resume))
            # RSL-RL 2.3 checkpoints store the last completed iteration index.
            runner.current_learning_iteration += 1
        if (args.resume or args.inherit_training_stage) and env.official is not None:
            # RSL-RL checkpoints record the completed PPO iteration; curricula count control steps.
            env.official.common_step_counter = runner.current_learning_iteration * cfg["num_steps_per_env"]
            # Apply the restored reset curriculum to the first episode, too.
            env.reset()
        if args.inherit_training_stage:
            print(f"Inherited training stage: iteration={runner.current_learning_iteration}, "
                  f"control_steps={env.official.common_step_counter}; current reward/curriculum configuration", flush=True)
        print(f"Training directory: {log_dir.resolve()}", flush=True)
        runner.learn(args.iterations, init_at_random_ep_len=False)
        print(f"Training complete: {log_dir}", flush=True)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
