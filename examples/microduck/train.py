"""MicroDuck train entry point for MuJoCo and SynReal."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
from datetime import datetime

from examples.microduck.common import create_env, setup


def main(argv=None):
    args, training_config, play_cfg = setup("train", argv)
    env = create_env(args)
    try:
        from rsl_rl.runners import OnPolicyRunner

        from examples.microduck.config import ppo_config

        cfg = ppo_config(args.num_envs)
        from examples.microduck.official_ppo import apply_official_ppo

        cfg = apply_official_ppo(cfg, env.official.spec, args.num_envs)
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
        log_dir = (
            args.log_root / f"microduck_{args.solver}" / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f") + "_" + args.run_name)
        )
        log_dir.mkdir(parents=True)
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
        (log_dir / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        runner = OnPolicyRunner(env, cfg, str(log_dir), args.device)
        if args.resume:
            runner.load(str(args.resume))
            # RSL-RL 2.3 checkpoints store the last completed iteration index.
            runner.current_learning_iteration += 1
            if env.official is not None:
                # RSL-RL checkpoints record the completed PPO iteration; curricula count control steps.
                env.official.common_step_counter = runner.current_learning_iteration * cfg["num_steps_per_env"]
                # Apply the restored reset curriculum to the first episode, too.
                env.reset()
        print(f"Training directory: {log_dir.resolve()}", flush=True)
        runner.learn(args.iterations, init_at_random_ep_len=False)
        print(f"Training complete: {log_dir}", flush=True)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
