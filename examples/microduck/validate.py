"""MicroDuck validate entry point for MuJoCo and SynReal."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json

import torch

from examples.microduck.common import create_env, setup


def main(argv=None):
    args, _training_config, _play_cfg = setup("simulate", argv)
    env = create_env(args)
    try:
        min_z, min_upright = float("inf"), float("inf")
        for _ in range(args.num_frames):
            _, _, _, extras = env.step(torch.zeros(args.num_envs, 14, device=args.device))
            min_z = min(min_z, extras["physics"]["min_trunk_z"])
            min_upright = min(min_upright, extras["physics"]["min_upright_cos"])
        print(
            json.dumps(
                {
                    "frames": args.num_frames,
                    "min_trunk_z": min_z,
                    "min_upright_cos": min_upright,
                    "resets": env.total_resets,
                }
            ),
            flush=True,
        )
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
