"""Run MicroDuck MuJoCo Warp PPO training from the Style3D Sim example directory.

Install the copied training project first; see examples/microduck/README.md.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "examples" / "microduck" / "microduck_rl-develop"


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Mjlab-Velocity-Flat-MicroDuck")
    parser.add_argument("--num-envs", type=positive_int, default=64)
    parser.add_argument("--iterations", type=positive_int, default=5)
    parser.add_argument("--run-name", default="style3d_smoke")
    parser.add_argument("--list-tasks", action="store_true")
    args, extra = parser.parse_known_args()
    if extra[:1] == ["--"]:
        extra = extra[1:]

    executable = PROJECT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not executable.is_file():
        parser.error(f"training environment missing; run: uv sync --project {PROJECT.relative_to(ROOT)} --locked")

    # mjlab pins Torch/Warp/RSL-RL versions that differ from Style3D's; always use its own interpreter.
    environment = os.environ.copy()
    environment["VIRTUAL_ENV"] = str(PROJECT / ".venv")
    environment["PATH"] = str(executable.parent) + os.pathsep + environment.get("PATH", "")
    environment["PYTHONUNBUFFERED"] = "1"
    if args.list_tasks:
        command = [str(executable), "-m", "mjlab.scripts.list_envs", *extra]
    else:
        command = [
            str(executable),
            "-m",
            "mjlab_microduck.train_cli",
            args.task,
            "--env.scene.num-envs",
            str(args.num_envs),
            "--agent.max-iterations",
            str(args.iterations),
            "--agent.run-name",
            args.run_name,
            "--agent.logger",
            "tensorboard",
            *extra,
        ]
    print("Backend: MuJoCo Warp; training project:", PROJECT, flush=True)
    print("Command:", subprocess.list2cmdline(command), flush=True)
    # Keep checkpoints under the top-level project's ignored logs/ directory.
    return subprocess.call(command, cwd=ROOT, env=environment)


if __name__ == "__main__":
    sys.exit(main())
