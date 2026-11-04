# Style3D Sim project navigation

Follow `AGENTS.md`; use a 128-character line width and English comments explaining non-obvious decisions.
Before changing MicroDuck, read `examples/microduck/AGENTS.md` and its linked `HANDOFF.md`.

## Architecture

- Python 3.11–3.12 package, built with scikit-build-core/CMake and pybind11.
- `src/style3d_sim/`: public Python API and Newton adapter for the native CUDA SynReal solver.
- `style3d-sim-bin/`: prebuilt native extension. The private solver source is not required for public installs.
- `examples/`: simulation and robot learning examples. `demos/`: complete showcase scenes.
- `examples/microduck/env.py`: shared training/playback environment, physics, and environment defaults.
- `examples/microduck/common.py`: CLI setup, checkpoint compatibility, authentication, environment construction.
- `examples/microduck/train.py` / `play.py`: RSL-RL PPO training and checkpoint replay.
- `examples/microduck/ui.py`: local Chinese Web workbench, launched with `python -m examples.microduck.ui`.
- `examples/microduck/configs/official_snapshot.json`: centralized task recipes and reward weights.
- `docs/`: usage documentation; `tests/` and `examples/microduck/tests/`: pytest suites.

## Development

Use the existing repository `.venv`; avoid rebuilding the native extension without a concrete need.
Run MicroDuck tests with `.venv/bin/python -m pytest -q examples/microduck/tests`.
Use `.venv/bin/ruff check --line-length 128 <changed files>` and `git diff --check`.
Tests needing native CUDA simulation or authentication have separate opt-in flags; inspect their requirements.
A CPU or configuration test does not validate SynReal dynamics or policy convergence.

## Working conventions

Inspect `git status --short` before changes. Preserve existing modified, deleted, and untracked files.
Do not edit bundled upstream reference projects without reading their own instructions.
Use explicit CLI overrides for experiments; do not duplicate environment defaults outside `env.py`.
Strict resume preserves checkpoint contracts; changed environments use the existing explicit transfer mechanism.
Keep credentials out of code, logs, configuration, and conversation. Do not start long training by default.
MicroDuck usage and handoff documentation belong in `HANDOFF.md`; do not create competing guides.
