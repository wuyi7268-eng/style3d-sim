# Troubleshooting

## Cannot Import `_core`

Typical error: `ModuleNotFoundError: No module named 'style3d_sim._core'`.

1. Complete [Installation](getting-started/installation.en.md) with `uv pip install -e .` or `uv sync`.
2. Confirm the extension is in the current virtual environment's `site-packages/style3d_sim/`.
3. Match the extension's Python ABI to `.venv`; a CPython 3.11 binary cannot run under 3.12.
4. Clear any old manual `PYTHONPATH` setting that could load another checkout.

```bash
uv run python -c "import sys, sysconfig; print(sys.executable); print(sysconfig.get_config_var('EXT_SUFFIX'))"
```

For Windows `DLL load failed` or missing Linux shared libraries, inspect binary runtime dependencies,
the driver, and CUDA versions. Python path changes alone may not resolve these errors.

On Linux, `libcudart.so.12` is provided by the declared `nvidia-cuda-runtime-cu12` dependency and resolved
automatically at import time. If it is missing, reinstall the project with its dependencies in the active environment;
installing PyTorch with CUDA 13 alone does not satisfy the native extension's CUDA 12 requirement.

## An Older Version Is Imported

```bash
uv run python -c "import style3d_sim as s3d; print(s3d.__file__)"
uv run python -c "import style3d_sim._core as core; print(core.__file__)"
```

If these point to system site-packages, activate the project's `.venv` and reinstall the package.
A newer wrapper paired with an older `_core` may cause missing attributes or enums; update both to matching versions.

## Submodule Initialization Fails

The public `style3d-sim-bin` submodule provides prebuilt extensions for external users.
The `synreal-solver` submodule is a closed-source native backend available only to authorized
internal developers. Cloning the public root repository does not grant access to its source.
External users should initialize `style3d-sim-bin` and use a matching binary release.

## CMake Cannot Find pybind11, Python, or CUDA

Package installation supplies pybind11 through the build backend and selects Python automatically; no interpreter path is needed.
For authorized internal source builds, check `nvcc --version`, the host compiler, and CMake.
Use a compatible developer terminal on Windows.
After toolchain changes, configure a new build directory to avoid stale cache entries.

## Login Fails or Waits for Input {#authentication}

1. Check authentication service access and account validity from the running process.
2. Set `STYLE3D_SIM_ACCOUNT` and `STYLE3D_SIM_PASSWORD` before launching the process.
3. Check the variable names and confirm the running process inherited both values.
4. Headless runs still request stdin when credentials are missing.
5. Check `is_logged_in()` in the simulation process; authentication in another Python process does not transfer.

Keep exception logs without sensitive data. Empty passwords or unlimited retries do not resolve authentication failures.

## “requires a Newton model allocated on a CUDA device”

```python
model = builder.finalize(device="cuda:0")
```

Confirm that `warp.is_cuda_available()` is `True`.
After CPU finalization, recreate model and state arrays on the correct device; changing configuration alone cannot move them.

## Cloth Reports Unsupported Generic Surface Elasticity

Register Style3D attributes after creating the builder, then use `style3d.add_cloth_grid()` or `add_cloth_mesh()`.
See [Cloth](guides/deformables.en.md) for the supported material path.

## State Particle Counts Do Not Match

Solver initialization may append rigid body proxy particles.
Create `model.state()`, viewers, and count-dependent buffers after the solver.
Use only physical particles for material selection and surface reconstruction.

## Penetration, Jitter, NaNs, or GPU Memory Exhaustion

| Symptom | First checks |
| --- | --- |
| Body passes through the ground | Shapes exist, contacts update each step, contacts are passed to step |
| Cloth penetration | Soft contacts, self-collision thickness, initial penetration, `dt` |
| Jelly/fluid collapse | Mass calibration, scale, material units, nonlinear budget |
| NaNs | Nonfinite inputs, degenerate mesh, nonpositive mass/radius, excessive deformation |
| Graph ignores new settings | Recapture after configuration changes |
| Slow surface reconstruction | Increase voxel size and lower rebuild frequency |
| GPU memory exhaustion | Reduce particles/environments, surface resolution, contact capacity; measure peak usage |

Reduce the scene, disable graphs and rendering, and follow [Configuration and Performance](guides/configuration.en.md).
Increasing stiffness with insufficient iterations may fail to reduce physical error.
