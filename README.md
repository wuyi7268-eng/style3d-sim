<div align="center">
  <h1>Style3D Sim 2.0</h1>
  <p>CUDA-accelerated implicit simulation for particles, cloth, deformable bodies, and affine rigid bodies.</p>
  <p>
    <a href="docs/index.en.md">English Docs</a> &middot;
    <a href="docs/index.md">中文文档</a> &middot;
    <a href="#demo-showcase">Demos</a> &middot;
    <a href="#build-and-install">Build</a> &middot;
    <a href="#authentication">Login</a> &middot;
    <a href="#run-examples">Run examples</a> &middot;
    <a href="#python-api">Use the API</a>
  </p>
</div>

Style3D Sim provides a Python API and a Newton Physics adapter, so a Newton
model can use the native backend for contact handling, constraints, and nonlinear solves.
The `synreal-solver` native backend is closed source and available only to authorized
internal developers. This public repository contains the Python adapter, examples,
documentation, and integration code; it does not contain the native backend source.
The Python library and type stubs are open source under the [Apache License 2.0](LICENSE).
See [License](#license) for the separate terms covering native binaries and third-party components.

Read the [English usage documentation](docs/index.en.md) or [中文使用文档](docs/index.md)
for installation, authentication, simulation tutorials, and API references.

## Demo Showcase

<table>
  <tr>
    <td width="50%" align="center" valign="top">
      <img src="docs/images/g1_jelly_drop_demo.gif" alt="G1 humanoid interacting with deformable jelly blocks" width="100%" />
      <p><strong>G1 Jelly Drop</strong><br />Humanoid interaction with deformable FEM blocks.</p>
      <p><a href="demos/g1_jelly_drop/demo_g1_jelly_drop.py">View demo code</a></p>
    </td>
    <td width="50%" align="center" valign="top">
      <img src="docs/images/piper_scene_demo.gif" alt="Piper robot arm transporting a deformable bag of apples" width="100%" />
      <p><strong>Piper Bag Transport</strong><br />Robotic manipulation of a deformable bag.</p>
      <p><a href="demos/piper_bag_transport/demo_piper_bag_transport.py">View demo code</a></p>
    </td>
  </tr>
</table>

## Features

- CUDA-native implicit integration and block-PCG linear solves.
- Particle, tetrahedral, cloth, and affine rigid-body support.
- Contact, friction, joint, spring, and drive constraints.
- Newton Physics solver integration through `NewtonSolver`.
- Optional prebuilt extensions from the public `style3d-sim-bin` repository.
- Python and CMake build workflows.

## Requirements

For normal development and execution:

- Windows or Linux.
- CPython 3.11 or 3.12.
- NVIDIA GPU, driver, and CUDA 12 runtime compatible with the installed Warp/Newton packages.
- [`uv`](https://docs.astral.sh/uv/).

For the public prebuilt workflow, install:

- Git, to fetch the public binary submodule.

Prebuilt extensions need the CUDA 12 runtime; the full CUDA Toolkit is required only for native source builds.

For native source development by authorized internal developers, also install:

- CMake 3.20 or newer (automatically provided by the Python build backend when needed).
- A C++20 compiler.
- CUDA Toolkit and a compatible host compiler.
- Visual Studio/MSVC on Windows.

The examples require the optional `examples` dependency group. The locomotion
examples require the larger `locomotion` dependency group.

## Clone and initialize the public components

Clone the public repository without recursively fetching the private native source:

```bash
git clone https://github.com/Style3D/style3d-sim.git
cd style3d-sim
```

The public prebuilt extension is maintained in `style3d-sim-bin`:

```bash
git submodule update --init style3d-sim-bin
```

The `synreal-solver` submodule is closed source and requires internal access. External
users should not try to initialize it; use a matching prebuilt extension instead.

## Build and install

The `scikit-build-core` backend runs CMake and installs the Python package together with the native extension.
Package builds use the public prebuilt extension by default, even when a private source checkout is present.
See the [installation guide](docs/getting-started/installation.md) for complete Windows and Linux instructions.

### Prepare and activate the environment

Install dependencies first, then activate the environment before installing the package.

Bash or Zsh:

```bash
uv sync --extra examples --no-install-project
source .venv/bin/activate
```

PowerShell:

```powershell
uv sync --extra examples --no-install-project
.\.venv\Scripts\Activate.ps1
```

Run the following installation commands in the same terminal. The backend selects the environment's Python automatically.

### Use a prebuilt native extension

Use this mode when a matching binary is available in `style3d-sim-bin`:

```bash
git submodule update --init style3d-sim-bin
uv pip install .
```

For development, use an editable install so Python source changes take effect directly:

```bash
uv pip install -e .
```

These commands work in PowerShell and Bash. No manual `PYTHONPATH` or `Python_EXECUTABLE` setting is required.
Once the binary submodule is initialized, `uv sync --extra examples` also installs the project in editable mode.

### Build the native solver from source (internal developers only)

This mode compiles the closed-source C++/CUDA `synreal-solver` submodule locally and
requires authorized internal access. It is unavailable to external contributors:

```bash
git submodule update --init --recursive synreal-solver
uv pip install -e . -C cmake.define.SYNREAL_SOLVER_USE_PREBUILT=OFF
```

Use source mode only when an authorized internal checkout includes `synreal-solver/`.
Re-run the installation command after C++/CUDA changes.
In source mode, use `uv sync --extra examples --no-install-project --inexact` when syncing dependencies,
and `uv run --no-sync` when running commands to retain the native install.

### Build a wheel

Build a wheel for the active Python version and platform:

```bash
uv build --wheel
```

The wheel includes the Python source, type stubs, matching native extension, and binary license.
An internal source build can use `uv build --wheel -C cmake.define.SYNREAL_SOLVER_USE_PREBUILT=OFF`.
Public source distributions include the prebuilt binaries and exclude the private native source.

### Verify the installation

```bash
python -c "import style3d_sim as s3d; print(s3d.__file__)"
python -c "import style3d_sim._core as core; print(core.__file__)"
```

The extension filename must match the active Python version, operating system,
and architecture. For example, a 64-bit CPython 3.12 Windows installation uses
a file similar to `_core.cp312-win_amd64.pyd`.

## Authentication

Style3D Sim authentication is required before using authenticated native solver
features. `NewtonSolver` checks the current session when it is created. If the
session is not authenticated, it reads credentials from the environment or
prompts in the terminal.

### Environment variables

Set the credentials in the terminal that will run the example.

PowerShell:

```powershell
$env:STYLE3D_SIM_ACCOUNT = "your-account"
$env:STYLE3D_SIM_PASSWORD = "your-password"
```

Bash or Zsh:

```bash
export STYLE3D_SIM_ACCOUNT="your-account"
export STYLE3D_SIM_PASSWORD="your-password"
```

The variables are process-local; set them again in a new terminal session.

### Interactive login

If either credential is missing, `NewtonSolver` asks for the account and reads the password without echoing it:

```text
Style3D Sim account: your-account
Password:
```

For applications that need to authenticate explicitly, use the public API:

```python
import getpass
import style3d_sim as s3d

account = input("Style3D Sim account: ").strip()
password = getpass.getpass("Password: ")
if not s3d.login(account, password, False):
    raise RuntimeError("Style3D Sim login failed")
```

Avoid committing credentials to source files or shell scripts. Prefer a temporary
terminal environment or an external secret manager.

## Run examples

All commands below run inside the managed environment created by `uv`.

### Example assets

`s3d.get_asset("joints/ball/ball.urdf")` returns a path in the Style3D Sim asset repository.
The first call clones `https://github.com/Style3D/style3d-sim-assets.git` into
`~/.cache/style3d_sim/assets`; later calls reuse it. If a requested file is missing,
`get_asset()` attempts one fast-forward update before reporting a missing asset.
Git and network access are required. The repository root should contain the
asset groups directly, such as `joints/` and `piper/`.

For a local checkout, set `STYLE3D_SIM_ASSETS_DIR` to its root. For example, in PowerShell:

```powershell
$env:STYLE3D_SIM_ASSETS_DIR = "D:\path\to\style3d-sim-assets"
```

`s3d.get_assets()` returns the asset root. Neither function is called when the
Python package is imported.

### Basic Newton scene

The basic-shapes example demonstrates the Newton adapter with SynReal:

```bash
uv run python examples/example_basic_shapes.py --solver synreal
```

### Particle and material examples

```bash
uv run python examples/example_fluid.py
uv run python examples/example_snow.py
uv run python examples/example_granular_beaf.py
```

The fluid example supports a headless regression run:

```bash
uv run python examples/example_fluid.py --viewer null --num-frames 180 --test
```

### Joint and rigid-body example

```bash
uv run python examples/example_rrabd_joints.py --login
```

Use `--viewer null --test` for a non-interactive run:

```bash
uv run python examples/example_rrabd_joints.py --login --viewer null --test --num-frames 300
```

### Locomotion examples

Install the optional dependencies first:

```bash
uv sync --extra locomotion
uv run --extra locomotion python -m examples.locomotion.validate --robot all
```

Training commands validate by default. Add `--run` to start training:

```bash
uv run --extra locomotion python -m examples.locomotion.train \
  --robot go2 --num-envs 2048 --run
```

### MicroDuck MuJoCo / SynReal training

Select either physics backend for the shared MicroDuck flat-ground PPO task:

```bash
uv sync --extra locomotion --inexact --no-install-project
source .venv/bin/activate
python -m examples.microduck.train --solver mujoco --num-envs 64 --iterations 5
python -m examples.microduck.train --solver synreal --num-envs 64 --iterations 5
```

Use `python -m examples.microduck.validate --solver synreal --viewer gl` for a visual pose-hold simulation.
For the official walking policy, run `python -m examples.microduck.play --solver synreal --num-frames 3000`
(or `--solver mujoco`). This reuses the original robot demo's model, controller, and CUDA graph.
Pass `--checkpoint <model.pt>` instead to replay a policy trained by this shared PPO task.
SynReal prompts for credentials. Both backends use the same position PD task;
this baseline differs from the copied project's BAM sim2real recipe, which remains
available through `python -m examples.microduck.mjlab_train` in its separate environment.
See [MicroDuck instructions](examples/microduck/README.md) for configuration,
validation, logs, and original mjlab training/export commands.
For a concise Chinese quick start covering installation, training, graphical
playback, keyboard control, and pretrained weights, see
[MICRODUCK_GUIDE.md](MICRODUCK_GUIDE.md).

## Python API

The Newton adapter uses a Newton model and state while the native SynReal solver owns the numerical solve:

```python
import newton
import style3d_sim as s3d

builder = s3d.ModelBuilder(up_axis=newton.Axis.Z)
builder.add_ground_plane()
builder.add_particle(
    pos=(0.0, 0.0, 1.0),
    vel=(0.0, 0.0, 0.0),
    mass=1.0,
    radius=0.05,
)

model = builder.finalize(device="cuda:0")
solver = s3d.NewtonSolver(model)
```

For lower-level native use, the package also exports `Model`, `State`, `Config`,
`Solver`, and `login`. See the type stubs in `src/style3d_sim/_core.pyi` for
the complete public native API.

## Tests and checks

Run the full Python test suite with:

```bash
uv run pytest -q
```

Many tests require a CUDA-capable device. Run the lightweight collection check on any machine:

```bash
uv run pytest --collect-only -q
```

Run the formatter and linter check with:

```bash
uv run ruff check src tests examples
```

## License

The Python source files (`.py`) and public type stubs (`.pyi`) under `src/style3d_sim/`
are licensed under the [Apache License 2.0](LICENSE). Preserve the applicable copyright,
license, and attribution notices when redistributing these files or modified versions.

The native `synreal-solver` implementation is closed source, and its source is available
only to authorized internal developers. The Apache-2.0 license for the Python API and
type stubs does not grant access to or license the native implementation.

Prebuilt native extensions (`.pyd` and `.so`), including copies installed alongside the
Python package, are governed by the
[Style3D Sim Binary License Agreement](style3d-sim-bin/LICENSE).
Their use requires an applicable account entitlement or written agreement;
redistribution requires prior written permission from the licensor.

Third-party code, dependencies, and assets retain their respective licenses and notices.
