# Installation and Build

Run all commands on this page from the repository root. Prepare the Python environment,
then install a compatible prebuilt extension. The native `synreal-solver` backend is
closed source and its source build is restricted to authorized internal developers.

## 1. Prepare the Tools

| Tool | Purpose | Requirement |
| --- | --- | --- |
| Git | Fetch the repository and public binary submodule | Access to the public repository and `style3d-sim-bin` |
| Python | Run the Python API | CPython 3.11 or 3.12; `.python-version` currently selects 3.12 |
| uv | Manage `.venv` and dependencies | [Install uv](https://docs.astral.sh/uv/getting-started/installation/) |
| NVIDIA driver | Run CUDA | Compatible with the GPU and CUDA versions used by Warp and the extension |
| CUDA runtime | Run prebuilt extensions | CUDA 12; the full CUDA Toolkit is not required |
| CMake | Configure or install the extension | 3.20+; the build backend installs it automatically when needed |
| CUDA Toolkit | Internal source development only | Required only for authorized `synreal-solver` development |
| C++ compiler | Internal source development only | C++20; required only for authorized native development |

```bash
git --version
uv --version
cmake --version
nvidia-smi
```

For authorized internal source builds, also check `nvcc --version`. The CUDA version reported by `nvidia-smi`
is the maximum supported by the driver, rather than the installed Toolkit version.
Internal Windows builds use Visual Studio **Developer PowerShell**; internal Linux builds use a CUDA-supported GCC/Clang toolchain.

## 2. Clone the Repository

```bash
git clone https://github.com/Style3D/style3d-sim.git
cd style3d-sim
```

Skip this step if you already have a working copy.

!!! note "Public and private components"
    `style3d-sim-bin` is the public prebuilt-extension repository used by external users.
    `synreal-solver` is a closed-source native backend and is available only to authorized
    internal developers. Cloning the public root repository does not grant source access.
    Documentation CI does not initialize either submodule.

## 3. Create the Python Environment

```bash
uv sync --extra examples --no-install-project
```

The optional `examples` extra includes viewer and example dependencies.
Omit `--extra examples` for the base API only. See [Demos](../guides/demos.en.md) for training dependencies.
`--no-install-project` prepares dependencies first; the package and native extension are installed together in the next step.

Activate the environment and run the following installation commands in the same terminal:

=== "Windows PowerShell"

    ```powershell
    .\.venv\Scripts\Activate.ps1
    ```

=== "Linux Bash"

    ```bash
    source .venv/bin/activate
    ```

The `scikit-build-core` backend runs CMake automatically and selects the target environment's Python ABI.
No manual `Python_EXECUTABLE` or `PYTHONPATH` setting is required.

## 4A. Install a Prebuilt Extension (public workflow)

Choose this route if you do not need to modify native code. The binary must match your operating system,
CPU architecture, Python ABI, and runtime environment.

```bash
git submodule update --init style3d-sim-bin
uv pip install -e .
```

`-e` creates an editable install, so Python source changes take effect directly. Use `uv pip install .` for a regular install.
Both install the matching native extension and its license; no separate CMake installation is needed.
After initializing the public binary submodule, `uv sync --extra examples` also performs an editable installation.

Package builds default to prebuilt mode even when an internal source checkout is present. The binary layout is:

```text
style3d-sim-bin/<system>/<config>/_core<EXT_SUFFIX>
```

For example, Windows Release with CPython 3.12 commonly uses `_core.cp312-win_amd64.pyd`.
If no matching binary exists, select a Python version provided by the public submodule or
contact the internal release owner. External users cannot build `synreal-solver` from source.

## 4B. Build from Source (authorized internal developers only)

This route requires authorized access to the closed-source `synreal-solver` submodule,
the CUDA Toolkit, and a C++20 compiler. It is not available to external contributors.

```bash
git submodule update --init --recursive synreal-solver
uv pip install -e . -C cmake.define.SYNREAL_SOLVER_USE_PREBUILT=OFF
```

The backend provides pybind11 in an isolated environment and handles CMake configuration, compilation, and installation.
The native submodule keeps its standalone install layout; the root repository handles the Python package layout.
In source mode, sync dependencies with `uv sync --extra examples --no-install-project --inexact` and run with
`uv run --no-sync` to retain the current installation. Repeat the installation command with `OFF` when rebuilding.

## 5. Verify the Loaded Build

```bash
python -c "import style3d_sim as s3d; print(s3d.__file__)"
python -c "import style3d_sim._core as core; print(core.__file__)"
python -c "import warp as wp; wp.init(); print(wp.is_cuda_available())"
```

With an editable install, Python files point into this checkout's `src/style3d_sim/`, while the extension lives in
the environment's `site-packages/style3d_sim/`. With a regular install, both live in the environment.
The CUDA check should print `True`.
Importing the library does not authenticate the session. Continue with [Authentication](authentication.en.md).

## 6. Update after Source Changes

- Python changes take effect directly with an editable installation.
- For public users, native updates arrive through a new `style3d-sim-bin` release.
- Authorized internal developers repeat the installation command with `OFF` after C++/CUDA changes.
- In a new terminal, activate `.venv` and set the required login variables again.

After changing Python versions, reinstall a matching extension. A `.pyd` or `.so` built for another ABI cannot be reused.

## 7. Build a Wheel

```bash
uv build --wheel
```

The wheel in `dist/` contains Python files, type stubs, the native extension for the current platform and Python ABI,
and the binary license. Internal source builds use `uv build --wheel -C cmake.define.SYNREAL_SOLVER_USE_PREBUILT=OFF`.
`uv build` also creates a public source distribution containing the prebuilt binaries and excluding private native source.
