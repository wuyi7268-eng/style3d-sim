# 安装与构建

公开用户请安装匹配的预编译扩展。`synreal-solver` 原生后端是闭源的，源码构建仅限获授权的内部开发人员。

## 1. 准备工具

| 工具 | 用途 | 要求 |
| --- | --- | --- |
| Git | 下载公开仓库与二进制子模块 | 能访问公开仓库和 `style3d-sim-bin` |
| Python | 执行 Python API | 支持 CPython 3.11、3.12；`.python-version` 当前选择 3.12 |
| uv | 管理 `.venv` 和依赖 | [安装说明](https://docs.astral.sh/uv/getting-started/installation/) |
| NVIDIA 驱动 | 运行 CUDA | 与 GPU、Warp 和扩展使用的 CUDA 版本兼容 |
| CUDA 运行时 | 运行预编译扩展 | CUDA 12；无需安装完整 CUDA Toolkit |
| CMake | 配置或安装原生扩展 | 3.20+；构建后端会在需要时自动安装 |
| CUDA Toolkit | 仅供内部源码开发 | 只有获授权的 `synreal-solver` 开发需要 |
| C++ 编译器 | 仅供内部源码开发 | 只有获授权的原生开发需要 |

```bash
git --version
uv --version
cmake --version
nvidia-smi
```

获授权的内部源码构建还需要检查 `nvcc --version`。`nvidia-smi` 显示的 CUDA 版本是驱动支持上限，不是本机 Toolkit 版本。
内部 Windows 构建建议从 Visual Studio 的 **Developer PowerShell** 启动；内部 Linux 构建使用 CUDA 支持的 GCC/Clang。

## 2. 下载仓库

```bash
git clone https://github.com/Style3D/style3d-sim.git
cd style3d-sim
```

已经有工作副本时无需重复 clone。

!!! note "公开组件与闭源组件"
    `style3d-sim-bin` 是外部用户使用的公开预编译扩展仓库。
    `synreal-solver` 是闭源原生后端，仅向获授权的内部开发人员开放。
    克隆公开根仓库不会自动获得源码权限。文档 CI 不初始化任何子模块。

## 3. 创建 Python 环境

```bash
uv sync --extra examples --no-install-project
```

`examples` 是可选依赖 extra，包含 viewer 和示例需要的依赖。
只使用基础 API 时可以省略 `--extra examples`。训练相关依赖见 [示例](../guides/demos.md)。
`--no-install-project` 先准备依赖，包和原生扩展将在下一步一起安装。

激活环境，并在同一个终端执行后续安装命令：

=== "Windows PowerShell"

    ```powershell
    .\.venv\Scripts\Activate.ps1
    ```

=== "Linux Bash"

    ```bash
    source .venv/bin/activate
    ```

`scikit-build-core` 构建后端会自动调用 CMake，并使用目标环境的 Python ABI。
无需手动指定 `Python_EXECUTABLE` 或设置 `PYTHONPATH`。

## 4A. 使用预编译扩展（公开用户流程）

不需要修改原生代码时使用此路径。二进制必须匹配操作系统、CPU 架构、Python ABI 和运行环境。

```bash
git submodule update --init style3d-sim-bin
uv pip install -e .
```

`-e` 使用 editable 安装，修改 Python 源码后直接生效。普通安装使用 `uv pip install .`。
两种安装都会安装匹配的原生扩展及其许可证；无需单独执行 CMake 安装。
公开二进制子模块初始化后，也可以直接使用 `uv sync --extra examples` 完成 editable 安装。

包构建默认使用预编译扩展，即使本地存在内部源码子模块也不会自动编译它。二进制在子模块中的路径为：

```text
style3d-sim-bin/<system>/<config>/_core<EXT_SUFFIX>
```

例如 Windows、Release、CPython 3.12 的扩展通常为 `_core.cp312-win_amd64.pyd`。
如果不存在匹配的二进制，请换用公开子模块提供的 Python 版本或联系内部发布负责人。外部用户不能从 `synreal-solver` 源码构建。

## 4B. 从源代码编译（仅限获授权的内部开发人员）

这条路径需要闭源 `synreal-solver` 子模块权限、CUDA Toolkit 和 C++20 编译器，外部贡献者无法使用。

```bash
git submodule update --init --recursive synreal-solver
uv pip install -e . -C cmake.define.SYNREAL_SOLVER_USE_PREBUILT=OFF
```

构建后端在隔离环境中提供 pybind11，并完成 CMake 配置、编译和安装。
原生子模块保留自己的独立安装布局，Python 包布局由主仓库负责。
源码模式下同步依赖时使用 `uv sync --extra examples --no-install-project --inexact`，运行时可以用
`uv run --no-sync` 保留当前安装；需要重建时重新执行带 `OFF` 参数的安装命令。

## 5. 验证加载的是本次构建

```bash
python -c "import style3d_sim as s3d; print(s3d.__file__)"
python -c "import style3d_sim._core as core; print(core.__file__)"
python -c "import warp as wp; wp.init(); print(wp.is_cuda_available())"
```

editable 安装时，Python 文件指向当前仓库的 `src/style3d_sim/`，扩展位于虚拟环境的 `site-packages/style3d_sim/`。
普通安装时两者都位于虚拟环境中。CUDA 检查应打印 `True`。
仅能导入库不代表完成账号认证，下一步是 [登录](authentication.md)。

## 6. 修改源码后的更新

- Python 文件：使用 editable 安装时直接生效，无需重新编译扩展。
- 公开用户通过新的 `style3d-sim-bin` 版本获取原生更新。
- 获授权的内部开发人员修改 C++/CUDA 后，重新执行带 `OFF` 参数的安装命令。
- 新建终端：激活 `.venv` 并设置需要的登录环境变量。

如果换了 Python 版本，要使用匹配版本重新安装扩展，不能直接复制另一个 ABI 的 `.pyd` 或 `.so`。

## 7. 构建 wheel

```bash
uv build --wheel
```

产物位于 `dist/`，包含 Python 文件、类型声明、当前平台和 Python ABI 的原生扩展及二进制许可证。
内部源码构建使用 `uv build --wheel -C cmake.define.SYNREAL_SOLVER_USE_PREBUILT=OFF`。
`uv build` 还会生成公开源码分发包，其中包含预编译二进制，不包含闭源原生源码。
