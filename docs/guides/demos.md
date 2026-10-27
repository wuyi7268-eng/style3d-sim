# 示例与渲染

示例用于确认调用方式与场景配置，不是所有脚本都默认选择 Style3D Sim。
先运行 `--help` 查看 solver、viewer 和设备选项。所有命令从仓库根目录执行。

## 运行前检查

1. 完成 [安装](../getting-started/installation.md)，将包安装到 `.venv`。
2. `uv sync --extra examples` 安装 viewer、资产与示例依赖。
3. 完成 [账号设置](../getting-started/authentication.md)，避免 headless 程序等待终端输入。
4. 如果需要资产下载，允许访问对应外部地址；库的离线 API 不会自动替所有示例下载资产。

## 常用示例

| 场景 | 脚本 | 说明 |
| --- | --- | --- |
| 基本刚体 | `examples/example_basic_shapes.py` | 使用 `--solver synreal`，部分资产会下载 |
| 流体 | `examples/example_fluid.py` | resolution 控制物理粒子规模，可选表面重建 |
| 雪/黏结颗粒 | `examples/example_snow.py` | 演示黏结、接触与塑性 |
| 颗粒切割 | `examples/example_granular_beaf.py` | 文件名沿用仓库的 `beaf` 拼写 |
| 布料袋 | `examples/example_plastic_bag.py` | 读取本地 USD 及版片 UV |
| 多自由度关节 | `examples/example_rrabd_joints.py` | 对比驱动目标与实际运动 |

```bash
uv run python examples/example_basic_shapes.py --solver synreal --device cuda:0
uv run python examples/example_fluid.py --resolution 16 --viewer null --num-frames 120 --test
uv run python examples/example_rrabd_joints.py --viewer null --num-frames 300 --test
```

有环境变量凭据时无需加 `--login`；RRABD 的 `--login` 会强制进入该示例自己的交互登录流程。
`--viewer null` 只关闭图形窗口，仍需要 GPU，并且不会自动禁用登录。

## G1 Jelly Drop

![G1 Jelly Drop](../images/g1_jelly_drop_demo.png){ width="640" }

G1 落到独立的 FEM 果冻块上，用于展示机器人与四面体软体之间的接触。

```bash
uv run python demos/g1_jelly_drop/demo_g1_jelly_drop.py
uv run python demos/g1_jelly_drop/demo_g1_jelly_drop.py --viewer null --num-frames 120 --test
```

短帧数用于确认能启动，不等同于完整轨迹验证。
`--jelly-mu`、`--jelly-lambda`、`--jelly-damping`、`--substeps` 等选项应先通过 `--help` 查看当前默认值。
改变渲染材质不会改变物理参数；修改弹性与阻尼后要重新导出仿真。

对应详细流程在仓库的 `demos/g1_jelly_drop/README.md`，包含 USD、Blender、验证和输出路径。

## Piper Bag Transport

![Piper Bag Transport](../images/piper_scene_demo.png){ width="640" }

机械臂夹住软袋，把袋内苹果提起、摆动，再放回台面。
袋子接触、布料形变、机器人运动和苹果运动由实际求解状态导出，离线渲染使用这些缓存。

```bash
uv run python demos/piper_bag_transport/demo_piper_bag_transport.py
uv run python demos/piper_bag_transport/workflow.py --help
```

Windows 上运行完整离线流程的例子：

```powershell
$blenderExe = 'C:/Program Files/Blender Foundation/Blender 5.2/blender.exe'
uv run python demos/piper_bag_transport/workflow.py --blender "$blenderExe" --draft --output-dir build/piper-preview
```

将 Blender 路径替换为实际安装位置。
workflow 按顺序导出、验证、生成遥测、构建场景、渲染与编码；缺少资产或外部工具时会停止。
细节见仓库的 `demos/piper_bag_transport/README.md`。

## 实时 viewer 与离线渲染

| 阶段 | 需要什么 | 产物 |
| --- | --- | --- |
| 仿真 | CUDA、模型、扩展、凭据 | 解出的状态 |
| USD 导出 | 仿真及 USD Python 支持 | 带采样时间的几何/姿态缓存 |
| Blender 构建 | 支持 USD import 的 Blender | 可编辑 `.blend` 工程 |
| 离线渲染/编码 | Cycles 与相应编码工具 | 帧图和视频 |

不能只改 `.blend` 外观就假定物理轨迹已经更新。
物理源文件与参数变化后重新导出，验证缓存和当前源码的一致性，再进行渲染。
生成缓存、视频和日志保存在各 demo 的输出目录或显式传入的 output directory，不放进公开文档资产目录。

## Locomotion 可选工作流

这部分依赖更多包，与文档 CI 完全独立：

```bash
uv sync --extra locomotion
uv run --extra locomotion python -m examples.locomotion.validate --robot all
uv run --extra locomotion python -m examples.locomotion.train --robot go2 --num-envs 2048 --run
```

训练默认可能仅校验配置，实际运行需要 `--run`。
先降低并行环境数量检查显存，再按训练目录的说明调节规模。
