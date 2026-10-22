> 当前性能设置：机器人自碰撞关闭，所有环境脚底摩擦固定为 1.0；地面/球体碰撞保留。自碰撞奖励不再有有效信号，摩擦随机化不再对齐官方。此变更需重启并新开训练，旧代码指纹检查点不可直接加载。

# MicroDuck 训练与回放

本目录支持 SynReal / MuJoCo 四任务训练。当前使用 `official_adapted_v1` 官方移植配方；
它仍有未移植项，详见 [移植状态](OFFICIAL_PORT_STATUS.md)。训练仅支持此配方，未指定时也默认使用它。旧简化训练分支已删除。

## 日常入口

在项目根目录执行：

```bash
source .venv/bin/activate
python -u -m examples.microduck.train \
  --training-config examples/microduck/configs/rtx4090_synreal_official_adapted.json \
  --task velstand --num-envs 128 --iterations 10000 \
  --run-name synreal_velstand_official_v2
```

`--task` 可选 `velocity`、`velstand`、`groundpick`、`ballkick`。建议先用 `--num-envs 4 --iterations 2` 短测。
128 不是已测得的最优环境数。训练默认无窗口，追加 `--viewer gl` 可显示窗口。
使用相同配方在 MuJoCo 训练时，追加 `--solver mujoco`。

```bash
# 回放：将路径换成实际检查点
python -u -m examples.microduck.play --solver synreal \
  --checkpoint /path/to/run/model_500.pt --viewer gl --num-frames 3000

# 续训：从检查点目录恢复配置，输出到新的运行目录
python -u -m examples.microduck.train --solver synreal \
  --resume /path/to/run/model_500.pt --num-envs 128 --iterations 1000 --run-name continued

# SynReal 模型在 MuJoCo 中回放
python -u -m examples.microduck.play --solver mujoco --allow-cross-backend \
  --checkpoint /path/to/run/model_500.pt --viewer gl --num-frames 3000

# 曲线
tensorboard --logdir logs/rsl_rl
```

结果目录：`logs/rsl_rl/microduck_<solver>/<时间>_<run-name>/`，包含配置、检查点和 TensorBoard 日志。
新训练时 CLI 覆盖 JSON 默认值；续训使用检查点旁的配置，不重新套用训练模板。
旧简化配方检查点不再支持。此次删除旧分支改变了配方代码指纹，整理前的检查点需使用对应旧代码；不能绕过校验。

回放 `--viewer null` 关闭窗口，`--benchmark` 取消实时等待。官方移植配方目前不启用 CUDA graph。
Velocity/VelStand 窗口中 `I/K` 调整前后速度、`J/L` 调整横移、`U/O` 调整转向，初始指令为零。
可用 `--command 0.1 0 0` 设置初始速度。跨后端动力学不同，动作效果不保证一致。

## 配置只需区分三层

| 文件 | 用途 |
| --- | --- |
| `configs/rtx4090_synreal_official_adapted.json` | 当前新训练入口配置：环境数、迭代数、网络、PPO、保存间隔 |
| `configs/official_snapshot.json` | 官方导出快照：任务奖励、权重课程、观测、事件与终止条件 |
| 训练目录的 `config.json` | 本次运行的实际配置及续训/回放元数据 |

`config.py` 提供官方移植配方的 PPO 默认值。旧版两个训练模板已删除。运行名称 `--run-name` 只是实验标签，任务由 `--task` 选择。

## 代码导航

| 范围 | 文件与职责 |
| --- | --- |
| 日常入口 | `train.py` 训练/续训；`play.py` 回放；`common.py` 参数、登录与检查点校验 |
| 共享仿真 | `env.py` 步进；`policy_spec.py` 资产/关节/PD；`tasks.py` 四任务名称与球体场景 |
| 官方奖励 | `official_recipe.py` 适配与汇总；`official_reward_functions.py` 官方任务公式；`official_velocity_rewards.py` 步态公式 |
| 官方观测与事件 | `official_observations.py`；`official_events.py`；`official_reset_functions.py`；`official_randomization.py` |
| 官方传感器适配 | `official_sensors.py`；`official_adapter.py` |
| 官方模型入口 | `official_play.py` 原 walk 演示；`pretrained.py` ONNX；`mjlab_train.py` 官方训练启动器 |
| 开发工具 | `export_official_recipe.py` 导出快照/函数；`validate.py` 站立检查；`tests/` 回归测试 |
| 官方原始目录 | `microduck_rl-develop/`、`microduck-main/`，保留原路径、内容及独立环境 |

查看奖励权重：`official_snapshot.json → tasks → 任务名 → rewards`；变化课程在 `curricula`。
[官方奖励索引](OFFICIAL_REWARDS_INDEX.md) 用于定位原始定义，
[移植状态](OFFICIAL_PORT_STATUS.md) 记录当前差异和验证结果，
[历史验证](VALIDATION.md) 仅记录旧版本测试，不代表当前配方表现。

## 登录与依赖

使用根目录 `.venv`。新机器先按根 README 安装原生扩展及依赖：

```bash
uv sync --extra locomotion --inexact --no-install-project
source .venv/bin/activate
```

SynReal 从 `STYLE3D_ACCOUNT`、`STYLE3D_PASSWORD` 读取登录信息，缺少时隐藏输入。
需要替换已有在线会话时使用 `--force-login`。凭据不写入训练配置。
若当前 Python 3.12 环境找不到 pip 安装的 CUDA 运行库，可设置：

```bash
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"
```

## 官方权重与验证

官方 ONNX 在 `logs/pretrained/microduck/`，与本地 PD 检查点契约不同：

```bash
python -m examples.microduck.pretrained --solver mujoco --policy velstand --keyboard
python -m examples.microduck.play --solver synreal --policy walk --num-frames 3000
python -m examples.microduck.mjlab_train --num-envs 64 --iterations 5
```

`pretrained` 的 MuJoCo 回放和 `mjlab_train` 使用官方项目独立环境。
旧的 ONNX→共享 PD 迁移实验已删除；官方 ONNX MuJoCo 回放和原 walk 演示仍保留。

```bash
python -m pytest -q examples/microduck/tests
python -m examples.microduck.validate --solver synreal --num-frames 150
```

单元测试验证奖励来源、观测、事件和检查点校验；真实 GPU 短测脚本保存在 `.validation/microduck-v2/`。
`.validation/` 保存本地开发验证脚本、日志和短测检查点，不是训练入口；其中缓存可以重新生成。

## 检查点回放提速

`play --checkpoint ...` 默认使用 `playback.py` 的 actor 专用路径：省去奖励、训练日志、critic 观测和
奖励专用的求解后碰撞传感器更新。保留策略观测噪声/延迟、PD 控制、物理步长、求解前碰撞、
终止课程、自动重置、命令及推扰。训练实现和配方指纹不变。

同一检查点可加 `--full-step` 使用原完整训练步对照；`--viewer null --benchmark --num-frames 300`
用于无窗口测速，图形窗口实际帧率还受渲染和同时运行的训练影响。`--disable-cuda-graph` 仍只影响原 walk 演示。
