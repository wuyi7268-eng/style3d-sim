> 历史验证记录：以下结果针对当时的旧配方。当前官方移植版本请看 [移植状态](OFFICIAL_PORT_STATUS.md)。

# MicroDuck 重构验证（2026-10-20）

设备：RTX 4090，根目录 `.venv`，CUDA 0。

| 后端 | 并行环境 | PPO 更新 | 采样步 | 检查点回放 |
|---|---:|---:|---:|---:|
| mujoco | 2 | 2 | 96 | 50 帧，无窗口 |
| synreal | 2 | 2 | 96 | 50 帧，无窗口 |

两种后端的网络参数均为有限值，且第 0 次与第 1 次更新后的参数不同；检查点包含优化器和观测归一化状态。
MuJoCo 额外执行了 1 次断点续训更新，迭代编号从 2 继续。

实际运行目录：

- `logs/rsl_rl/microduck_mujoco/2026-10-20_15-24-49_597377_restructure_smoke`
- `logs/rsl_rl/microduck_synreal/2026-10-20_15-26-12_895200_restructure_smoke`

物理 GPU 回归：MuJoCo、SynReal、SynReal CUDA graph 三种模式均通过 150 控制帧站立保持、
单环境关节响应、局部重置和有限状态检查。最低躯干高度分别为 0.11536、0.11643、0.11645 m，
最低直立余弦分别为 0.99901、0.99943、0.99934。

资产映射、奖励行为和检查点后端校验测试通过；Ruff 检查和格式检查通过。
Newton 当前版本产生旧目标数组布局和标量重力的弃用提示，未影响本次运行。

这些结果验证训练、保存、加载、推理和重置流程；仅做了短程测试，未验证步态收敛。
本次没有修改 `src/style3d_sim/` 或原生 SynReal 求解器代码。

## 回放性能优化

`.pt` 回放默认捕获 CUDA graph，跳过奖励和训练日志计算，复用上一步观测。
训练的物理步长、控制频率、执行器参数与求解器迭代数均保持不变。

同一机器、同一短程 `.pt` 检查点，无窗口、无实时等待；预热 20 帧后计时 60 帧：

| 后端 | 优化前 FPS | 优化后 FPS | 优化后仿真时间 / 实际时间 |
|---|---:|---:|---:|
| MuJoCo | 36.2 | 293.7 | 5.87× |
| SynReal | 9.2 | 12.1 | 0.24× |

原始数据为 `logs/microduck_play_baseline.json` 和 `logs/microduck_play_after.json`。
这些是无窗口吞吐数据；正常 `.pt` 图形回放按 50 Hz 节奏运行。

官方 `walk.onnx` 回放直接复用原演示的 `Example`；它与 `.pt` 使用不同的机器人/控制配置，
每帧为 4 个物理步，而共享训练任务为 10 个，不能将两种任务的 FPS 当成同负载对比。

SynReal 两份独立的相同 full-step 环境也出现接触求解数值差异（12 帧内观测最大绝对差约 0.17，
关节坐标最大差约 0.0036）；因此不声称独立轨迹逐位一致。
回放测试检查返回观测与自身物理状态一致、命令正确以及重复局部超时重置；
MuJoCo 另检查完整 step 与快速回放的逐帧状态一致性。

同一进程、同一图形窗口、同一官方 `walk.onnx` 配置，预热 20 帧后计时 80 帧：

| 后端 | 原演示入口 FPS | 新 `play --policy walk` FPS |
|---|---:|---:|
| MuJoCo | 168.3 | 170.6 |
| SynReal | 25.0 | 24.7 |

官方权重的两种后端回放性能均已达到原演示入口水平。短程测量会随机器负载波动。

最终回归：9 项测试全部通过，包括 MuJoCo / SynReal 各自的 CUDA graph 开关模式、
动作响应、站立保持、局部重置，以及快速回放观测和超时重置检查。
Ruff 检查、128 列格式检查及 `git diff --check` 通过。

## SynReal 自碰撞

已确认官方 URDF 演示与共享 MJCF 训练环境之前均显式禁用自碰撞；现仅在 SynReal 路径中开启，
保留导入器默认的关节相邻部件过滤。MuJoCo 路径保持原设置。

开启后，两条 SynReal 路径均完成 150 控制帧零指令/零动作检查：官方策略通过有限状态和躯干高度检查，
共享训练环境没有跌倒或超时重置。碰撞输出中检测到不同机器人刚体之间的接触候选，
受控足部重叠检查也生成了自接触候选。资产回归验证左右足碰撞对在开启时未被过滤，关闭时被过滤。
普通测试为 4 passed、6 skipped；本次另外独立执行了上述 GPU 检查。

这些是短程检查，不代表开启自碰撞后已重新验证长期行走表现；上面的性能数字为开启前的历史测量。
没有修改 `src/style3d_sim/` 或原生求解器。

## Shared task adaptations: 2026-10-20

Added `velstand`, `groundpick`, and `ballkick` to the shared position-PD environment.
The current shared environment disables self-collisions (superseding the earlier shared-environment setting above).
New tasks use convex collision proxies and task-specific ground-contact models.

Validation on RTX 4090 with SynReal:

- Nine tests passed: asset mapping/filtering, reward behavior, checkpoint backend/task guards,
  GroundPick target cycle, and each new task's finite observations, partial reset, and timeout behavior.
- Each task completed a two-environment physics/PPO/checkpoint reload smoke run.
- Each actual `train.main` CLI path completed one PPO iteration with four environments,
  24 rollout steps, and actor/critic `[512, 256, 128]` from `rtx4090_synreal_probe.json`.
- Each resulting checkpoint completed eight headless `play.main` frames with CUDA graph enabled;
  the scene/task was recovered from checkpoint metadata without an explicit `--task`.
- Summary: `logs/microduck_task_cli_verification.json`; checkpoints: `logs/microduck_task_cli_verify/`.

These checks validate execution and serialization, not learned skill quality or convergence.
VelStand does not reproduce upstream protective-fall curricula or teacher distillation.
BallKick uses a prepositioned ball and blind actor; GroundPick has no grasped object.
New-task MuJoCo execution and graphical appearance were not separately validated in this run.
