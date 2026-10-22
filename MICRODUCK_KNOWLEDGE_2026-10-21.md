# MicroDuck 知识库：2026-10-21

记录范围：当天对话中已确认的实现、使用方法、性能变化与限制。当前配置值已在整理时核对本地文件。
这是一份日期快照；后续代码或配置更新后，以实际代码和新运行目录的 `config.json` 为准。
不收录账号、密码或其他登录凭据。

最新交接结论与命令见文末“会话结束交接”；早期短测结果不能代替后续训练效果评估。

## 1. 当前项目状态

- 训练只保留 `official_adapted_v1` 官方移植配方，默认自动启用。
- 四个任务：`velocity`、`velstand`、`groundpick`、`ballkick`。
- SynReal / MuJoCo 共用任务接口，但物理求解器不同；当前都使用 position-PD，尚不是官方 BAM 执行器。
- **不是完整复现官方训练。**奖励覆盖、权重配置相同不意味着动力学、传感器或最终效果相同。
- 官方原始目录 `examples/microduck/microduck_rl-develop/`、`examples/microduck/microduck-main/` 保留。
- 官方权重、历史训练记录和检查点保留，但旧配方模型不再由当前训练/回放入口兼容。

旧简化功能已移除：`rewards.py`、`observations.py`、`simulation_kernels.py`、`pretrained_synreal.py`，
以及 `rtx4090_synreal_probe.json`、`rtx4090_mujoco.json`。`tasks.py` 只保留任务名称、场景资产和相关工具。
原官方 ONNX 回放、官方训练启动器与 walk 演示仍保留。

## 2. 任务与官方奖励对齐情况

| 任务 | 目标 | 接入 / 官方奖励条目 | 未接入条目 |
| --- | --- | --- | --- |
| Velocity | 速度、转向、姿态指令跟随 | 16 / 16 | 无遗漏条目 |
| VelStand | 移动、姿态控制、倒地恢复 | 26 / 29 | `servo_impact`、`head_impact`、`trunk_impact` |
| GroundPick | 嘴部接近地面并恢复姿态 | 16 / 19 | `soft_landing`、`mouth_payload_force`、`head_impact_penalty` |
| BallKick | 踢预定位球并维持姿态 | 12 / 12 | 无遗漏条目 |

计数包含权重为零和近似实现的条目。**当前关闭自碰撞后，各任务的 `self_collisions` 虽仍在奖励表中，
但不再具有有效的自碰撞信号。**GroundPick 当前不包含实际抓取物体。

对齐依据是本地官方注册任务导出的 `configs/official_snapshot.json`，不是自动跟随上游更新。
当天已用官方独立环境重新加载四任务，确认快照的 rewards、curricula、observations、events、terminations
与注册配置一致；提取的官方函数还通过 AST 来源检查。

已接入奖励使用快照中的参数、基础权重和权重课程。总奖励按控制步长 0.02 秒缩放。
配置中初始为零的奖励可能在后期课程中启用，不能只看初始权重判断是否有效。

### 已移植的主要流程

- actor/critic 非对称观测；actor 61 维，critic：Velocity/VelStand 70、GroundPick 68、BallKick 74 维。
- actor 噪声、编码器常量偏差、IMU 安装偏差、观测延迟、截断与缩放；critic 保持干净。
- 当前四任务快照没有启用额外观测滤波或历史，因此未额外添加；不等于延迟未启用。
- 根位姿和关节初始化、VelStand 倒地/侧卧/坐姿等出生课程、终止课程与持续倒地计时。
- 周期推扰、指令与奖励的部分课程。
- 物理步长 0.005 秒、每控制步 4 个物理子步，即策略频率 50 Hz。

### 尚未对齐或存在近似

- SynReal 接口未暴露所需的求解后接触力/冲量；冲击、软着陆等奖励缺失，critic 也省略了双脚 6 维接触力。
- `mouth_payload_force` 是未移植的外部负载施加流程，不是缺少接触力读取接口；当前官方快照权重为零。
- 接触、腾空计时和脚滑等使用几何接触近似；地形高度适配当前无限平地。
- PD 控制替代 BAM，力矩相关奖励使用 PD 命令力矩。
- 质量/惯量、质心、armature、BAM 摩擦等随机化未完整移植；专家 BC/教师蒸馏未接入。
- 完整指令混合采样、求解器设置和物理结果未完全对齐。
- 非有限物理状态会报错，而非完整复用官方逐环境 NaN 终止处理。
- 自碰撞与分环境脚底摩擦随机化已按性能需求关闭，属于主动偏离官方。

## 3. 配置层级与当前数值

| 文件 | 用途 |
| --- | --- |
| `examples/microduck/configs/rtx4090_synreal_official_adapted.json` | 新训练模板：后端、环境数、更新次数、PPO、网络、保存间隔 |
| `examples/microduck/config.py` | PPO 代码默认值 |
| `examples/microduck/configs/official_snapshot.json` | 官方任务奖励、课程、观测、事件等导出配置 |
| 每次训练目录中的 `config.json` | 该次运行实际配置及检查点加载元数据 |

`--training-config 路径` 表示读取指定 JSON。新训练时，终端显式参数覆盖 JSON 同名参数。
例如 JSON 环境数为 1024，但命令写了 `--num-envs 128`，最终使用 128。
`--run-name` 只是运行目录标签；任务由 `--task` 选择。

整理文档时，新训练 JSON 的实际值：

| 参数 | 当前值 |
| --- | --- |
| solver / device | synreal / cuda:0 |
| num_envs / iterations | 1024 / 10000 |
| viewer | null（无窗口） |
| num_steps_per_env | 24 |
| save_interval | **300** |
| actor / critic hidden dims | [512, 256, 128] |
| activation / init_noise_std | ELU / 1.0 |
| learning_rate / entropy_coef | 0.001 / 0.01 |
| epochs / mini-batches | 5 / 4 |

注意：此前对话中出现过保存间隔 500、环境数 128，属于当时配置，不是本次核对的当前值。
`config.py` 的默认保存间隔仍为 500，使用上述 JSON 新训练时会被 300 覆盖。
保存间隔和训练预算是本项目设置，不保证与各个官方任务相同，例如官方 VelStand 配置为 250 / 6000。

## 4. 常用终端命令

在项目根目录执行；SynReal 使用现有登录机制，缺少环境变量时隐藏输入凭据。

```bash
cd /home/ubuntu/work/style3d-sim
source .venv/bin/activate

# 任选 velocity、velstand、groundpick、ballkick
TASK=velstand
python -u -m examples.microduck.train \
  --training-config examples/microduck/configs/rtx4090_synreal_official_adapted.json \
  --solver synreal \
  --task "$TASK" \
  --run-name "synreal_${TASK}_official"
```

以上环境数、更新次数使用 JSON。首次检查可追加 `--num-envs 4 --iterations 2`。
训练显示窗口追加 `--viewer gl`，可能影响吞吐。切换 MuJoCo 训练使用 `--solver mujoco`。
四任务是分别选择训练，不是这条命令自动训练一个融合四技能的模型。

```bash
# 同后端回放：替换成实际检查点
python -u -m examples.microduck.play \
  --solver synreal \
  --checkpoint /完整路径/训练目录/model_300.pt \
  --viewer gl --num-frames 3000

# 跨后端回放：SynReal 训练的模型用 MuJoCo 物理运行
python -u -m examples.microduck.play \
  --solver mujoco --allow-cross-backend \
  --checkpoint /完整路径/训练目录/model_300.pt \
  --viewer gl --num-frames 3000

# 续训：需兼容的检查点，后端保持相同
python -u -m examples.microduck.train \
  --solver synreal \
  --resume /完整路径/训练目录/model_300.pt \
  --iterations 1000 --run-name continued

# 曲线
tensorboard --logdir logs/rsl_rl
```

路径中的 `/` 与模型文件名之间不能有空格；`--viewer` 不能拆成 `--` 和 `viewer`。
回放从检查点目录恢复任务、网络及归一化，不需要再次传训练模板或任务参数。
Velocity/VelStand 的窗口按键：I/K 前后、J/L 横移、U/O 转向，每按一次增减一档。
默认移动指令为零，可追加 `--command 0.1 0 0`。

跨后端保留策略与任务接口，改变实际物理求解器；接触、摩擦和动力学结果可能不同，动作效果不保证一致。
窗口共用 ViewerGL，因此外观相同不代表引擎相同。

## 5. SynReal 路径确认

`--solver synreal` 创建 `s3d.ModelBuilder` 和 `s3d.NewtonSolver`，每个子步调用该实例的 `step()`，
重置时同步 SynReal affine body 状态。MuJoCo 路径创建 `newton.solvers.SolverMuJoCo`。
没有自动回退到 MuJoCo 的逻辑。

`newton.eval_fk`、`eval_ik`、碰撞管线与 Viewer 是共享工具，出现 `newton` 名称不等于使用 MuJoCo。
启动输出 `MicroDuck: solver=synreal, ...` 可用于核对 CLI 选择，但实际实现以环境中的 solver 实例为准。

## 6. 检查点与续训限制

结果目录：`logs/rsl_rl/microduck_<solver>/<时间>_<run-name>/`。
保留 `model_*.pt` 及同目录 `config.json`，不要只搬走权重文件。

- 当前仅支持 `official_asymmetric_pd_v2` 观测契约及当前配方指纹匹配的模型。
- 报 `legacy or unsupported checkpoint`：观测契约不兼容，不是路径文件名问题。
- 报配方文件不一致：当前代码指纹不同，需要对应版本代码，不能修改标签绕过。
- 整理旧分支、改变碰撞/摩擦设置曾改变指纹；旧模型文件虽保留，不一定能在当前代码加载。
- 本次专用回放提速未修改训练配方文件，不改变指纹，无需因此重新训练。
- 续训恢复 PPO、优化器、归一化和课程计数；首局按恢复的课程重置。
- 环境状态、随机数状态和启动随机化样本不完整保存，因此不是逐步完全一致的继续执行。
- 续训使用检查点旁保存的 PPO 配置，不重新套用新训练模板中的 PPO 参数。

`model_0.pt` 是第 0 次 PPO 更新完成后的保存，不是零训练随机权重。
初期回放看起来能站立，可能来自站立出生姿态、PD 保持默认关节角、均值动作推理和零移动指令。
这不证明已学会行走、指令跟随或倒地起身；恢复能力需要倒地初始状态评估。

## 7. 当前性能设置与影响

### 训练性能设置

- 机器人自碰撞关闭；机器人与地面、球体的碰撞保留。
- 当前机器人 XML 资产和凸包几何保留；不因自碰撞开关改变资产。
- 所有环境脚底摩擦固定为 1.0，其余材质不由该设置修改。
- 地面/球体未设置碰撞掩码的整数哨兵值按 MuJoCo 默认掩码 1 处理，避免误过滤。
- 已有测试检查四任务的脚—地面碰撞过滤、固定脚底摩擦及其他材质未被修改。

关闭自碰撞使身体部件可能相互穿透，特别会影响起身真实性；自碰撞惩罚不再有效。
固定摩擦减少地面泛化训练，但其他奖励公式/权重可保留。
相同材质允许 SynReal 在其他条件满足时启用共享布局；是否启用需查实际求解器状态，不能只凭吞吐认定。
源代码修改需要重启进程才生效，正在运行的训练不会自动应用新文件。

### 如何解读吞吐日志

用户曾提供：598 steps/s，collection 40.764 秒、learning 0.283 秒。
后续提供：6447 steps/s，collection 3.757 秒、learning 0.055 秒。
每次仍是 `1024 × 24 = 24576` 个采样步，总吞吐约增加 10.8 倍。
主要变化在仿真采样，不能将全部提速直接归因于某一个开关；首次编译、碰撞状态和 GPU 竞争也会影响结果。
采样包含物理、观测、奖励和重置，不只是纯求解器时间。

`steps/s` 是所有环境合计，不是可视化窗口 FPS。按每次 3.81 秒估算，剩余约 9968 次更新约需 10.5 小时；
ETA 受历史耗时和后续课程影响。低 fall_fraction 也不等于已学会稳定动作，仍应结合 episode length 和实际回放。

### 专用回放提速

原 `step_playback()` 调用完整训练步。当前 `play.py` 默认使用独立的 `PolicyPlayback`：
省去奖励、训练日志、critic 观测和奖励专用的求解后碰撞/传感器更新。
保留 actor 噪声/延迟、PD、物理子步、求解前碰撞、终止课程、重置、命令和推扰。
它不用于训练，所以本次回放优化不会直接加速训练。

```bash
# 无窗口测速
python -u -m examples.microduck.play --solver synreal \
  --checkpoint /完整路径/训练目录/model_300.pt \
  --viewer null --benchmark --num-frames 300
```

追加 `--full-step` 可使用原完整路径进行对照。`--benchmark` 取消实时等待，不会消除计算瓶颈。
当前官方移植路径没有启用 CUDA graph；`--disable-cuda-graph` 对原 walk 演示仍有效。

SynReal 单环境零动作、无网络推理/渲染短测：

| 任务 | 完整路径（控制步/秒） | 快速路径（控制步/秒） |
| --- | ---: | ---: |
| Velocity | 9.55 | 15.72 |
| VelStand | 8.57 | 15.66 |
| GroundPick | 9.73 | 15.86 |
| BallKick | 9.62 | 14.79 |

测量可能受其他 GPU 任务影响，不是窗口 FPS 或 50 Hz 实时保证。
同一 GPU 上同时训练与回放也可能降低窗口帧率。

## 8. 验证结论及边界

- 最后一次回放优化后：33 项单元测试通过。
- 四任务完成两后端 GPU 短测；此前验证过 PPO 更新、保存/加载，SynReal VelStand 验证过实际新训练→续训→回放。
- 优化回放验证 actor 噪声/延迟/局部重置的一致性、GPU 有限状态、重置次数及课程计数。
- MuJoCo Velocity/VelStand/GroundPick 通过逐步轨迹对照；BallKick 两份原完整环境也存在独立轨迹差异，
  因此只保证测试中的同状态 actor 观测一致，不宣称独立物理轨迹逐位一致；SynReal 同样如此。
- 尚未验证长期收敛、所有训练规模稳定性、最终技能质量或实机部署。

本地验证证据：`.validation/microduck-v2/`、`.validation/playback/`。这些不是生产训练入口。

## 9. 文件导航

| 文件 | 职责 |
| --- | --- |
| `examples/microduck/train.py` | 训练、保存与续训 |
| `examples/microduck/play.py` / `playback.py` | 窗口回放入口 / 专用快速回放 |
| `examples/microduck/common.py` | CLI、配置读取、登录与检查点校验 |
| `examples/microduck/env.py` | 后端构建、物理步进、重置与训练主循环 |
| `examples/microduck/policy_spec.py` | 资产、关节顺序、PD、碰撞过滤 |
| `examples/microduck/official_recipe.py` | 奖励适配、加权汇总、课程与指纹 |
| `examples/microduck/official_reward_functions.py` | 提取的官方任务奖励函数 |
| `examples/microduck/official_velocity_rewards.py` | 提取的官方步态奖励函数 |
| `examples/microduck/official_observations.py` | actor/critic、噪声与延迟 |
| `examples/microduck/official_sensors.py` | 接触计时、高度及角动量等计算 |
| `examples/microduck/official_events.py` / `official_reset_functions.py` | 事件适配 / 官方重置与终止函数 |
| `examples/microduck/official_randomization.py` | 当前固定脚底摩擦设置 |
| `examples/microduck/export_official_recipe.py` | 从官方独立环境导出快照与函数 |

相关文档：[日常使用](examples/microduck/README.md)、[移植差异与验证](examples/microduck/OFFICIAL_PORT_STATUS.md)、
[官方奖励索引](examples/microduck/OFFICIAL_REWARDS_INDEX.md)。

## 10. 训练提速追加审计

读取真实 TensorBoard 记录后确认：

| 运行目录时间 | 配置 | 采样耗时 |
| --- | --- | --- |
| 16:47 | 1024 环境，分环境随机摩擦，调整前配置 | 约 41 秒/次，前 20 次仍慢 |
| 17:03 | 1024 环境，自碰撞关闭、固定摩擦 1.0 | 前期约 11～13 秒/次 |
| 17:50 | 与 17:03 保存的完整 config.json、配方指纹、Git diff 相同 | 从第 0 次就约 3.8 秒/次 |

用户贴出的第 31 次 6447 steps/s 来自 17:50 的新进程，不是该进程在第 31 次突然变快。
17:03→17:50 的约 3 倍差异不能归因于配置开关或回放优化；运行时 GPU 竞争、时钟/资源状态等为候选，
缺少历史资源采样，不能确定单一原因。16:47 前 20 次持续约 41 秒，因此首次编译也不能解释那次持续低吞吐。

独立 4 环境 SynReal 实测：每控制步 4 次 solver.step、4 次传感器更新、1 次奖励和观测计算，
仿真时间前进 0.02 秒，物理状态变化，actor/critic 有限；初期有 23 个非零权重奖励项进入计算。
一次正式 PPO 更新确实采样 24 控制步、执行 96 物理子步，网络参数更新且有限。
4 环境实例 `_shared_world_count=4`，确认当前设置能够启用共享布局；未读取用户运行中 1024 环境实例的该字段。
33 项测试再次通过。证据：`.validation/training_audit/result.log`。
训练代码未修改，未停止用户训练。

日志提醒：`Physics/reset_count` 是累计重置数在 rollout 内的均值，因此可能为小数，不是本次迭代新增重置数。
`Physics/fall_fraction` 是固定高度/倾斜阈值的诊断量，与课程中的实际终止条件不同，不应当作 episode 终止率。

## 11. 长期训练发散（推翻“短测意味着稳定”的推断）

17:50 的 VelStand 运行在迭代 240 首次出现数量级异常：机身角速度奖励约 -6.7e9，
价值损失约 4.3e31；241 次价值损失为 inf，下一轮在 angular_momentum 奖励报错。
异常时刻约控制步 5760～5807，并非快照中该时刻的新课程边界。不能断言课程切换或自碰撞关闭就是根因。
该运行目录只保存了 model_0.pt（保存间隔 300，尚未到下一保存点）。

已确认缺口：原环境仅检查 NaN/Inf，有限但已发散的刚体速度能进入奖励和 PPO。
新增 stability.py，在每个物理子步后、碰撞传感器/奖励前检查所有刚体：
位置分量绝对值 >1000 m、线速度分量 >100 m/s、角速度分量 >1000 rad/s 或非有限值会停止。
这些是宽松诊断阈值，不是官方奖励/终止定义；不截断奖励、不静默重置继续训练。
保存 logs/microduck_diagnostics/*_unstable.pt，记录异常环境、前后物理状态、动作、目标和课程步数。
训练与快速回放均启用；配方指纹因此改变。此防护有额外检查开销。

底层求解器首次发散的触发机制尚未确定。防护仅防止明显异常继续污染 rollout，不能宣称已修复长期稳定性。
之前“短测通过、未发现少执行物理”的结论仍成立，但不能用于保证长时间训练安全收敛。
# 稳定性修复补记：SynReal 积分器（2026-10-21 晚）

后台 `attempt_01` 在第 100 次迭代后触发物理保护：环境 366 的右脚踝在 0.005 秒内旋转约 176.5°，
Midpoint-Cayley 反解将角速度放大至约 88733 rad/s。故障前状态重建实验中，原积分器第 2 个物理步
再次发散；仅换隐式欧拉后连续 100 个物理步通过。该实验不包含原求解器内部缓存，不能确定更早的大角速度来源。

`examples/microduck/env.py` 现使用 `InertiaMode.IMPLICIT_EULER`，运行配置记录 `inertia_integrator`。
奖励/权重、PD、0.005 秒步长与 4 子步未修改，异常保护保留。隐式欧拉增加数值阻尼，不等同官方 MuJoCo implicitfast。
35 项现有测试及新增的 SynReal GPU 故障回归通过。新代码指纹与旧检查点不同，本轮从头训练。

新运行：`logs/rsl_rl/microduck_synreal/2026-10-21_18-31-24_850816_synreal_velstand_implicit_euler`。
已观察至第 250 次迭代，越过此前 ~101、~241 两处故障位置，日志指标全部有限，无新异常快照。
第 250 次平均奖励 11.95、价值损失 0.3212、角动量均值 0.0087；这不代表 10000 次训练已完成或动作已收敛。

查看本轮实时输出：

```bash
cd /home/ubuntu/work/style3d-sim
tail -n 80 -f logs/microduck_supervised/attempt_02.log
```

只读监测器每 5 秒更新 `logs/microduck_supervised/latest_status.json`，记录退出/报错，不自动修复或重启。
详细证据：`.validation/microduck-stability/REPORT.md`。

## 会话结束交接：2026-10-21 晚

### 当前运行与训练效果

- 22:51:08 只读监测状态：`attempt_02` 仍运行，PID 1031850，迭代 4167/10000。
  当次奖励 -13.76、价值损失 0.6558、吞吐 3730 steps/s。这是当时快照，不保证以后仍运行。
- 最新状态文件：`logs/microduck_supervised/latest_status.json`；实时输出：
  `logs/microduck_supervised/attempt_02.log`。
- 监测器只记录状态，不自动修复或重启；关闭会话后不能假定助手仍持续监测。
- 当前轮次仍在训练 VelStand。本次会话未停止它，也未启动下方 Velocity 命令。

已分析的约 100 次迭代均值：

| 迭代区间 | 平均奖励 | 价值损失 | 倒地采样占比 |
| --- | ---: | ---: | ---: |
| 1900–1999 | -21.55 | 1.02 | 95.2% |
| 2900–2999 | -14.41 | 0.71 | 95.7% |
| 3900–3990 | -10.52 | 0.69 | 95.5% |

近期奖励有所恢复、价值损失总体降低，但策略起身效果仍不理想，不能称为成功收敛。
价值损失下降只说明 critic 拟合变化；PPO surrogate loss 不要求单调下降。
前期存在课程和奖励权重变化，不宜把整个训练的奖励直接作同条件比较。
`recovery_success` 日志是每控制步的一次性恢复事件的加权均值，不是回合成功率。

### 为什么回放经常倒地

VelStand 同时训练移动指令、姿态控制和倒地恢复，不是纯起身任务。
`play.py` 从检查点迭代恢复课程计数后重新 reset，因此沿用该阶段的出生分布。
后期重置约 45% 倒地、20% 蹲姿、35% 正常站姿，还存在推扰动。
但约 95% 的倒地采样占比不能仅由倒地出生解释：站姿保持和倒地恢复都需要进一步评估。
可视化中“出生就躺着”和“站着出生后跌倒”应分别观察；单次站立不证明已经学会起身。

### 只训练移动指令：新建 Velocity 训练

用户希望获得纯移动指令训练命令。这里“纯移动”指 Velocity 配方，不含 VelStand 起身课程；
仍保留 Velocity 官方配方中的姿态等辅助奖励和观测，并非只剩速度奖励。
这是当前 `official_adapted_v1` 适配版本，不是已删除的早期简化训练。
奖励项接入不等于完整复现官方；PD/BAM、接触近似、物理随机化及积分器等差异仍见本文前述章节。

```bash
cd /home/ubuntu/work/style3d-sim
source .venv/bin/activate

python -u -m examples.microduck.train \
  --training-config examples/microduck/configs/rtx4090_synreal_official_adapted.json \
  --solver synreal \
  --task velocity \
  --num-envs 1024 \
  --iterations 20000 \
  --run-name synreal_velocity_official
```

该命令从头训练 Velocity，不加载 VelStand。运行前先确认旧训练是否仍占用 GPU。

### 续训轮数的准确语义

`train.py` 加载模型、优化器、归一化状态，并将已完成迭代索引加一，再调用
`runner.learn(args.iterations, ...)`。因此 `--iterations` 是本次追加轮数，不是目标总轮数。
修改 JSON 不会改变已经运行中的进程。续训会恢复课程计数，但不会精确恢复所有仿真/随机数状态。

从下列 model_3900.pt 追加 20000 次更新：

```bash
python -u -m examples.microduck.train \
  --solver synreal \
  --resume /home/ubuntu/work/style3d-sim/logs/rsl_rl/microduck_synreal/2026-10-21_18-31-24_850816_synreal_velstand_implicit_euler/model_3900.pt \
  --num-envs 1024 \
  --iterations 20000 \
  --run-name synreal_velstand_resume_20000
```

若目标是从该检查点训练至累计 20000 次更新，追加轮数为 16099。
若先跑完当前 10000 次，再追加 20000 次，则累计约 30000 次。
实际续训时优先选择当时最新的有效检查点；续训输出进入新的日志目录。
单纯延长预算不保证解决倒地恢复失败。

### 本轮检查点可视化评估

以下 model_3900.pt 已确认存在，且在本次检查时与当前配方指纹一致。
任务、配方从检查点目录读取；无需附带包含 1024 环境的新训练模板。

```bash
cd /home/ubuntu/work/style3d-sim
source .venv/bin/activate

python -u -m examples.microduck.play \
  --solver synreal \
  --checkpoint /home/ubuntu/work/style3d-sim/logs/rsl_rl/microduck_synreal/2026-10-21_18-31-24_850816_synreal_velstand_implicit_euler/model_3900.pt \
  --num-envs 1 \
  --viewer gl \
  --num-frames 3000
```

实时读日志可使用前文的 `tail -f` 命令，Ctrl+C 只关闭日志查看，不会停止训练。
下一次会话应先检查进程、最新日志及检查点，再决定继续训练或分开评估站立移动与倒地起身；
不要把本次“数值暂时稳定”误认为策略质量已经合格。

