> 当前性能设置：机器人自碰撞关闭，所有环境脚底摩擦固定为 1.0；地面/球体碰撞保留。自碰撞奖励不再有有效信号，摩擦随机化不再对齐官方。此变更需重启并新开训练，旧代码指纹检查点不可直接加载。

> 2026-10-21 稳定性修复：MicroDuck 的 SynReal 惯性积分从 Midpoint-Cayley 改为 implicit Euler，
> 并在运行配置 `inertia_integrator` 中记录。真实故障前状态的 GPU 回归在原实现第 2 个物理步失败，
> 修改后连续 100 个物理步通过。隐式欧拉增加数值阻尼，不等同于官方 MuJoCo 的 implicitfast。
> 保留物理异常检查，不裁剪奖励或静默重置异常环境。短回归通过不能证明长训练已稳定或策略已收敛。

# MicroDuck 官方流程移植状态

使用 `--recipe official_adapted_v1` 启用本版移植，支持 SynReal 和 MuJoCo。
当前只保留此配方，默认自动启用；旧 `legacy` 训练已删除。**奖励条目覆盖不代表训练流程、传感器数值或动作效果与官方完全一致。**

## 已移植奖励

从本地官方注册环境导出解析后的奖励、权重课程和指令配置至 `configs/official_snapshot.json`。
`official_reward_functions.py` 和 `official_velocity_rewards.py` 提取官方函数体；测试对照去除文档字符串后的 AST。
基础状态与奖励在 `official_recipe.py` 中适配。

本轮补充角动量、腾空时间、脚部离地间隙、摆动抬脚高度、脚部滑动、自碰撞、双脚着地和支撑脚着地。
保留官方公式、权重、指令门控及 VelStand 倒地门控。覆盖数包含权重为 0 的条目，也包含下述近似。

| 任务 | 接入 / 官方条目 | 仍遗漏的条目 |
| --- | --- | --- |
| `velocity` | 16 / 16 | 无 |
| `velstand` | 26 / 29 | `servo_impact`、`head_impact`、`trunk_impact` |
| `groundpick` | 16 / 19 | `soft_landing`、`mouth_payload_force`、`head_impact_penalty` |
| `ballkick` | 12 / 12 | 无 |

运行时差异及覆盖记录在训练目录 `config.json` 的 `official_alignment` 中。

## 传感器和碰撞实现

- `official_sensors.py` 计算全身各刚体自转及绕整体质心的轨道角动量，排除球体。
- 每个物理子步更新脚部几何接触计数、腾空/接触计时及首次落地状态；局部重置清理历史与摆动峰值。
  排除正间隙预测接触、球体及其他环境。`found` 保留接触数量，不压成二值。
- 脚部 site 速度由刚体质心线速度和角速度换算。高度传感器专用于当前无限平地，不支持复杂地形扫描。
- 新配方开启自碰撞，使用 MJCF 碰撞位掩码；为后添加的地面和球体补充过滤，避免只允许自碰撞的网格接触地面。
  BallKick 使用官方全碰撞资产。网格仍采用凸代理，几何及接触点数可能与官方不同。
- 自碰撞调用官方无力历史时的 `found` 分支。几何接触状态不能等同于求解后接触状态；两后端可能产生不同计数。

## 仍与官方不同

1. SynReal SDK 未提供求解后的接触力/冲量，所以上表冲击、软着陆奖励未接入，不会伪造零接触力。
   GroundPick 的 `mouth_payload_force` 是施加外部负载的钩子，并非读取接触力；官方快照权重为 0，
   本版尚未接入该外力流程，不能将它归因于缺少接触力接口。
2. 执行器仍是 position-PD，KP=2.0、KD=0.053、力矩上限 0.96，并非 BAM。
   力矩、力矩变化和堵转奖励使用最后子步的 PD 命令力矩，这是近似。
3. 特权 critic 已接入，但明确省略双脚的 6 维接触力。actor 61 维；velocity/velstand critic 70 维、
   groundpick 68 维、ballkick 74 维，对应官方 critic 为 76/74/80 维。
   接触标志和计时仍依赖几何接触近似，非求解后接触力。
4. 已接入脚底摩擦随机化和区间推扰，尚未移植质量/惯量、质心、armature、BAM 摩擦等物理随机化，
   也未接入 VelStand 专家 BC/教师蒸馏。
5. 已接入官方根位姿、关节偏移、VelStand 倒地/侧卧/蜷缩/坐姿/SO(3) 出生及其课程，
   并接入姿态终止课程、连续倒地 8 秒终止。BallKick 使用机身相对球位置与官方站姿初始化。
   这些分布与公式的移植不保证已训练出官方起身能力；非有限物理状态仍会报错退出。
6. 指令范围、重采样间隔、站立概率和部分课程已对齐，完整转向/前进混合采样及随机序列未对齐。
7. 后端求解器参数、约束容量与官方不同。MuJoCo 为倒地姿态预留 4096 个接触候选槽，避免站姿估计不足。
   训练器仍是本项目 RSL-RL PPO；新配方暂未启用 CUDA graph。

## 本轮新增观测与事件

- `official_observations.py` 按快照顺序组织 actor/critic，接入官方配置的均匀噪声、编码器常量偏差、
  IMU 安装偏差、截断/缩放与分环境延迟。critic 保持无噪声；BallKick 增加球相对位置与机身坐标球速。
  当前四任务官方快照没有启用观测滤波/历史，因此本版没有额外添加滤波；非空 modifiers/history 会明确拒绝。
- `official_reset_functions.py` 提取官方出生姿态与倒地判断函数体，AST 测试检查来源；
  `official_events.py` 适配状态写入、课程边界和推扰。区间推扰按官方实现累加平面速度。
- `official_randomization.py` 在求解器创建前，为每个环境采样脚底摩擦系数 [0.7, 1.3]，同一环境双脚共享。
  **不同环境的摩擦会使 SynReal 退出相同材质的 shared-world 优化，可能降低吞吐。**
  这是保留分环境摩擦随机化的性能代价，128 环境配置尚未测得最优吞吐。

已对齐物理步长 0.005 秒、4 子步、策略 50 Hz，奖励乘控制步长 0.02 秒。
新配置使用 actor/critic `[512, 256, 128]`、ELU、初始标准差 1.0、学习率 0.001、entropy 0.01、
rollout 24、5 epoch、4 mini-batch。这些参数一致不能代表整个训练过程一致。
BallKick 解析目标球速 1.0 m/s，episode 5 秒；其余任务 20 秒。
GroundPick 周期 4 秒，下降/保持/回升结束 phase 为 0.375/0.425/0.8。

## 启动训练

在项目根目录使用原 Python 环境和现有 SynReal 登录机制：

```bash
source .venv/bin/activate
python -u -m examples.microduck.train \
  --training-config examples/microduck/configs/rtx4090_synreal_official_adapted.json \
  --task velocity --num-envs 128 --iterations 10000 \
  --run-name synreal_velocity_official_adapted
```

替换 `--task` 为 `velstand`、`groundpick`、`ballkick`。128 不是已测得的最佳并行数，可先用 4 个环境、2 次迭代短测。
旧简化配方配置已删除。

```bash
python -u -m examples.microduck.play --solver synreal \
  --checkpoint /path/to/new_run/model_500.pt --viewer gl --num-frames 3000
python -u -m examples.microduck.train --solver synreal \
  --resume /path/to/new_run/model_500.pt --num-envs 128 --iterations 1000 --run-name continued
```

检查点恢复任务、配方、网络、优化器和归一化，并恢复课程计数；随后按恢复的课程重新初始化首局。
环境、随机数状态以及启动时采样的物理/传感器随机化值不保存，续训会重新采样。
本轮代码指纹已变化，之前 `official_adapted_v1` 检查点会被拒绝加载，请新开训练目录。
旧 `legacy` 配方已删除，不再支持其检查点。
不要绕过指纹保护；评估旧版新配方模型需恢复对应代码。

重新导出使用官方独立环境，训练使用主项目环境，避免依赖混用：

```bash
examples/microduck/microduck_rl-develop/.venv/bin/python examples/microduck/export_official_recipe.py
```

导出更新快照和指纹，应在新实验前进行。

## 验证范围（2026-10-21）

CPU 测试 **27 passed / 9 skipped**；跳过项需要显式启用 GPU/SynReal。Ruff 检查与 128 列格式检查通过。
覆盖官方函数来源、角动量自转/轨道项及平移不变性、接触计时、落地、摆动峰值/重置、
几何接触过滤及自碰撞网格的地面过滤，以及观测延迟/局部重置、噪声与干净 critic、课程出生及终止边界。

本轮 RTX 4090 SynReal 四任务各 4 环境通过：20 控制步、课程推进至 33600 步后的重置、立即推扰、
10 控制步，以及一次 PPO 更新（8 步 rollout）、参数有限值、检查点保存/加载。
MuJoCo 对照首轮发现倒地姿态碰撞候选容量不足，扩容后四任务通过相同 GPU 短测。
SynReal VelStand 还通过实际 `train.main` 新训练一次、续训一次及 `play.main` 无窗口回放 5 帧，
包含启动时恢复课程后重置首局的流程。该回放仅检查功能，不用于帧率评估。
这属于功能短测，未完成可视化动作质量或长期收敛验证。
临时脚本与日志在 `.validation/microduck-v2`，包括保留的容量溢出失败日志；此前传感器验证在
`.validation/microduck-sensors`。CPU 测试在沙箱内需将 `WARP_CACHE_PATH` 指向可写缓存目录。
账号密码未写入脚本或配置。

## 删除旧简化配方后的整理

默认和唯一训练配方为 `official_adapted_v1`。删除旧 `rewards.py`、`observations.py`、
`simulation_kernels.py`、`pretrained_synreal.py` 及两个旧 RTX 4090 配置。
`tasks.py` 仅保留任务名称、球体创建、出生状态容器和旋转工具；奖励、观测、课程全部走官方适配模块。
旧 CUDA graph 训练/回放分支删除，原官方 walk 演示的独立实现保留。
本次源文件变更会改变检查点代码指纹；此前模型需使用对应旧代码，不能绕过校验。
整理后单元测试 25 项通过；原来仅适用于旧简化配方的测试已删除，新增默认配方及旧检查点拒绝测试。
整理后 SynReal / MuJoCo 四任务均再次通过 GPU 短测：4 环境、后期课程重置与推扰、一次 PPO 更新及保存/加载。
日志为 `.validation/microduck-v2/cleanup_synreal.log` 和 `cleanup_mujoco.log`；未重新进行长期收敛验证。

## 整理后复查（2026-10-21）

- 使用官方独立环境重新加载四个注册任务，将 rewards、curricula、observations、events、terminations
  与当前快照逐项比较，四任务全部一致；未重新导出或修改官方文件。
- 25 项单元测试及 Ruff 检查通过，包括官方函数 AST 来源、观测、课程、传感器及检查点校验。
- SynReal VelStand 实际入口再次完成新训练 1 次更新、续训 1 次更新和无窗口回放 5 帧。
  使用 4 环境、正式 PPO 设置（24 步 rollout、5 epoch、4 mini-batch），日志为
  `.validation/microduck-v2/audit_entrypoints.log`。
- 四任务 GPU 覆盖沿用同一精简后代码的 `cleanup_synreal.log` / `cleanup_mujoco.log`，
  本次额外复查了正式训练入口；没有开展 128 环境长期收敛测试。
- 快照一致不代表所有条目已实现。前述遗漏奖励、接触近似、PD/BAM、BC、随机化和指令采样差异仍存在。
  保存间隔 500、训练预算 10000 为本项目设置；例如官方 VelStand 为 250、6000。

## 专用回放验证

新增 `playback.py`，训练配方文件不变。33 项 CPU 测试通过，包含 actor 噪声、延迟和局部重置的逐值对照。
四任务在两后端完成 GPU 有限状态、重置次数、课程计数及同一状态 actor 逐值对照。
MuJoCo Velocity/VelStand/GroundPick 还通过完整轨迹数值对照；BallKick 两份原始完整环境也出现
独立轨迹差异，因此只验证同状态观测等价，不承诺逐位轨迹一致。SynReal 同样不承诺独立轨迹一致。
单环境零动作无窗口短测（30 控制步，不含网络推理/渲染，可能受并行 GPU 任务影响）：
SynReal Velocity 9.55→15.72、VelStand 8.57→15.66、GroundPick 9.73→15.86、BallKick 9.62→14.79 步/秒。
这些结果不是实时 50 Hz 或图形窗口 FPS 保证。日志与脚本在 `.validation/playback/`。
