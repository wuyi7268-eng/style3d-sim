# MicroDuck / SynReal 项目介绍与 AI 交接

更新日期：2026-10-27（Asia/Shanghai）。适用目录：`examples/microduck/`。

本文是本目录唯一维护的项目说明、使用向导和 AI 交接正文。
合并了顶层 README、AGENTS、ALL_TASKS、OFFICIAL_PORT_STATUS、VALIDATION 和 OFFICIAL_REWARDS_INDEX。
旧文件只保留导航，不再维护独立事实。两个官方子项目中的 Markdown 保持原样，不属于本次合并范围。

接手时先读第 1～12 节；按需查看第 13 节历史证据和附录源码索引。

快速导航：[目的](#purpose) · [接手规则](#rules) · [代码导航](#architecture) · [任务配置](#configuration) ·
[运行命令](#commands) · [物理契约](#contracts) · [移植差异](#alignment) · [Velocity](#velocity) ·
[VelStand](#velstand) · [回放续训](#checkpoints) · [实验与待办](#next) · [验证](#validation) ·
[历史问题](#history) · [来源维护](#sources) · [奖励源码索引](#reward-index)

文档与代码不一致时核对代码；解释某次实验时以其 `config.json`、checkpoint 和对应代码版本为准。

<a id="purpose"></a>

## 1. 项目目的与当前结论

**主要目的是用 SynReal 仿真引擎训练并复现官方 MicroDuck 任务。**
用户先用本地共享环境的 MuJoCo 后端建立基线，核对奖励和训练流程，再与 SynReal 做受控对照。
长期目标包括官方全部任务及 Rough / Backlash 变体；当前关注稳定行走、指令跟踪、零指令停稳和自然步态。

本地官方来源：

- [microduck_rl-develop](microduck_rl-develop/README.md)：用户从官方 GitHub 拉下来的训练项目。
  上游为 `https://github.com/pollen-robotics/microduck_rl`，使用 mjlab / MuJoCo Warp、BAM、PPO。
- [microduck-main](microduck-main/README.md)：官方机器人运行时、资产和部署资料。
  上游为 `https://github.com/pollen-robotics/microduck`。它的 daemon 仿真不是这里的共享 RL 训练入口。
- 两个目录是本地官方参考副本，不代表自动跟踪最新上游；不要擅自更新、清理或重写。

**当前属于部分官方配方适配，不是已经完整复现官方。**
33 个注册任务已接入共享入口；这表示配置与代码路径已覆盖，不表示 33 个技能都收敛。
奖励来源对齐、短测通过和仿真动作自然是三种不同证据。

MuJoCo / SynReal 共用 PPO、奖励、观测、课程和 PD 控制代码；区别在后端求解和必要状态适配。
两者仍使用 position-PD，不是官方 BAM；教师 BC、部分随机化和力传感器相关项尚未完整移植。

<a id="rules"></a>

## 2. 下一位 AI 首先要知道

1. 先执行 `git status --short` 并看相关 diff。当前目录有大量用户未提交实现和新文件；不要 reset 或覆盖。
2. 使用仓库根 `.venv` 训练本地共享环境。官方子项目 `.venv` 仅用于官方训练/导出等对应入口。
3. 修改代码行宽为 128，必要代码注释用英文，解释意图、约束和取舍。
4. 用户要求奖励权重集中放置：`configs/official_snapshot.json → tasks → task → rewards`。
   本地定制奖励使用 `local: true`；不要重新把单个权重散放成 Python 全局常量。
5. 默认配方为 `official_adapted_v1`，观测契约为 `official_asymmetric_pd_v2`；旧简化配方已删除。
6. 不把未经验证的根因说成结论；先看实际 Actor 指令、物理状态和版本，不仅看窗口外观或总奖励。
7. 更改奖励/动力学会改变配方指纹。旧 checkpoint 不会因新奖励自动改善；运行中的训练也不会热更新文件。
8. 当前没有本地跨任务热启动或双教师训练接口。用户讨论过如何实现，但尚未授权并完成这项实现。
   不要声称给出两个 checkpoint 路径就已经等价于官方 VelStand。
9. 用户早期希望无教师训练；后续在了解官方双教师流程。下一步以用户最新目标为准，不擅自切换训练方法。
10. 在已授权范围内直接完成检查和修改，不反复请求已有许可；不要默认启动 1024 环境长训练。
11. 不泄露 `STYLE3D_ACCOUNT` / `STYLE3D_PASSWORD`，不把凭据写入文档、脚本、日志或运行元数据。
12. 进入官方子目录修改前分别阅读其 `AGENTS.md`；本文描述本地适配层，不替代官方目录内部规则。

<a id="architecture"></a>

## 3. 代码导航与三条运行路径

| 文件 / 目录 | 职责 |
| --- | --- |
| [train.py](train.py) | 本地共享 PPO 新训练、保存、同任务续训 |
| [common.py](common.py) | CLI、JSON 默认值、SynReal 登录、checkpoint 契约检查 |
| [config.py](config.py) / [official_ppo.py](official_ppo.py) | PPO 默认值、任务参数映射、Roulade 镜像损失 |
| [env.py](env.py) | 后端创建、PD 控制、步进、重置、奖励与事件调度 |
| [policy_spec.py](policy_spec.py) | 机器人资产、14 舵机顺序、HOME、增益、碰撞过滤 |
| [task_registry.py](task_registry.py) / [tasks.py](tasks.py) | 33 任务映射、球体场景、出生状态、旋转工具 |
| [official_recipe.py](official_recipe.py) | 配方加载、指令采样、奖励适配、课程与指纹 |
| [official_reward_functions.py](official_reward_functions.py) | 提取的官方任务奖励函数 |
| [official_velocity_rewards.py](official_velocity_rewards.py) | 提取的 mjlab 步态奖励函数 |
| [official_observations.py](official_observations.py) | Actor/Critic、噪声、偏差、延迟、缩放和缓存 |
| [official_events.py](official_events.py) | 重置、终止课程和推扰适配 |
| [official_reset_functions.py](official_reset_functions.py) | 提取的官方重置/倒地判断公式 |
| [official_randomization.py](official_randomization.py) | 启动时脚底摩擦随机化 |
| [official_sensors.py](official_sensors.py) / [official_adapter.py](official_adapter.py) | 几何接触计时、角动量、状态约定转换 |
| [official_terrain.py](official_terrain.py) | Rough/Slope 几何、出生原点、地形高度查询 |
| [stability.py](stability.py) | 子步物理异常检查与故障快照 |
| [play.py](play.py) / [playback.py](playback.py) | 本地 `.pt` 回放及快速 Actor 路径 |
| [official_play.py](official_play.py) | 原官方 `walk` 演示路径，控制配置与本地训练不同 |
| [pretrained.py](pretrained.py) | 官方 ONNX 的 CPU MuJoCo 推理，可选择 BAM |
| [mjlab_train.py](mjlab_train.py) | 转发给官方独立环境训练，不是 SynReal 入口 |
| [export_official_recipe.py](export_official_recipe.py) / [export_terrain.py](export_terrain.py) | 官方配方/函数与固定地形导出 |
| [validate.py](validate.py) / [tests](tests/) | 站立检查与回归测试；不是技能收敛证明 |

不要混淆以下路径：

- `python -m examples.microduck.train --solver mujoco|synreal`：本项目共享 PD 训练。
- `python -m examples.microduck.mjlab_train`：官方 mjlab 训练；使用官方任务 ID 和独立 Python。
- `python -m examples.microduck.play --checkpoint ...`：本地 PD checkpoint 回放。
  不传 checkpoint 时默认进入原 `walk` 演示，并不是刚训练的模型。

共享主流程：

```text
CLI / JSON → MicroDuckEnv → 官方快照 + 本地适配 → RSL-RL OnPolicyRunner
                              ↓
Actor 动作 → HOME + actions → 每个物理子步计算并裁剪 PD 力矩
                              ↓
                    MuJoCo 或 SynReal 步进
                              ↓
几何传感器 → 奖励 × 0.02 → 终止/重置/指令/推扰 → 新观测
                              ↓
              每轮 24 个控制步 → PPO 更新 → 保存
```

<a id="configuration"></a>

## 4. 任务与配置

### 4.1 全部 33 个任务

| 类别 | `--task` |
| --- | --- |
| 基础 13 项 | `velocity`, `velstand`, `standup`, `sitstand`, `groundpick`, `ballkick` |
| 基础续 | `rollers`, `swizzle`, `rollercrouch`, `rollerslope`, `rollerstandup`, `spin`, `roulade` |
| Rough 5 项 | `velocity_rough`, `velstand_rough`, `standup_rough`, `sitstand_rough`, `groundpick_rough` |
| Backlash 10 项 | `velocity_backlash`, `velstand_backlash`, `standup_backlash`, `sitstand_backlash`, `groundpick_backlash` |
| Backlash 续 | `ballkick_backlash`, `rollers_backlash`, `swizzle_backlash`, `rollercrouch_backlash`, `rollerslope_backlash` |
| Rough + Backlash 5 项 | `velocity_rough_backlash`, `velstand_rough_backlash`, `standup_rough_backlash` |
| Rough + Backlash 续 | `sitstand_rough_backlash`, `groundpick_rough_backlash` |

与本地官方注册一一对应；没有擅自添加 `spin_backlash` 或 `roulade_rough` 等未注册组合。
每次选择一个任务训练，不会自动训练一个融合所有技能的模型。

### 4.2 配置来源和优先级

| 配置来源 | 含义 |
| --- | --- |
| [RTX4090 模板](configs/rtx4090_synreal_official_adapted.json) | 新训练 CLI 默认值及 PPO 覆盖 |
| [任务快照](configs/official_snapshot.json) | 任务奖励、课程、观测、事件、终止、任务 PPO 和本地扩展 |
| `config.py` → `official_ppo.py` → 模板 `ppo` | 新训练 PPO 配置的覆盖顺序 |
| 每个运行目录的 `config.json` | 实际环境/PPO 配置、种子、任务、配方指纹及省略项 |

显式 CLI 参数覆盖模板 `training` 默认值。文件名含 `synreal` 不限制引擎，可用 `--solver mujoco` 覆盖。
`--run-name` 只是标签，不切任务，不启用自碰撞。普通 CLI 默认 solver 是 MuJoCo，模板默认是 SynReal。

2026-10-27 核对的模板值：

| 参数 | 值 |
| --- | --- |
| 后端 / 设备 / 种子 | `synreal` / `cuda:0` / `42` |
| 环境数 / 本次迭代数 | `1024` / `30000` |
| 每轮采样 / 保存间隔 | `24` / `300` |
| Actor / Critic 隐藏层 | `[512, 256, 128]`，ELU |
| 初始动作标准差 | `1.0`，scalar |
| 学习率 / entropy / desired KL | `0.001` / `0.01` / `0.01` |
| PPO epochs / mini-batches | `5` / `4` |
| viewer / 脚底摩擦随机化 | `null` / 开启 |

`config.py` 的保存间隔 500 是默认值，会被任务和模板覆盖。
模板也会覆盖某些任务的官方参数，例如 entropy 0.01 覆盖 Rollers 的官方 0.03。
`config.json` 不等于所有执行源码的完整存档；必须同时保留对应代码版本。

<a id="commands"></a>

## 5. 日常命令与依赖

所有本地共享入口在仓库根目录执行：

```bash
cd /home/ubuntu/work/style3d-sim
source .venv/bin/activate

# 先做目标后端的短测
python -u -m examples.microduck.train \
  --training-config examples/microduck/configs/rtx4090_synreal_official_adapted.json \
  --solver synreal --task velocity --num-envs 4 --iterations 2 \
  --run-name synreal_velocity_smoke

# SynReal 正式实验；需要明确启动长训练时再执行
python -u -m examples.microduck.train \
  --training-config examples/microduck/configs/rtx4090_synreal_official_adapted.json \
  --solver synreal --task velocity --run-name synreal_velocity

# MuJoCo 基线，使用同一套本地配方
python -u -m examples.microduck.train \
  --training-config examples/microduck/configs/rtx4090_synreal_official_adapted.json \
  --solver mujoco --task velocity --run-name mujoco_velocity
```

StandUp / VelStand / BallKick 分别换成 `--task standup` / `velstand` / `ballkick`，同时改运行标签。
本地 VelStand 当前不加载教师；命令能运行不代表流程已等价于官方。
CPU 诊断用 `--device cpu --num-envs 1`；MuJoCo CPU 后端不支持当前复制多世界模式。

```bash
# 本地 checkpoint 回放：替换为真实文件，同时保留旁边的 config.json
python -u -m examples.microduck.play \
  --solver mujoco --checkpoint /path/to/run/model_300.pt \
  --num-envs 1 --viewer gl --num-frames 3000 --command 0.2 0 0

# 同任务、同后端、同配方续训：显式核对物理开关
python -u -m examples.microduck.train \
  --solver mujoco --resume /path/to/run/model_300.pt \
  --num-envs 1024 --randomize-foot-friction --iterations 1000 --run-name continued

# TensorBoard
tensorboard --logdir logs/rsl_rl
```

回放不能把运行目录当 checkpoint；必须选择其中的 `model_*.pt`。
`--command` 三个量分别为前进、横移速度（m/s）与 yaw 角速度（rad/s）。
Velocity/VelStand 窗口中 I/K 前后、J/L 横移、U/O 转向，每次按键增减一档，松开不会自动归零。
需要确认零指令时用 `--command 0 0 0 --diagnostics`。

`--viewer null --benchmark --num-frames 300` 可做无窗口测速；`--full-step` 对照完整环境步。
跨后端回放需 `--allow-cross-backend`；只做旧版本诊断时才使用 `--allow-recipe-mismatch`。
正常效果评估应使用匹配配方，不应修改元数据或删除检查来绕过契约。

新机器依赖以仓库根 README 为准；本地曾使用：

```bash
uv sync --extra locomotion --inexact --no-install-project
```

该命令不安装本项目原生扩展；扩展安装遵循仓库根说明，不要无故重建现有工作环境。
Python 3.12 找不到 pip 安装的 CUDA runtime 库时，曾使用：

```bash
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"
```

SynReal 由 `common.py` 使用现有登录接口；环境变量缺少时隐藏输入。`--force-login` 会尝试替换已有会话。
官方独立环境缺失时，`mjlab_train.py` 提示使用 `uv sync --project examples/microduck/microduck_rl-develop --locked`。

以下是官方参考路径，不是共享 SynReal 训练：

```bash
python -m examples.microduck.mjlab_train --list-tasks
python -m examples.microduck.mjlab_train --num-envs 64 --iterations 5
python -m examples.microduck.pretrained --solver mujoco --policy velstand --keyboard
python -m examples.microduck.play --solver synreal --policy walk --num-frames 3000
```

`pretrained.py` 当前只支持 MuJoCo，默认 `--actuator metadata`；显式 `--actuator bam` 才选 BAM 分支。
它使用 `logs/pretrained/microduck/` 下已有 ONNX；原 `walk` 演示和本地 `.pt` 的动力学负载不可直接比较。

<a id="contracts"></a>

## 6. 控制、观测和物理契约

- 物理 dt=0.005 秒、每控制步 4 子步、策略 dt=0.02 秒（50 Hz）；奖励总和乘策略 dt。
- PD：KP=2.0、KD=0.053、力矩上限 0.96。动作是 HOME 的关节位置偏移。
  每子步计算裁剪后的命令力矩，再换算位置目标交给后端；不是直接使用官方 BAM 电机模型。
- 14 舵机顺序：左腿 0～4、颈/头 5～8、右腿 9～13。被动轮和 backlash DOF 保留在物理模型中，
  使用规范舵机视图处理策略动作/观测，不能按原始模型交错索引直接访问。
- Actor 固定 61 维：角速度 3、重力投影 3、关节位置偏差 14、关节速度 14、上次动作 14，
  然后 twist `48:51`、head `51:55`、body `55:61`。未使用的指令槽必须按任务 zero padding。
- Critic 按任务组织特权观测；velocity/velstand 70、groundpick 68、ballkick 74 维。
  两后端均明确省略官方双脚 6 维接触力，其他任务看运行配置，不泛化这四个数字。
- 训练保留配置中的噪声、编码器偏差、IMU 安装误差、延迟、缩放/截断；Critic 无噪声。
  非空 observation history/modifiers 未适配时会报错，不静默忽略。
- Newton 四元数主要用 xyzw，官方 facade 用 wxyz。Newton body 线速度是 COM 速度，
  奖励读取 root/site link-origin 速度时要补偿角速度与 COM 偏移；相关转换已有测试。
- Torch/Warp 交界显式同步；SynReal 重置或外部状态写入后要同步 affine body 状态。

后端区别：

| 项目 | MuJoCo | SynReal |
| --- | --- | --- |
| 构建 / 求解器 | `newton.ModelBuilder` / `SolverMuJoCo` | `s3d.ModelBuilder` / `s3d.NewtonSolver` |
| 积分 | `implicitfast` | `IMPLICIT_EULER` |
| 主要迭代设置 | Newton solver，20 次 | 增广拉格朗日；非线性 5、线性 30、alpha 200 |
| 接触提交 | 后端求解路径处理 | 每子步求解前显式碰撞 |

MuJoCo 容量当前设为 `njmax=256`、`nconmax=4096`；倒地姿态曾超过站姿估计的碰撞候选容量。
SynReal 路径没有自动回退 MuJoCo。共享 `newton.eval_fk/eval_ik`、碰撞或 Viewer 不意味着偷偷使用 MuJoCo。
共用公式不保证同动作下产生同状态、同奖励或同训练结果。

自碰撞默认关闭，显式 `--self-collisions` 才开启；这也可能改变资产选择。
当前 `policy_spec.py`：轮式用 rollers 资产，VelStand/BallKick 或开启自碰撞时选 allcollisions，
普通 Velocity 用 walk，其余用 groundcontact，Backlash 用相应 twin。网格转凸包代理。
地面、球体碰撞保留；自碰撞关闭时其奖励虽然有注册，却没有有效自碰撞信号。

脚底摩擦随机化由模板开启，启动时每环境采样 `[0.7,1.3]`，同环境双脚共享。
不使用模板且不传开关时固定 1.0。分环境材质会影响 SynReal shared-world 优化与吞吐。

<a id="alignment"></a>

## 7. 已完成移植与仍有差异

### 7.1 已完成

- 默认入口统一到 `official_adapted_v1`；旧 `legacy`、旧简化奖励/观测/训练分支已删除。
- 官方注册全部 33 任务接入；增加轮式/齿隙资产、被动关节映射与相应观测。
- 从本地官方环境导出最终配置，提取奖励/重置公式并有 AST 来源检查；适配基础状态与奖励汇总。
- Actor/Critic、延迟/偏差、指令、重置姿态、终止、推扰及多数课程已接入。
- SitStand 二值指令及约 2 秒过渡、GroundPick 周期相位、相对航向、前进/原地转向混合采样已实现。
- 脚部/头部/全身几何接触、腾空/触地计时、摆动峰值和全身角动量已接入。
- Rough / Slope 几何、出生原点和地形高度查询已接入；Roulade 接触门控及镜像损失已适配。
- 修正 RollerStandUp 官方旧轮式索引与规范 14 舵机视图不匹配的问题，修正在本地适配层。
- 本地 checkpoint 回放、跨后端检查、同任务续训和课程恢复已接入。
- 数值异常保护、SynReal 隐式欧拉稳定性修复，以及本地 Velocity 步态奖励已接入；效果见第 11 节。

### 7.2 未完成或近似项

| 项目 | 当前边界 |
| --- | --- |
| BAM | 使用共享 PD；电机电压模型、摩擦、动作延迟等没有完整对齐官方 |
| 力矩相关奖励 | 使用末子步的命令 PD 力矩，不是官方 BAM 实际输出的完全等价量 |
| 接触力/冲量 | 几何接触不是求解后力；不能用候选数冒充接触力 |
| 力/冲击奖励 | VelStand 的 servo/head/trunk impact 等仍省略 |
| GroundPick | soft_landing、head_impact_penalty 未完整接入；mouth_payload_force 外力钩子未接入 |
| 物理随机化 | 质量/惯量、COM、armature、BAM/轮轴摩擦及轮轴摩擦课程仍未完整适配 |
| 地形 | 官方生成器 seed=42 的固定样本，非每次重新生成；高度射线只检测地形 |
| 教师训练 | 没有官方 `PpoWithExpertBc`，没有双教师、跨任务 warm-start、教师归一化/指令映射 |
| 物理与算法版本 | 接触代理、求解器设置、积分器及 RSL-RL/mjlab 版本有差异 |
| 部署 | 本地元数据 `deployment_compatible=False`，不能据此宣称可直接上真机 |

`mouth_payload_force` 是施加负载的流程，官方快照权重为零；不要把其未接入解释为缺少力读取接口。
按运行元数据 `official_alignment.supported_rewards / omitted_rewards / approximate_rewards` 判定覆盖。
静态快照中有字段或计数包含零权重项，都不等于它在该次训练有效。

BAM 是否值得移植：PD 能训练出稳定步态，目前优先建立可靠基线；严格官方复现和 sim-to-real 时再提高优先级。
可以先研究环境子步里的执行器适配，尚未完成 SynReal 力矩/摩擦接口审计，不能承诺完全不需要底层改动。
不要把蹦跳或起身失败未经验证地归因于 PD，也不要声称换 BAM 必然改善。

<a id="velocity"></a>

## 8. Velocity 奖励、课程与最新步态修改

### 8.1 官方基础项

| 奖励 | 基础权重 / 课程 |
| --- | --- |
| 线速度、角速度跟踪 | 各 +2 |
| upright / pose | +2 / +1 |
| body_ang_vel / angular_momentum | -0.05 / -0.02 |
| dof_pos_limits | -1 |
| action_rate_l2 | -0.1，后期课程到 -1（约 1500 轮） |
| air_time | +3，每只脚独立计入，腾空时长区间 0.125～0.300 秒 |
| foot_clearance / foot_swing_height | -2 / -0.25，目标 0.02 m |
| foot_slip / self_collisions | -0.1 / -1；后者依赖实际自碰撞开关 |
| head_pose_tracking | +2 |
| body_pose_tracking | 0 |
| head_pose_bias | 0 → +1 → +2 → +3；函数自身返回惩罚，不能只看权重符号 |

body-frame 线速度跟踪公式本身也包含 Z 速度误差；新增项使用 world-frame Z。
upright 限制倾斜，不直接要求身体高度不变或左右脚交替。
`air_time`、净空、摆动高度、脚滑都有指令门控，真正零 twist 时均为零。
因此零指令踏步不能直接认定为“刷腾空奖励”；身体平稳也不能证明脚不在动。

速度范围 vx `[-0.4,0.4]`、vy `[-0.3,0.3]` m/s、yaw `[-1,1]` rad/s，重采样 3～8 秒。
纯前进采样参数 20%；原地转向覆盖概率 Velocity 15%、VelStand 20%，会清除 standing 标记。

下表保留官方/旧本地课程，当前精确 `velocity` 已采用第 8.3 节的本地课程。

| 控制步阈值 | 约 PPO 轮数（rollout=24） | 官方/旧本地 standing 参数 |
| --- | --- | --- |
| 0 | 0 | 2% |
| 12000 | 500 | 5% |
| 18000 | 750 | 10% |
| 24000 | 1000 | 15% |
| 36000 | 1500 | 20% |
| 48000 | 2000 | 25% |

指令/奖励阶段通常用严格 `>`，事件/终止阶段用 `>=`，精确边界查实现。
表中参数不是最终零指令占比：旧课程约为 standing 参数 × 0.85，例如 1200 轮约 12.75%，后期约 21.25%。
不是把前进/转向/站立比例直接相加，也不等于按 episode 长度加权后的状态占比。

### 8.2 本地奖励（当前有效配置）

只对精确任务名 `velocity` 生效；不自动扩展到 Rough、Backlash、VelStand 或其他恢复任务。
通过共享 `official_recipe.py` 计算；髋侧摆和目标超限两项复用 `official_reward_functions.py` 中已有官方函数。
权重/参数统一放在快照的 `tasks.velocity.rewards`，标记 `local: true`。

| 奖励 | 当前权重 | 定义与门控 |
| --- | --- | --- |
| `vertical_body_velocity` | **-1.0** | trunk_base 世界系 link-origin 竖直速度平方 |
| `standing_foot_motion` | **-1.5** | 双脚 world xyz 速度平方和；平移指令范数及 yaw 绝对值分别 ≤0.01，倾角 ≤35° |
| `standing_foot_contact` | **-0.5** | 在与静止脚速项相同门控下，未与地面几何接触的脚数（0、1、2） |
| `straight_foot_heading` | **-1.5** | 双脚相对机身 yaw 超过 5° 容差后的弧度误差平方和 |
| `hip_roll_neutral` | **-1.0** | 左右 hip_roll 相对 HOME 的绝对偏差之和（弧度），只选规范舵机索引 1/10 |
| `action_over_limit` | **-0.5** | 14 舵机目标角超出硬限位加 0.3 rad 余量的绝对超限量之和 |

脚尖朝向项还要求 yaw 指令绝对值 ≤0.1 rad/s、横移指令绝对值 ≤0.05 m/s、倾角 ≤35°。
它以机身朝向为参考，不固定世界朝向；在当前 Velocity 资产 HOME 中，两脚 site +X 与 trunk +X 对齐。
静止项只读取脚部，不惩罚头部运动；大倾角时关闭以允许恢复平衡。
静止接触项逐脚判断 `found > 0`，不把一只脚的多个接触点当成双脚落地。
它补充脚速项无法区分“静止落地”和“静止悬空”的缺口；一只未落地脚每控制步扣 0.01（权重 -0.5 × dt 0.02）。
它不约束有移动/转向指令时的支撑相位；尚未增加左右脚交替奖励，不能声称强制自然交替步态。
接触为几何检测结果，不是接触力。新项验证了触发及门控，尚未证明新策略训练收敛后能完全停稳。

髋侧摆项约束实际角度；目标超限项约束 `HOME + action`，允许低增益 PD 的正常超调余量。
它们不会裁剪动作或修改物理限位，也没有改变执行器或引擎；行走和转向时同样计算。
髋侧摆项单独短训未纠正旧策略顶限位，不能把它单独宣传为已验证的解决方案，详见第 11.3 节。

各项贡献都乘 `step_dt=0.02`；`Task/<reward>` 日志是乘 dt 前的加权值。
`official_alignment.local_rewards` 会记录这些配置；导出器保留已有 `local: true` 奖励及其调参。

版本提醒：最初新增静止/朝向两项时权重是 **-2.0**，其后当前文件已调为 **-1.5**。
`velocity_still_heading` 的运行配置也保存了 -1.5。不要用早期聊天或短测记录覆盖当前值。
相关 28 项测试和四环境两轮 PPO 短测是初始 -2.0 版本的验证；未据此宣称 -1.5 长训练已收敛。

### 8.3 当前本地站立采样课程

用户在 `velocity_stand_contact/model_1500.pt` 反馈内八已改善、零指令仍踏步。
实测静止奖励已触发；目前没有证据表明继续增加惩罚就能解决。此次仅增加站立命令的训练覆盖，六项奖励权重不变。
只覆盖精确 `velocity`，配置位于 `tasks.velocity.curricula.standing_envs`，标记 `local: true`。

| 控制步阈值 | 约 PPO 轮数 | standing 参数 | 经原地转向覆盖后的零 twist 概率 |
| --- | ---: | ---: | ---: |
| 0 | 0 | 20% | 17% |
| 12000 | 500 | 25% | 21.25% |
| 18000 | 750 | 25% | 21.25% |
| 24000 | 1000 | 30% | 25.5% |
| 36000 | 1500 | 30% | 25.5% |
| 48000 | 2000 | 35% | 29.75% |

阶段按 `>` 选取；原地转向仍优先覆盖约 15% 样本，速度范围、命令重采样时长和头部课程未改。
此概率指一次命令重采样的期望结果，不是按时间加权的精确训练数据占比，也不是专门保证停走转换次数的课程。
提高站立样本比例是待训练验证的改进假设，不是已证明踏步根因仅为样本不足。
`official_alignment.local_curricula` 记录实际本地课程，导出器与本地奖励一样保留其 `local: true` 配置。
运行中的旧进程不会热更新；旧 checkpoint 的课程仍以对应运行的代码版本为准。

<a id="velstand"></a>

## 9. VelStand 流程、教师与已知训练问题

VelStand 是一个同时行走、站立、姿态控制、倒地起身并继续执行指令的策略，不是运行时切换两个策略。

官方主要路线是：Walking 权重热启动 → 带恢复课程的 PPO → PPO 更新后做双教师 BC。
热启动保留策略、归一化和优化器，重新开始 VelStand 课程计数；官方后续版本也使用普通续训保持课程最终阶段。
依据：[官方训练谱系](microduck_rl-develop/docs/velstand_policy.md) 与
[官方 distill.py](microduck_rl-develop/src/mjlab_microduck/tasks/distill.py)。

无特殊 body 指令时，按 Actor 重力投影计算倾角 θ：

| 条件 | 官方 BC 教师 |
| --- | --- |
| θ ≤25° | Walking anchor |
| 25° < θ ≤35° | 无教师，仅 PPO |
| θ >35° | StandUp expert |

零 twist 且有 body 指令时，退出 Walking 约束；仅倾斜指令交给 StandUp，高度变化由 PPO 学习。
已明显倒地时仍按 StandUp 恢复处理。教师处理采样状态，环境动作始终由学生输出，教师冻结。
给 StandUp 的输入副本将 twist 清零；严重倾倒 >75° 时官方还清零 `6:48` 的关节/上次动作槽。
学生真实输入不变；关节位置偏差清零表示 HOME，不是把仿真关节实际归零。
Walking 教师的 yaw 指令映射表针对官方教师标定，自训教师不能直接照搬。

本地 `official_ppo.py` 目前保持普通 PPO，未加载 BC 或教师。
新开 VelStand 是随机初始化，但快照仍为官方 `WARM_START=True` 课程，这是已确认的不等价点。
训练好 velocity / standup 只是准备；还需跨任务初始化、各教师归一化、BC、路由和输入处理，不能仅填写路径。
当前 `--resume` 拒绝跨任务/跨后端加载，不要偷偷放宽检查充当 warm-start。

本地课程（每轮 24 步，近似轮数）：

| 轮数附近 | 变化 |
| --- | --- |
| 开始 | 动作变化惩罚 -1，standing 参数 25%；常规直立出生外已有 5% 坐姿 |
| 100 | 取消倾角超过 70° 的立即倒地终止，推扰增强 |
| 150 | 20% 蹲姿出生 |
| 300 | body 指令范围扩大 |
| 400 | 推扰再次增强 |
| 600 | 开启倒地惩罚、恢复成功奖励和倒地向上运动奖励 |
| 700 | 倒地出生 15% |
| 800 | topple push 水平各轴速度增量最大到 ±1.2 m/s |
| 1000 | 倒地出生 30% |
| 1400 | 倒地出生 45% |

后期出生约 45% 倒地、20% 蹲姿、5% 坐姿、30% 常规直立，倒地内含侧躺/随机姿态/关节变体。
连续倒地判定满 8 秒会重置，每局最多 20 秒。BallKick 则是 5 秒；不可把 Velocity/VelStand 时间泛化到全部任务。
若要无教师从零训练，需要另外设计并验证课程；延长现有预算不能保证解决失败。

<a id="checkpoints"></a>

## 10. 回放、续训与版本陷阱

- `recipe_fingerprint()` 哈希配方、函数、环境、资产适配和地形文件；甚至这些文件的注释改动也会影响它。
  2026-10-27 步态优化新增奖励后，旧训练的指纹不再与当前配方匹配。
- 当前六项本地奖励 + 站立课程指纹为 `06c466c95e0946d2dc349a3d9e5c7f711d621266a3cd323623865def8418011b`。
  六项奖励、旧站立课程版为 `86d3281b54df70888bfb9ad5b5b2cf6269107e82a6bc7fab7cb4e808ede31187`。
  前一轮五项版指纹为 `f2a11cc1648d34f2e6f640b92e1c9688542999e213f97a106824e286137d9c67`。
  它是时间快照，不要把这个值手动写入旧运行配置。
- 保留 checkpoint 旁的 `config.json`。加载检查任务、后端、观测契约和指纹。
- `--resume` 恢复 PPO/优化器/归一化与课程计数，不恢复完整 RNG、仿真状态或启动随机化样本。
  `--iterations` 是本次追加轮数，不是目标累计轮数。`model_0.pt` 已完成一次 PPO 更新，不是纯随机权重。
- 续训的 PPO 配置来自保存值；环境 CLI 开关并非全从元数据恢复。
  自碰撞、摩擦随机化、环境数等必须显式核对；不要声称恢复了整个环境。
- CLI 回放不宜加载含 1024 环境的新训练模板；指定单环境，并核对 checkpoint 的物理开关。
  上面的通用回放命令未启用摩擦随机化；严格复现实验时应补上保存配置中启用的开关。
- `play.py` 已修复每帧写入指定 twist，避免 reset/自动重采样覆盖；实际观测指令仍须诊断验证。
- 回放关闭逐步 Actor 随机噪声，但保留偏差与延迟。
  对启用头指令的任务使用固定 HEAD_UP_COMMAND，对 zero-padding 槽保持零，BallKick 不强行抬头。
- `PolicyPlayback` 省略训练奖励、Critic 和奖励专用求解后传感器更新；当前快速回放的随机推扰调用被注释。
  `--full-step` 走完整步进，仍受 `play.py` 的噪声/头部指令设置影响，不能等同原训练条件。
  快速专用物理路径主要覆盖 velocity/velstand/groundpick/ballkick，其他任务回退完整 `env.step()`。
- 当前共享 `.pt` 配方无 CUDA graph；`--disable-cuda-graph` 主要影响原 walk 演示。
- CUDA 保存的 checkpoint 做 CPU 诊断时，当前 RSL-RL `load()` 未显式设置 `map_location`；历史诊断曾临时适配。
  不要把 CPU 环境可步进等同于 CLI 已支持任意 CUDA checkpoint 的 CPU 加载。
- 同一 Viewer 不说明同一引擎；相同代码、种子也不保证独立接触轨迹逐位一致。

<a id="next"></a>

## 11. 近期实验与下一步工作

### 11.1 已确认的步态基线

运行：`logs/rsl_rl/microduck_mujoco/2026-10-27_10-26-04_633842_velocity_z/`。
诊断对象是 **model_1200.pt**，不是交接时后来出现的最新文件。
诊断时指纹 `ce48ad4c95db82afbd4c2241e83384804e0a88ffd8c5c28de2066700166df72a` 与当时代码一致，只有 Z 惩罚。

MuJoCo GPU 四环境、每场景 400 步、前 100 步排除；固定指令逐帧写入 Actor，标准快速回放、无推扰。
三场景都没有重置。这不是停走切换测试，也不是当前新增奖励的训练效果评估。

| 场景 | 平均前进速度 | 双脚同时几何接触 | 左 / 右脚相对机身 yaw |
| --- | --- | --- | --- |
| 零 twist | -0.00547 m/s | 2.33% | -9.31° / -0.65° |
| vx=0.2 m/s | 0.09568 m/s | 2.08% | -9.95° / +3.49° |
| yaw=0.5 rad/s | -0.01165 m/s | 2.25% | -7.94° / -2.88° |

零指令下脚部仍运动，双脚各约 51% 采样有接触，复现原地踏步；不是命令覆盖导致。
HOME 朝向检查排除了 site 固定旋转误差；前进时脚尖朝向与内八反馈一致。
这些接触是几何标志，不是接触力测量。脚尖朝向不足以单独解释所有运动学根因。

证据在仓库根 `.validation/microduck-gait/`：
`REPORT.md`、`results.json`、`diagnose.py`、`diagnose.log`、测试日志、`train-smoke.log` 和短测 checkpoint。
原诊断脚本断言旧指纹，不能直接拿当前奖励版运行并称为原版本重现。

### 11.2 交接时发现的运行文件

以下只说明文件检查结果，不保证进程此刻仍在运行，也不是收敛结论：

| 日志根下的运行目录 | 当次看到的最新 checkpoint | 配方关系 |
| --- | --- | --- |
| `microduck_mujoco/2026-10-27_10-26-04_633842_velocity_z` | model_2100.pt | 仅 Z 惩罚，与当前指纹不同 |
| `microduck_mujoco/2026-10-27_11-09-59_978957_velocity_still_heading` | model_600.pt | 三项本地奖励，后两项 -1.5，与当前六项版不同 |
| `microduck_synreal/2026-10-27_09-54-10_508977_synreal_velocity` | model_1200.pt | 元数据未记录本地三项奖励，与当前指纹不同 |

根目录是 `logs/rsl_rl/`；下一位 AI 应重新检查 checkpoint 和进程，不假定上表仍是最新。

### 11.3 优先待办

后续用户仍反馈轻微踏步和内八，已对 `velocity_still_heading/model_1800.pt` 做匹配配方的
MuJoCo GPU 四环境诊断（截图未指定具体 checkpoint，选择的是诊断开始时最新的 1800）：
零 Actor twist 确认，双脚同时几何接触约 4%；脚尖 yaw 约 -7.39°/+3.79°。
更明显的是脚部横向间距仅 3.91 cm（HOME 8.36 cm），hip_roll 约 -22.56°/+22.55°，
已贴近并略越过名义硬限位 ±22°（soft limit ±19.8°），HOME 仅 ±5°。
零指令加权 foot-motion≈-0.0538、heading≈-0.00594，整体 pose≈1.7e-6；惩罚并非未注册或未触发。
前进 0.2 m/s 实测约 0.101 m/s。更应先研究髋侧摆/窄站姿，而不是盲目增大脚尖或 Z 惩罚。
完整记录：`.validation/microduck-gait/STILL_HEADING_1800.md` 及对应 JSON/脚本。
上述首次诊断没有改奖励；后续用户明确授权“你来优化”，进行了以下实验。

**2026-10-27 实施与受控短训：**

- 新增 `hip_roll_neutral=-1`，复用官方 `joint_deviation_l1`；单独试验与原奖励对照各追加 200 轮。
- 两组都从同一个 `model_1800.pt` 恢复网络、优化器和归一化，延续课程；各 256 环境、24 步/轮，
  即 1,228,800 样本。不是从零长训，也不是不同后端对比。
- 六场景固定指令回放显示，单独髋惩罚组仍卡在 ±22.56°，零指令站距约 3.49 cm、双脚接触 3.7%。
  原奖励对照约 3.70 cm / 3.5%，不能声称该项单独解决内八或踏步。
- 新测到旧策略零指令髋目标角约 **-108.4°/+157.2°**，远超实际硬限位 ±22°。
  改变策略输出后仍可能得到相同饱和力矩和限位姿态，因此仅惩罚实际角度可能难以退出这一行为。
- 据此新增 `action_over_limit=-0.5`，复用官方目标超限函数，保留 0.3 rad 的 PD 余量。
  这与只看实际角度的 `dof_pos_limits` 作用不同。原有 Z/静止/脚尖权重保持不变。
- 实验脚本在 `.validation/microduck-gait/hip-roll-trial/`，记录原/新快照、实际指纹、父 checkpoint 哈希和配置。
  这是隔离的奖励迁移实验，普通 `--resume` 检查没有放宽，原运行目录/用户训练进程没有被替换。
- 本轮相关 30 项测试通过，含 MuJoCo CPU/GPU 注册、日志及重置。尚未完成 SynReal 行为验证。

**髋侧摆/目标超限实验的结论：问题尚未解决，两项约束仍待从零训练验证。**

同父 checkpoint、同 200 轮预算，固定零指令回放结果如下：

| 配方 | 站距 cm | 双脚同时几何接触 | 双脚速度平方和 | 前进 0.2 指令的实测 vx |
| --- | ---: | ---: | ---: | ---: |
| 原奖励继续训练 | 3.70 | 3.50% | 0.02943 | 0.126 m/s |
| 只加髋侧摆 L1 | 3.49 | 3.67% | 0.02784 | 0.117 m/s |
| 髋侧摆 L1 + 目标超限 | 4.30 | 2.00% | 0.04163 | 0.099 m/s |

组合组零指令髋目标角收回到约 -39.9°/+35.3°，但实际髋角仍约 -21.95°/+22.19°；
过大目标改善，不能等同于站姿纠正。脚部运动增加、前进跟踪低于等预算对照，不能把该短训称为步态改善。

随后从组合组同一个 `model_2000.pt` 出发，各用 256 环境追加 100 轮，单独比较静止脚部权重 -1.5 / -6：
零指令双脚速度平方和约 0.04066 / 0.04008，改善不足 2%；yaw 漂移从 -0.105 增至 -0.400 rad/s。
转向指令 +0.5 时两组实际 yaw 为 +0.024 / -0.254 rad/s，均未良好跟踪，后者还反向。
因此 **-6 已回退**，当前仍为 -1.5；该轮未增加双脚着地项。
这些微调 checkpoint 只用于诊断，不推荐作为已修复策略演示。

当前保留 `hip_roll_neutral=-1`、`action_over_limit=-0.5`，理由是约束已测到的限位利用行为；
这不构成两项会改善从零训练的证据。下一步应在新配方下从零学习，并与原配方从零训练按相同预算对照；
不要继续仅用已饱和旧策略的很短微调结果判断最终效果。需要检查是否重新顶限位、是否停稳及是否损害指令跟踪。
普通 `--resume` 仍拒绝旧指纹；源配置改动不会自动改变旧 checkpoint 的动作。

评估使用 4 环境、固定 HEAD_UP、保留偏差/延迟、关闭逐步观测噪声及推扰；区分超时和非超时重置。
上述全部六场景回放都没有重置；短训仅单种子，部分时间与用户训练/其他实验共享 GPU，
不能据此认定长期收敛或真实机器人效果。详细结果见该目录 `REPORT.md` 和 `eval-*.json`。
`.validation/` 是本机忽略目录，对外复现需单独归档脚本、快照、日志和 checkpoint；关键结论已保留在本文。

**随后针对零指令踏步补充静止接触目标（2026-10-27）：**

用户继续要求修复原地踏步。现有脚速平方项对完全悬空但不动的脚返回零；
此前也已测到踏步时双脚同时接触仅约 2%～5%，而单纯增大脚速权重无效并损害转向。
本轮仅增加 `standing_foot_contact=-0.5`，没有继续增大脚速权重或改变课程/引擎。
当真实 twist 的平移范数 ≤0.01、yaw 绝对值 ≤0.01、倾角 ≤35° 时，每只未落地脚计一个单位成本。
头部指令不关闭站立约束，移动/转向/大倾角则关闭，允许行走、转弯和恢复平衡。

验证结果：

- MuJoCo GPU 四环境、HOME 零动作保持 400 步，排除前 100 步：双脚接触率 **100%**、无重置，
  双脚速度平方和约 2.60e-8。该姿态在当前 PD 下可以静止；这不是训练策略的成功率。
- 每只落地脚常有约 4～5 个几何匹配，因此使用每脚布尔接触，不能对原始匹配总数截断来判断双脚落地。
- 对不变的旧 `model_1800.pt` 做显式允许配方不匹配的只读诊断，每场景 300 步、排除前 100 步：
  零指令双脚接触率 5%，旧脚速加权项约 -0.0573、新接触加权项 **-0.475**；前进 0.2 和转向 0.5 时两项均为 0。
  三场景无重置。此结果证明新奖励会处罚已观测的踏步行为，不代表旧策略已改善。
- **31 项测试通过**，包含左右脚独立计数、静止悬空、双脚离地、头部/平移/侧移/转向/倾角门控、任务范围，
  以及 MuJoCo CPU/GPU 注册、日志、重置检查。
- 新配方 MuJoCo 64 环境、5 轮 PPO（7,680 样本）启动、训练、保存完成；这仅是功能短测。
  没有运行从零长训练，也没有本轮 SynReal 行为验证，不宣称已经学会停稳。

证据：`.validation/microduck-gait/stand-contact/` 内的 `home_probe.json`、`policy_probe.json`、`tests.log`、
`train_smoke.log`；保存了修改前的 `baseline_recipe.py` 和 `baseline_snapshot.json`。
旧实验脚本的指纹断言在当前源码下可能不再通过，应使用归档源码做隔离复现，不修改旧运行元数据绕过检查。
下一轮从零训练可用：

```bash
python -u -m examples.microduck.train \
  --training-config examples/microduck/configs/rtx4090_synreal_official_adapted.json \
  --solver mujoco --task velocity --run-name velocity_standing_contact
```

新奖励在 MuJoCo/SynReal 共用配方中，仅对精确 `velocity` 启用。旧模型不会因改奖励自动停止踏步。

**`velocity_stand_contact/model_1500.pt` 的后续核查与站立课程修改：**

用户明确提供该 checkpoint 的 MuJoCo 回放命令（单环境、摩擦随机化、`--allow-recipe-mismatch`）。
核查时源码与模型指纹均为 `86d328...`，允许不匹配开关不是本次问题的原因。
四环境 GPU 全步进诊断与当前回放使用相同 PD/观测配置，关闭逐步 Actor 噪声和推扰、保留偏差/延迟，
固定真实 Actor twist 逐帧写入。每场景 300 控制步，排除前 100 步，头部使用回放默认 HEAD_UP。
与 Viewer 快速步进不是逐轨迹一致的比较；未据此宣称完全重现单环境截图。

| 场景 | 双脚同时几何接触 | 双脚速度平方和 | 平均 vx | 非超时重置 |
| --- | ---: | ---: | ---: | ---: |
| 出生后零指令 | 28.88% | 0.03035 | -0.00085 m/s | 0 |
| 前进指令 0.2 | 3.75% | 0.08960 | 0.10685 m/s | 0 |
| 前进后停止 | 4.63% | 0.03972 | +0.00257 m/s | 0 |

- 站距约 8.7 cm、hip_roll 约 -4.0°/+3.75°，改善与用户内八反馈一致。
- 出生零指令场景四环境的双脚接触率分别约 6%、100%、5%、4.5%；只能说明一个环境在这段窗口能持续落脚。
  前进后停止时四环境都继续踏步，每只脚约每秒 2.3 次新触地；平移速度接近零不等于脚静止。
- 真正零 twist 下 `air_time=0`。出生静止/接触加权项约 -0.0455/-0.3556；停走后约 -0.0596/-0.4769，
  两项均已生效，不能再说接触奖励漏接或踏步在刷腾空奖励。
- 额外比较头部零指令：出生零 twist 的双脚接触率约 32.88%，脚速平方和约 0.03015，仍有踏步。
  只完成该场景，随后停止额外 GPU 诊断以避免争用；没有完整的中立头部停走测试结论。
- 仅诊断中强制腿部动作归零、头部继续由策略控制，出现 **21 次非超时重置**。
  不能把此前“所有关节 HOME 可静止”的结果推广到“抬头策略也能靠冻结双腿稳定站立”；本轮没有加入动作冻结。
- 旧课程在最初 500 轮只有约 1.7% 的零 twist 重采样机会；1500 轮附近才提高到约 17%。
  当前假设是站立及停走经验不足以形成稳定切换，故采用第 8.3 节课程；没有声称该假设已被因果实验验证。

本轮 **35 项测试通过、1 项 GPU 测试跳过**，包括 4 个阶段各 20,000 次真实 CPU 指令采样，
核对零命令概率、原地转向概率、任务范围和导出保留；CPU 单环境 2 PPO 轮/48 样本训练及保存通过。
没有追加 GPU 训练，没有修改用户运行中的训练或另一个 SynReal 进程。新课程的训练效果尚未验证。
证据：`.validation/microduck-gait/stand-contact-1500/`，含完整诊断、头部比较、修改前源码/快照和测试/CPU 短测日志。
旧实验脚本断言旧指纹，需使用对应源码复现。

后续评估优先看同一运行的 2000/3000 及之后 checkpoint；1500 轮附近刚发生动作平滑、头部和站立课程变化。
这些轮数是比较节点，不是保证停稳的收敛阈值。新站立课程只对新启动训练生效，普通 `--resume` 仍禁止跨指纹。

1. 验证新约束从零训练的结果：零指令、行走后停止、重新起步、转向、侧移。
   记录真实 Actor 指令、脚部速度/接触、机身 vz、足部 yaw、速度误差及重置原因。
2. 检查新约束是否导致僵硬、单脚悬停或速度跟踪进一步变差；权重 -1.5 不是已验证最优值。
   当前静止接触项已实现，但仍需新训练对照；特别检查是否出现双脚滑动、单脚长期悬停或失去起步能力。
3. 在相同任务/配置/采样量/物理开关下比较 MuJoCo 与 SynReal；当前两条长期实验指纹不同，不能直接归因引擎。
4. 对 StandUp、BallKick、VelStand 分开评估任务成功。高奖励或能站着不证明会起身/踢球。
5. 用户决定采用官方教师路线时，再做显式 warm-start/双教师接口；或者独立设计无教师课程。
6. BAM、物理随机化、接触力及 Rough/Backlash 全面动力学验证属于进一步官方对齐工作，尚未完成。

<a id="validation"></a>

## 12. 验证方法与交付标准

日常检查：

```bash
WARP_CACHE_PATH=/tmp/microduck-warp-cache .venv/bin/python -m pytest -q examples/microduck/tests
git diff --check
```

最新步态修改对应的定向测试：

```bash
MICRODUCK_GPU_TESTS=1 WARP_CACHE_PATH=/tmp/microduck-warp-cache \
  .venv/bin/python -m pytest -q \
  examples/microduck/tests/test_gait_rewards.py \
  examples/microduck/tests/test_vertical_reward.py \
  examples/microduck/tests/test_official_recipe.py \
  examples/microduck/tests/test_play_commands.py
```

- `test_gait_rewards.py`：零指令脚部 xyz 运动、头部不误罚、转向/侧移/大倾角门控、yaw 跨 ±π、HOME 轴、注册日志/重置；
  髋侧摆关节选择和大偏差纠正、物理顶限位时策略目标纠正仍有奖励差异、PD 超调余量和任务作用范围；
  静止接触项区分落地/悬空，单脚多个匹配不能替代双脚接触。
  本地站立课程的实际指令分布、转向保留、元数据与导出保留也在该文件验证。
- `test_vertical_reward.py`：竖直速度正反对称、水平不误罚、权重/dt、任务范围和真实 CPU 环境。
- `test_play_commands.py`：命令持续写入、zero padding。
- 官方配方/观测/传感器/任务变体测试：来源 AST、关节映射、局部重置和课程边界等。
- `test_synreal_stability.py`：需要 CUDA、有效登录及 `MICRODUCK_GPU_TESTS=1`；运行前查看条件。
  不要为普通文档修改启动 SynReal 登录或完整 GPU 测试。

真实变更验证至少按影响覆盖动作响应、有限状态、重置、奖励/观测/事件；GPU 短测先小环境少迭代。
报告时区分：代码已实现、单元测试通过、物理/PPO 短测通过、长期收敛验证。
不把 CPU 测试说成 SynReal GPU 动力学验证，也不把过去的测试计数当成本轮执行结果。

`Task/*` 是加权项（未乘 dt）；`Physics/fall_fraction` 是采样状态比例，不是 episode 摔倒率。
`Physics/reset_count` 为累计重置数，rollout 日志平均后可为小数，包含超时；不是该轮新增失败次数。
`recovery_success` 是事件奖励的均值，不是回合恢复成功率。
训练 `steps/s` 是全部环境合计吞吐，不是窗口 FPS；GPU 并行负载、材质、碰撞和编译都会影响耗时。

<a id="history"></a>

## 13. 历史问题与证据边界

### 13.1 SynReal 数值故障与扩展版本

2026-10-21 的 VelStand 曾在约 240 轮发生巨大有限数值，随后价值损失 inf。
新增 `stability.py` 在每物理子步检查位置分量 >1000 m、线速度分量 >100 m/s、角速度分量 >1000 rad/s 或非有限值，
保存 `logs/microduck_diagnostics/*_unstable.pt` 后报错。阈值是诊断保护，不是奖励裁剪或官方任务终止。

故障前状态回放中，旧 Midpoint-Cayley 在第 2 物理步复现发散；脚踝单步约 176.5° 旋转后的反解角速度被放大。
改为 implicit Euler 后同状态连续 100 步通过，保留异常保护。不能由此推断更早的全部故障来源或长期稳定性。
证据：`.validation/microduck-stability/REPORT.md`。

2026-10-27 曾报 `style3d_sim._core.Model` 无 `cable_count`：根 `.venv` 旧原生扩展与当前 Python 源不匹配。
当时执行 `uv pip install --python .venv/bin/python -e . --no-deps --offline`，核对二进制哈希和字段可写。
这是已做过的环境修复记录，不要求每次接手都重装；该次检查没有完成 SynReal GPU 步进验证。

早期会话中的“无 CUDA 驱动”是当时环境限制；后续步态诊断已实际使用 RTX4090 / MuJoCo GPU。
不要复制成当前机器永久没有 GPU，也不要反过来把后续 MuJoCo 验证算作 SynReal 验证。

### 13.2 训练失败与回放修复

- `microduck_mujoco/2026-10-26_23-17-43_618423_mujoco_velstand`：历史分析末 100 轮
  fall_fraction≈0.986、upright≈-0.013；100 轮附近取消立即倒地终止后退化。
  随机初始化、无 BC 却使用热启动课程是重要不等价证据，不是已排除全部动力学原因。
- `microduck_mujoco/2026-10-26_23-27-31_611798_mujoco_ballkick`：历史日志到 3430 轮，最新当时为 model_3300.pt。
  末 100 轮 ball_forward_velocity 加权项≈11.40，fall_fraction≈0.922；不能用高总奖励证明站立踢球成功。
- BallKick 回放曾把训练为零的头部槽强制抬头，归一化后输入幅度约 34.9；已修复 zero padding。
  Velocity/VelStand 指令被 reset/采样覆盖的问题也已修复为逐帧写入。
- 早期 SynReal Velocity 约 21000 轮 episode≈999.2/1000、upright≈0.9987；MuJoCo 约 1200 轮 episode≈1000。
  两实验指纹、摩擦设置不同，不能由此直接比较引擎；这些数值也不能证明步态自然。

### 13.3 验证历史归档摘要

以下归并原 VALIDATION/OFFICIAL_PORT_STATUS 的记录，均标记版本，不重新宣称本轮已执行：

| 阶段 | 当时验证 | 证据 / 限制 |
| --- | --- | --- |
| 2026-10-20 旧简化版 | 两后端 2 环境、2 PPO 更新、96 样本、50 帧回放；MuJoCo 续训 | `restructure_smoke` 运行；旧配方已删除 |
| 同日旧物理检查 | MuJoCo / SynReal / 旧 CUDA graph 各 150 帧保持与重置 | 不适用于当前“共享无 graph”实现 |
| 2026-10-21 官方四任务版 | 公式/观测/传感器测试与两后端各 4 环境短测，保存/加载 | `.validation/microduck-v2/`、`.validation/microduck-sensors/` |
| 四任务实际入口 | SynReal VelStand 新训练→续训→回放；后期课程/推扰 | `audit_entrypoints.log`、`cleanup_synreal.log`、`cleanup_mujoco.log` |
| 专用回放 | 33 项 CPU 测试；两后端同状态观测与有限状态检查 | `.validation/playback/`；不保证独立接触轨迹逐位一致 |
| 训练吞吐审计 | 4 环境核实 24 控制步 / 96 子步 / PPO 参数更新 | `.validation/training_audit/result.log`；非 1024 环境性能结论 |
| 2026-10-27 全任务扩展 | 历史记录：33 任务早/后期 CPU 短步进，Roulade 一次 PPO/镜像损失 | 非全部任务 GPU 收敛验证 |
| Z 惩罚 | 初期 21 项相关测试，含 CPU 三步 | 非步态改善验证 |
| 静止/朝向惩罚初版 -2.0 | 28 项相关测试；MuJoCo GPU 4 环境 2 PPO 轮；定向复跑 5 项 | `.validation/microduck-gait/`；未训练验证最终质量 |

旧简化版曾测得 MuJoCo 回放 36.2→293.7 FPS、SynReal 9.2→12.1 FPS，依赖当时的 graph/模型，不能套用当前版本。
后来的共享 PD 快速回放历史测量（单环境零动作、无网络/渲染）如下，仍不是实时 50 Hz 保证：

| 任务 | SynReal 完整步 / 快速步（控制步每秒） |
| --- | --- |
| Velocity | 9.55 / 15.72 |
| VelStand | 8.57 / 15.66 |
| GroundPick | 9.73 / 15.86 |
| BallKick | 9.62 / 14.79 |

历史日志 `logs/microduck_play_baseline.json`、`logs/microduck_play_after.json`、
`logs/microduck_task_cli_verification.json` 可帮助追溯，不是当前入口依赖。
历史 fixed-friction / self-collision 开关和 10 子步记录已被新配置替代。
不要重新引入“四任务限定”“只支持平地”“始终固定摩擦”“回放始终保留噪声/推扰”等过时描述。

<a id="sources"></a>

## 14. 导出、来源与后续文档维护

本地适配优先改 `official_recipe.py` 等适配层；生成公式和官方源保持可追溯。
不得仅为让测试通过而同步改官方源与生成文件。奖励符号要核对函数返回值，有些官方惩罚函数自身为负。

重新导出使用官方独立环境，操作会修改源码/快照和指纹，不要在纯排查或文档整理时顺手运行：

```bash
examples/microduck/microduck_rl-develop/.venv/bin/python examples/microduck/export_official_recipe.py
examples/microduck/microduck_rl-develop/.venv/bin/python examples/microduck/export_terrain.py
```

配方导出保留快照现有的 `local: true` 奖励和课程，包括其参数和权重；不代表所有其他手工修改都会保留。
变更后检查实际支持/省略项、公式来源、观测维度、任务表和指纹，再做必要短测。

此前文档归并阶段仅改顶层 Markdown，检查了本地链接、源码索引迁移完整性及当时配方指纹不变；
未重跑训练或声明新的物理验证结果。仓库根的历史知识库不在本次合并范围，读取时按日期记录处理。

后续只更新本文中的当前事实、问题和验证记录，不再创建重复的本地项目总览。
原官方资料保留在子目录：

- [官方训练工程规则](microduck_rl-develop/AGENTS.md)：BAM、14 舵机、61D 观测、奖励符号和 sim-to-real 经验。
- [VelStand 训练记录](microduck_rl-develop/docs/velstand_policy.md)：热启动、双教师、各阶段实验。
- [SitStand 训练记录](microduck_rl-develop/docs/sitstand_policy.md)：坐站任务及后期强化。
- [RollerStandUp 记录](microduck_rl-develop/docs/roller_standup_policy_summary.md)：目标高度、轮式关节索引和恢复任务。
- [运行时文档目录](microduck-main/docs/README.md)、[策略 manifest](microduck-main/docs/policy-manifest.md)：部署契约。
- [官方运行时仿真](microduck-main/docs/robot/simulation.md)：daemon + MuJoCo 路径，非共享 RL 训练。

官方 `docs/superpowers/` 是历史设计/实施计划，不能仅凭其条目认定本地已实现。
官方部署、网络、BLE、UI 等文档不合并进本地 RL 指南；本次只清理 `examples/microduck/` 顶层六份 Markdown 正文。

<a id="reward-index"></a>

## 附录 A：官方奖励源码索引

以下从原 `OFFICIAL_REWARDS_INDEX.md` 归并，保留声明、权重表达式和实现位置。
这是静态源码索引，包含条件分支、零权重项、测试台及被覆盖定义，不等于当前运行奖励列表。
实际使用快照的解析配置，再结合本地 `supported/omitted` 和课程判断；本地步态奖励见第 8 节。
来源行号是整理时的定位提示，源码变化后应用函数名搜索复核。

### microduck_ball_kick_env_cfg

| 奖励项/赋值目标 | 函数或类 | 声明时权重表达式 | 配置位置 | 实现位置 |
|---|---|---|---|---|
| `cfg.rewards['ball_forward_velocity']` | `microduck_mdp.ball_forward_velocity` | `12.0` | [声明][reward-source-1] | [mdp.py][reward-source-2] |
| `cfg.rewards['ball_speed_overshoot']` | `microduck_mdp.ball_speed_overshoot_penalty` | `-4.0` | [声明][reward-source-3] | [mdp.py][reward-source-4] |
| `cfg.rewards['support_foot_grounded']` | `microduck_mdp.single_foot_grounded_reward` | `2.0` | [声明][reward-source-5] | [mdp.py][reward-source-6] |
| `cfg.rewards['pose_stand_legs']` | `microduck_mdp.pose_target_match` | `2.0` | [声明][reward-source-7] | [mdp.py][reward-source-8] / [mdp.py][reward-source-9] |
| `cfg.rewards['pose_stand_neck']` | `microduck_mdp.pose_target_match` | `1.0` | [声明][reward-source-10] | [mdp.py][reward-source-8] / [mdp.py][reward-source-9] |
| `cfg.rewards['height_stand']` | `microduck_mdp.height_target_gaussian` | `1.0` | [声明][reward-source-11] | [mdp.py][reward-source-12] |
| `cfg.rewards['self_collisions']` | `mdp.self_collision_cost` | `-1.0` | [声明][reward-source-13] | [rewards.py][reward-source-14] |

### microduck_ground_pick_env_cfg

| 奖励项/赋值目标 | 函数或类 | 声明时权重表达式 | 配置位置 | 实现位置 |
|---|---|---|---|---|
| `cfg.rewards['mouth_ground_proximity']` | `microduck_mdp.mouth_ground_proximity_phased` | `3.0` | [声明][reward-source-15] | [mdp.py][reward-source-16] |
| `cfg.rewards['mouth_perpendicular_to_ground']` | `microduck_mdp.mouth_perpendicular_phased` | `2.0` | [声明][reward-source-17] | [mdp.py][reward-source-18] |
| `cfg.rewards['ground_pick_return_pose_legs']` | `microduck_mdp.ground_pick_return_pose_phased` | `6.0` | [声明][reward-source-19] | [mdp.py][reward-source-20] |
| `cfg.rewards['ground_pick_return_pose_neck']` | `microduck_mdp.ground_pick_return_pose_phased` | `6.0` | [声明][reward-source-21] | [mdp.py][reward-source-20] |
| `cfg.rewards['return_upright']` | `microduck_mdp.ground_pick_return_upright_phased` | `4.0` | [声明][reward-source-22] | [mdp.py][reward-source-23] |
| `cfg.rewards['neck_vel_descent']` | `microduck_mdp.neck_vel_descent_penalty` | `-0.1` | [声明][reward-source-24] | [mdp.py][reward-source-25] |
| `cfg.rewards['mouth_payload_force']` | `microduck_mdp.apply_mouth_payload_force` | `0.0` | [声明][reward-source-26] | [mdp.py][reward-source-27] |
| `cfg.rewards['feet_grounded']` | `microduck_mdp.feet_grounded_reward` | `3.0` | [声明][reward-source-28] | [mdp.py][reward-source-29] |
| `cfg.rewards['feet_flat']` | `microduck_mdp.feet_flat_penalty` | `-2.0` | [声明][reward-source-30] | [mdp.py][reward-source-31] |
| `cfg.rewards['action_rate_l2']` | `mdp.action_rate_l2` | `-2.0` | [声明][reward-source-32] | [rewards.py][reward-source-33] |
| `cfg.rewards['neck_action_rate_l2']` | `microduck_mdp.neck_action_rate_l2` | `-1.0` | [声明][reward-source-34] | [mdp.py][reward-source-35] |
| `cfg.rewards['joint_torques_l2']` | `microduck_mdp.joint_torques_l2` | `-0.005` | [声明][reward-source-36] | [mdp.py][reward-source-37] |
| `cfg.rewards['self_collisions']` | `mdp.self_collision_cost` | `-1.0` | [声明][reward-source-38] | [rewards.py][reward-source-14] |
| `cfg.rewards['head_impact_penalty']` | `microduck_mdp.body_impact_cost` | `-2.0` | [声明][reward-source-39] | [mdp.py][reward-source-40] |

### microduck_roller_crouch_env_cfg

| 奖励项/赋值目标 | 函数或类 | 声明时权重表达式 | 配置位置 | 实现位置 |
|---|---|---|---|---|
| `cfg.rewards['crouch_glide_pose']` | `microduck_mdp.crouch_glide_pose_by_phase` | `6.0` | [声明][reward-source-41] | [mdp.py][reward-source-42] |
| `cfg.rewards['crouch_glide_pose_l1']` | `microduck_mdp.crouch_glide_pose_l1` | `2.0` | [声明][reward-source-43] | [mdp.py][reward-source-44] |
| `cfg.rewards['forward_speed']` | `microduck_mdp.forward_speed_reward` | `1.0` | [声明][reward-source-45] | [mdp.py][reward-source-46] |
| `cfg.rewards['crouch_forward_lean']` | `microduck_mdp.crouch_forward_lean` | `1.0` | [声明][reward-source-47] | [mdp.py][reward-source-48] |
| `cfg.rewards['feet_flat']` | `microduck_mdp.feet_flat_penalty` | `-2.0` | [声明][reward-source-49] | [mdp.py][reward-source-31] |
| `cfg.rewards['self_collisions']` | `mdp.self_collision_cost` | `-1.0` | [声明][reward-source-50] | [rewards.py][reward-source-14] |
| `cfg.rewards['neck_action_rate_l2']` | `microduck_mdp.neck_action_rate_l2` | `-0.5` | [声明][reward-source-51] | [mdp.py][reward-source-35] |
| `cfg.rewards['joint_torques_l2']` | `microduck_mdp.joint_torques_l2` | `-0.001` | [声明][reward-source-52] | [mdp.py][reward-source-37] |

### microduck_roller_slope_env_cfg

| 奖励项/赋值目标 | 函数或类 | 声明时权重表达式 | 配置位置 | 实现位置 |
|---|---|---|---|---|
| `cfg.rewards['upright']` | `microduck_mdp.body_upright_gaussian` | `3.0` | [声明][reward-source-53] | [mdp.py][reward-source-54] |
| `cfg.rewards['alive']` | `microduck_mdp.is_alive` | `1.0` | [声明][reward-source-55] | [mdp.py][reward-source-56] |
| `cfg.rewards['wheel_glide']` | `microduck_mdp.wheel_glide_reward` | `2.0` | [声明][reward-source-57] | [mdp.py][reward-source-58] |
| `cfg.rewards['heading_hold']` | `microduck_mdp.heading_hold_reward` | `1.5` | [声明][reward-source-59] | [mdp.py][reward-source-60] |
| `cfg.rewards['feet_flat']` | `microduck_mdp.feet_flat_penalty` | `-2.0` | [声明][reward-source-61] | [mdp.py][reward-source-31] |
| `cfg.rewards['neck_action_rate_l2']` | `microduck_mdp.neck_action_rate_l2` | `-0.5` | [声明][reward-source-62] | [mdp.py][reward-source-35] |
| `cfg.rewards['neck_joint_pos_l2']` | `microduck_mdp.neck_joint_pos_l2` | `-0.75` | [声明][reward-source-63] | [mdp.py][reward-source-64] |
| `cfg.rewards['joint_torques_l2']` | `microduck_mdp.joint_torques_l2` | `-0.001` | [声明][reward-source-65] | [mdp.py][reward-source-37] |

### microduck_roller_standup_env_cfg

| 奖励项/赋值目标 | 函数或类 | 声明时权重表达式 | 配置位置 | 实现位置 |
|---|---|---|---|---|
| `cfg.rewards['pose_stand_legs']` | `microduck_mdp.pose_target_match` | `8.0` | [声明][reward-source-66] | [mdp.py][reward-source-8] / [mdp.py][reward-source-9] |
| `cfg.rewards['pose_stand_l1']` | `microduck_mdp.pose_l1_penalty` | `5.0` | [声明][reward-source-67] | [mdp.py][reward-source-68] |
| `cfg.rewards['height_stand']` | `microduck_mdp.height_target_gaussian` | `4.0` | [声明][reward-source-69] | [mdp.py][reward-source-12] |
| `cfg.rewards['height_stand_sharp']` | `microduck_mdp.height_target_gaussian` | `4.0` | [声明][reward-source-70] | [mdp.py][reward-source-12] |
| `cfg.rewards['height_stand_l1']` | `microduck_mdp.height_l1_penalty` | `30.0` | [声明][reward-source-71] | [mdp.py][reward-source-72] |
| `cfg.rewards['com_upward_velocity']` | `microduck_mdp.com_upward_velocity` | `3.0` | [声明][reward-source-73] | [mdp.py][reward-source-74] |
| `cfg.rewards['gentle_rise']` | `microduck_mdp.trunk_vertical_accel_penalty` | `+0.02` | [声明][reward-source-75] | [mdp.py][reward-source-76] |
| `cfg.rewards['upright_linear']` | `microduck_mdp.body_upright_linear` | `6.0` | [声明][reward-source-77] | [mdp.py][reward-source-78] |
| `cfg.rewards['upright_sharp']` | `microduck_mdp.upright_gaussian_at_height` | `6.0` | [声明][reward-source-79] | [mdp.py][reward-source-80] |
| `cfg.rewards['standing_composite']` | `microduck_mdp.standing_composite_score` | `15.0` | [声明][reward-source-81] | [mdp.py][reward-source-82] |
| `cfg.rewards['joint_torque_rate_l2']` | `microduck_mdp.joint_torque_rate_l2` | `-0.2` | [声明][reward-source-83] | [mdp.py][reward-source-84] |

### microduck_roulade_env_cfg

| 奖励项/赋值目标 | 函数或类 | 声明时权重表达式 | 配置位置 | 实现位置 |
|---|---|---|---|---|
| `cfg.rewards['roulade_progress']` | `microduck_mdp.roulade_progress` | `8.0` | [声明][reward-source-85] | [mdp.py][reward-source-86] |
| `cfg.rewards['roulade_overspeed']` | `microduck_mdp.roulade_overspeed_penalty` | `-0.1` | [声明][reward-source-87] | [mdp.py][reward-source-88] |
| `cfg.rewards['roulade_head_pivot']` | `microduck_mdp.roulade_head_pivot` | `0.5` | [声明][reward-source-89] | [mdp.py][reward-source-90] |
| `cfg.rewards['roulade_landing_composite']` | `microduck_mdp.roulade_landing_composite` | `4.0` | [声明][reward-source-91] | [mdp.py][reward-source-92] |
| `cfg.rewards['roulade_upright_after_roll']` | `microduck_mdp.roulade_upright_after_roll` | `1.5` | [声明][reward-source-93] | [mdp.py][reward-source-94] |
| `cfg.rewards['roulade_height_after_roll']` | `microduck_mdp.roulade_height_after_roll` | `1.0` | [声明][reward-source-95] | [mdp.py][reward-source-96] |
| `cfg.rewards['roulade_landing_sharp']` | `microduck_mdp.roulade_landing_sharp` | `2.0` | [声明][reward-source-97] | [mdp.py][reward-source-98] |
| `cfg.rewards['roulade_stand_tax']` | `microduck_mdp.roulade_stand_tax` | `5.0` | [声明][reward-source-99] | [mdp.py][reward-source-100] |
| `cfg.rewards['roulade_rise_velocity']` | `microduck_mdp.roulade_rise_velocity` | `0.75` | [声明][reward-source-101] | [mdp.py][reward-source-102] |
| `cfg.rewards['roulade_sagittal']` | `microduck_mdp.roulade_sagittal_penalty` | `-0.1` | [声明][reward-source-103] | [mdp.py][reward-source-104] |
| `cfg.rewards['roulade_lateral_vel']` | `microduck_mdp.roulade_lateral_velocity_penalty` | `-0.5` | [声明][reward-source-105] | [mdp.py][reward-source-106] |
| `cfg.rewards['roulade_flatness']` | `microduck_mdp.roulade_flatness_penalty` | `-0.5` | [声明][reward-source-107] | [mdp.py][reward-source-108] |
| `cfg.rewards['action_rate_l2']` | `mdp.action_rate_l2` | `-0.1` | [声明][reward-source-109] | [rewards.py][reward-source-33] |
| `cfg.rewards['joint_torque_rate_l2']` | `microduck_mdp.joint_torque_rate_l2` | `0.0` | [声明][reward-source-110] | [mdp.py][reward-source-84] |
| `cfg.rewards['arrival_damping']` | `microduck_mdp.body_ang_vel_at_height` | `0.0` | [声明][reward-source-111] | [mdp.py][reward-source-112] |
| `cfg.rewards['gentle_landing']` | `microduck_mdp.trunk_vertical_accel_penalty` | `0.002` | [声明][reward-source-113] | [mdp.py][reward-source-76] |
| `cfg.rewards['self_collisions']` | `mdp.self_collision_cost` | `-0.1` | [声明][reward-source-114] | [rewards.py][reward-source-14] |

### microduck_sitstand_env_cfg

| 奖励项/赋值目标 | 函数或类 | 声明时权重表达式 | 配置位置 | 实现位置 |
|---|---|---|---|---|
| `cfg.rewards['posture_pose_legs']` | `microduck_mdp.posture_pose_match` | `4.0` | [声明][reward-source-115] | [mdp.py][reward-source-116] |
| `cfg.rewards['head_pose_tracking']` | `microduck_mdp.head_pose_tracking` | `0.75` | [声明][reward-source-117] | [mdp.py][reward-source-118] |
| `cfg.rewards['posture_pose_l1']` | `microduck_mdp.posture_pose_l1` | `1.0` | [声明][reward-source-119] | [mdp.py][reward-source-120] |
| `cfg.rewards['posture_height']` | `microduck_mdp.posture_height_gaussian` | `1.0` | [声明][reward-source-121] | [mdp.py][reward-source-122] |
| `cfg.rewards['posture_height_sharp']` | `microduck_mdp.posture_height_gaussian` | `1.0` | [声明][reward-source-123] | [mdp.py][reward-source-122] |
| `cfg.rewards['posture_height_l1']` | `microduck_mdp.posture_height_l1` | `6.0` | [声明][reward-source-124] | [mdp.py][reward-source-125] |
| `cfg.rewards['rise_bootstrap']` | `microduck_mdp.posture_rise_bootstrap` | `0.75` | [声明][reward-source-126] | [mdp.py][reward-source-127] |
| `cfg.rewards['descent_speed']` | `microduck_mdp.trunk_downward_velocity_penalty` | `10.0` | [声明][reward-source-128] | [mdp.py][reward-source-129] |
| `cfg.rewards['rise_speed']` | `microduck_mdp.trunk_upward_velocity_penalty` | `0.0` | [声明][reward-source-130] | [mdp.py][reward-source-131] |
| `cfg.rewards['gentle_motion']` | `microduck_mdp.trunk_vertical_accel_penalty` | `0.05` | [声明][reward-source-132] | [mdp.py][reward-source-76] |
| `cfg.rewards['upright_linear']` | `microduck_mdp.body_upright_linear` | `2.5` | [声明][reward-source-133] | [mdp.py][reward-source-78] |
| `cfg.rewards['upright_while_tall']` | `microduck_mdp.upright_while_tall` | `1.5` | [声明][reward-source-134] | [mdp.py][reward-source-135] |
| `cfg.rewards['posture_stillness']` | `microduck_mdp.posture_stillness` | `2.0` | [声明][reward-source-136] | [mdp.py][reward-source-137] |
| `cfg.rewards['posture_composite']` | `microduck_mdp.posture_composite` | `3.0` | [声明][reward-source-138] | [mdp.py][reward-source-139] |
| `cfg.rewards['action_rate_l2']` | `mdp.action_rate_l2` | `-0.1` | [声明][reward-source-140] | [rewards.py][reward-source-33] |
| `cfg.rewards['joint_torque_rate_l2']` | `microduck_mdp.joint_torque_rate_l2` | `0.0` | [声明][reward-source-141] | [mdp.py][reward-source-84] |
| `cfg.rewards['neck_strain_leaned']` | `microduck_mdp.neck_torque_when_tilted` | `-0.5` | [声明][reward-source-142] | [mdp.py][reward-source-143] |
| `cfg.rewards['self_collisions']` | `mdp.self_collision_cost` | `-1.0` | [声明][reward-source-144] | [rewards.py][reward-source-14] |

### microduck_spin_env_cfg

| 奖励项/赋值目标 | 函数或类 | 声明时权重表达式 | 配置位置 | 实现位置 |
|---|---|---|---|---|
| `cfg.rewards['spin_rate_track']` | `microduck_mdp.spin_rate_track` | `6.0` | [声明][reward-source-145] | [mdp.py][reward-source-146] |
| `cfg.rewards['spin_rate_l1']` | `microduck_mdp.spin_rate_l1` | `0.5` | [声明][reward-source-147] | [mdp.py][reward-source-148] |
| `cfg.rewards['spin_stay_in_place']` | `microduck_mdp.spin_stay_in_place` | `-3.0` | [声明][reward-source-149] | [mdp.py][reward-source-150] |
| `cfg.rewards['spin_wheel_differential']` | `microduck_mdp.spin_wheel_differential` | `1.0` | [声明][reward-source-151] | [mdp.py][reward-source-152] |
| `cfg.rewards['leg_antisymmetry']` | `microduck_mdp.leg_antisymmetry` | `1.0` | [声明][reward-source-153] | [mdp.py][reward-source-154] |
| `cfg.rewards['spin_grounded']` | `microduck_mdp.spin_grounded` | `0.5` | [声明][reward-source-155] | [mdp.py][reward-source-156] |
| `cfg.rewards['feet_flat']` | `microduck_mdp.feet_flat_penalty` | `-2.0` | [声明][reward-source-157] | [mdp.py][reward-source-31] |
| `cfg.rewards['self_collisions']` | `mdp.self_collision_cost` | `-1.0` | [声明][reward-source-158] | [rewards.py][reward-source-14] |
| `cfg.rewards['neck_action_rate_l2']` | `microduck_mdp.neck_action_rate_l2` | `-0.5` | [声明][reward-source-159] | [mdp.py][reward-source-35] |
| `cfg.rewards['neck_joint_pos_l2']` | `microduck_mdp.neck_joint_pos_l2` | `-0.2` | [声明][reward-source-160] | [mdp.py][reward-source-64] |
| `cfg.rewards['joint_torques_l2']` | `microduck_mdp.joint_torques_l2` | `-0.001` | [声明][reward-source-161] | [mdp.py][reward-source-37] |

### microduck_standup_env_cfg

| 奖励项/赋值目标 | 函数或类 | 声明时权重表达式 | 配置位置 | 实现位置 |
|---|---|---|---|---|
| `cfg.rewards['pose_stand_legs']` | `microduck_mdp.pose_target_match` | `2.0` | [声明][reward-source-162] | [mdp.py][reward-source-8] / [mdp.py][reward-source-9] |
| `cfg.rewards['head_pose_tracking']` | `microduck_mdp.head_pose_tracking` | `0.75` | [声明][reward-source-163] | [mdp.py][reward-source-118] |
| `cfg.rewards['head_pose_bias']` | `microduck_mdp.head_pose_bias_penalty` | `0.0` | [声明][reward-source-164] | [mdp.py][reward-source-165] |
| `cfg.rewards['pose_stand_l1']` | `microduck_mdp.pose_l1_penalty` | `1.25` | [声明][reward-source-166] | [mdp.py][reward-source-68] |
| `cfg.rewards['height_stand']` | `microduck_mdp.height_target_gaussian` | `1.0` | [声明][reward-source-167] | [mdp.py][reward-source-12] |
| `cfg.rewards['height_stand_sharp']` | `microduck_mdp.height_target_gaussian` | `1.0` | [声明][reward-source-168] | [mdp.py][reward-source-12] |
| `cfg.rewards['height_stand_l1']` | `microduck_mdp.height_l1_penalty` | `7.5` | [声明][reward-source-169] | [mdp.py][reward-source-72] |
| `cfg.rewards['com_upward_velocity']` | `microduck_mdp.com_upward_velocity` | `0.75` | [声明][reward-source-170] | [mdp.py][reward-source-74] |
| `cfg.rewards['gentle_rise']` | `microduck_mdp.trunk_vertical_accel_penalty` | `0.005` | [声明][reward-source-171] | [mdp.py][reward-source-76] |
| `cfg.rewards['arrival_damping']` | `microduck_mdp.body_ang_vel_at_height` | `0.0` | [声明][reward-source-172] | [mdp.py][reward-source-112] |
| `cfg.rewards['upright_linear']` | `microduck_mdp.body_upright_linear` | `1.5` | [声明][reward-source-173] | [mdp.py][reward-source-78] |
| `cfg.rewards['upright_sharp']` | `microduck_mdp.upright_gaussian_at_height` | `1.5` | [声明][reward-source-174] | [mdp.py][reward-source-80] |
| `cfg.rewards['standing_composite']` | `microduck_mdp.standing_composite_score` | `3.75` | [声明][reward-source-175] | [mdp.py][reward-source-82] |
| `cfg.rewards['body_pose_tracking']` | `microduck_mdp.body_pose_tracking_locomotion` | `0.0` | [声明][reward-source-176] | [mdp.py][reward-source-177] |
| `cfg.rewards['action_rate_l2']` | `mdp.action_rate_l2` | `-0.1` | [声明][reward-source-178] | [rewards.py][reward-source-33] |
| `cfg.rewards['joint_torque_rate_l2']` | `microduck_mdp.joint_torque_rate_l2` | `0.0` | [声明][reward-source-179] | [mdp.py][reward-source-84] |
| `cfg.rewards['self_collisions']` | `mdp.self_collision_cost` | `-1.0` | [声明][reward-source-180] | [rewards.py][reward-source-14] |

### microduck_velocity_env_cfg

| 奖励项/赋值目标 | 函数或类 | 声明时权重表达式 | 配置位置 | 实现位置 |
|---|---|---|---|---|
| `cfg.rewards['self_collisions']` | `mdp.self_collision_cost` | `-1.0` | [声明][reward-source-181] | [rewards.py][reward-source-14] |
| `cfg.rewards['head_pose_tracking']` | `microduck_mdp.head_pose_tracking` | `2.0` | [声明][reward-source-182] | [mdp.py][reward-source-118] |
| `cfg.rewards['body_pose_tracking']` | `microduck_mdp.body_pose_tracking_6d` | `0.0` | [声明][reward-source-183] | [mdp.py][reward-source-184] |
| `cfg.rewards['head_pose_bias']` | `microduck_mdp.head_pose_bias_penalty` | `0.0` | [声明][reward-source-185] | [mdp.py][reward-source-165] |

### microduck_velocity_rollers_env_cfg

| 奖励项/赋值目标 | 函数或类 | 声明时权重表达式 | 配置位置 | 实现位置 |
|---|---|---|---|---|
| `cfg.rewards['com_height_target']` | `microduck_mdp.com_height_target` | `2.0` | [声明][reward-source-186] | [mdp.py][reward-source-187] |
| `cfg.rewards['self_collisions']` | `mdp.self_collision_cost` | `-1.0` | [声明][reward-source-188] | [rewards.py][reward-source-14] |
| `cfg.rewards['feet_flat']` | `microduck_mdp.feet_flat_penalty` | `-2.0` | [声明][reward-source-189] | [mdp.py][reward-source-31] |
| `cfg.rewards['neck_action_rate_l2']` | `microduck_mdp.neck_action_rate_l2` | `-0.5` | [声明][reward-source-190] | [mdp.py][reward-source-35] |
| `cfg.rewards['neck_joint_pos_l2']` | `microduck_mdp.neck_joint_pos_l2` | `-0.5` | [声明][reward-source-191] | [mdp.py][reward-source-64] |
| `cfg.rewards['joint_torques_l2']` | `microduck_mdp.joint_torques_l2` | `-0.001` | [声明][reward-source-192] | [mdp.py][reward-source-37] |
| `cfg.rewards['action_over_limit']` | `microduck_mdp.action_over_limit_penalty` | `-0.5` | [声明][reward-source-193] | [mdp.py][reward-source-194] |
| `cfg.rewards['hip_roll_neutral']` | `microduck_mdp.joint_deviation_l1` | `-2.0` | [声明][reward-source-195] | [mdp.py][reward-source-196] |
| `cfg.rewards['wheel_speed']` | `microduck_mdp.wheel_speed_reward` | `10.0` | [声明][reward-source-197] | [mdp.py][reward-source-198] |
| `cfg.rewards['braking']` | `microduck_mdp.braking_reward` | `1.0` | [声明][reward-source-199] | [mdp.py][reward-source-200] |
| `cfg.rewards['skating_air_time']` | `microduck_mdp.skating_air_time_reward` | `1.5` | [声明][reward-source-201] | [mdp.py][reward-source-202] |
| `cfg.rewards['glide']` | `microduck_mdp.glide_reward` | `4.0` | [声明][reward-source-203] | [mdp.py][reward-source-204] |
| `cfg.rewards['single_support']` | `microduck_mdp.single_support_reward` | `3.0` | [声明][reward-source-205] | [mdp.py][reward-source-206] |
| `cfg.rewards['gait_symmetry']` | `microduck_mdp.gait_symmetry_penalty` | `-1.0` | [声明][reward-source-207] | [mdp.py][reward-source-208] |
| `cfg.rewards['forward_lean']` | `microduck_mdp.forward_lean_reward` | `1.5` | [声明][reward-source-209] | [mdp.py][reward-source-210] |
| `cfg.rewards['heading_hold']` | `microduck_mdp.heading_hold_reward` | `1.0` | [声明][reward-source-211] | [mdp.py][reward-source-60] |

### microduck_velocity_swizzle_env_cfg

| 奖励项/赋值目标 | 函数或类 | 声明时权重表达式 | 配置位置 | 实现位置 |
|---|---|---|---|---|
| `cfg.rewards['leg_symmetry']` | `microduck_mdp.leg_symmetry_reward` | `2.0` | [声明][reward-source-212] | [mdp.py][reward-source-213] |
| `cfg.rewards['grounded']` | `microduck_mdp.grounded_reward` | `1.0` | [声明][reward-source-214] | [mdp.py][reward-source-215] |
| `cfg.rewards['heading_tracking']` | `microduck_mdp.heading_tracking_reward` | `0.0` | [声明][reward-source-216] | [mdp.py][reward-source-217] |
| `cfg.rewards['head_pose_tracking']` | `microduck_mdp.head_pose_tracking` | `0.0` | [声明][reward-source-218] | [mdp.py][reward-source-118] |

### microduck_velstand_env_cfg

| 奖励项/赋值目标 | 函数或类 | 声明时权重表达式 | 配置位置 | 实现位置 |
|---|---|---|---|---|
| `cfg.rewards['body_pose_tracking']` | `microduck_mdp.body_pose_tracking_locomotion` | `BODY_TRACK_WEIGHT` | [声明][reward-source-219] | [mdp.py][reward-source-177] |
| `cfg.rewards['pose']` | `microduck_mdp.variable_posture_body_relaxed` | `pose.weight` | [声明][reward-source-220] | [mdp.py][reward-source-221] |
| `cfg.rewards['upright']` | `microduck_mdp.upright_body_cmd_relative` | `up.weight` | [声明][reward-source-222] | [mdp.py][reward-source-223] |
| `cfg.rewards['track_yaw_fine']` | `microduck_mdp.track_yaw_rate_fine` | `TRACK_YAW_FINE_WEIGHT` | [声明][reward-source-224] | [mdp.py][reward-source-225] |
| `cfg.rewards['track_yaw_fine']` | `microduck_mdp.track_yaw_rate_fine` | `TRACK_YAW_FINE_WEIGHT` | [声明][reward-source-226] | [mdp.py][reward-source-225] |
| `cfg.rewards['upright_progress']` | `microduck_mdp.upright_progress` | `5.0` | [声明][reward-source-227] | [mdp.py][reward-source-228] |
| `cfg.rewards['height_progress']` | `microduck_mdp.height_progress` | `30.0` | [声明][reward-source-229] | [mdp.py][reward-source-230] |
| `cfg.rewards['com_upward_velocity']` | `microduck_mdp.com_upward_velocity` | `0.0` | [声明][reward-source-231] | [mdp.py][reward-source-74] |
| `cfg.rewards['joint_torque_rate_l2']` | `microduck_mdp.joint_torque_rate_l2_fallen_scaled` | `-0.002` | [声明][reward-source-232] | [mdp.py][reward-source-233] |
| `cfg.rewards['action_rate_l2']` | `microduck_mdp.action_rate_l2_fallen_scaled` | `ar.weight` | [声明][reward-source-234] | [mdp.py][reward-source-235] |
| `cfg.rewards['servo_impact']` | `microduck_mdp.body_impact_cost` | `SERVO_IMPACT_WEIGHT * PROTECT_STAGE0_FRAC` | [声明][reward-source-236] | [mdp.py][reward-source-40] |
| `cfg.rewards['head_impact']` | `microduck_mdp.body_impact_cost` | `HEAD_IMPACT_WEIGHT * PROTECT_STAGE0_FRAC` | [声明][reward-source-237] | [mdp.py][reward-source-40] |
| `cfg.rewards['trunk_impact']` | `microduck_mdp.body_impact_cost` | `TRUNK_IMPACT_WEIGHT * PROTECT_STAGE0_FRAC` | [声明][reward-source-238] | [mdp.py][reward-source-40] |
| `cfg.rewards['servo_acc_spike']` | `microduck_mdp.servo_acc_spike_penalty` | `SERVO_ACC_SPIKE_WEIGHT * PROTECT_STAGE0_FRAC` | [声明][reward-source-239] | [mdp.py][reward-source-240] |
| `cfg.rewards['servo_stall']` | `microduck_mdp.servo_stall_penalty` | `SERVO_STALL_WEIGHT * PROTECT_STAGE0_FRAC` | [声明][reward-source-241] | [mdp.py][reward-source-242] |
| `cfg.rewards['gentle_rise']` | `microduck_mdp.trunk_vertical_accel_penalty` | `GENTLE_RISE_WEIGHT * PROTECT_STAGE0_FRAC` | [声明][reward-source-243] | [mdp.py][reward-source-76] |
| `cfg.rewards['air_time']` | `microduck_mdp.feet_air_time_upright` | `at.weight` | [声明][reward-source-244] | [mdp.py][reward-source-245] |
| `cfg.rewards['fallen_tax']` | `microduck_mdp.fallen_state_penalty` | `0.0` | [声明][reward-source-246] | [mdp.py][reward-source-247] |
| `cfg.rewards['recovery_success']` | `microduck_mdp.recovery_success` | `0.0` | [声明][reward-source-248] | [mdp.py][reward-source-249] |

### testbench_env_cfg

| 奖励项/赋值目标 | 函数或类 | 声明时权重表达式 | 配置位置 | 实现位置 |
|---|---|---|---|---|
| `track_target` | `target_angle_tracking` | `3.0` | [声明][reward-source-250] | [testbench_env_cfg.py][reward-source-251] |
| `dof_pos_limits` | `base_mdp.joint_pos_limits` | `-1.0` | [声明][reward-source-252] | [rewards.py][reward-source-253] |
| `action_rate_l2` | `base_mdp.action_rate_l2` | `-0.6` | [声明][reward-source-254] | [rewards.py][reward-source-33] |
| `joint_torques_l2` | `base_mdp.joint_torques_l2` | `-0.001` | [声明][reward-source-255] | [rewards.py][reward-source-256] |
| `joint_vel_l2` | `base_mdp.joint_vel_l2` | `-0.001` | [声明][reward-source-257] | [rewards.py][reward-source-258] |

### velocity_env_cfg

| 奖励项/赋值目标 | 函数或类 | 声明时权重表达式 | 配置位置 | 实现位置 |
|---|---|---|---|---|
| `track_linear_velocity` | `mdp.track_linear_velocity` | `2.0` | [声明][reward-source-259] | [rewards.py][reward-source-260] |
| `track_angular_velocity` | `mdp.track_angular_velocity` | `2.0` | [声明][reward-source-261] | [rewards.py][reward-source-262] |
| `upright` | `mdp.upright` | `1.0` | [声明][reward-source-263] | [rewards.py][reward-source-264] |
| `pose` | `mdp.variable_posture` | `1.0` | [声明][reward-source-265] | [rewards.py][reward-source-266] |
| `body_ang_vel` | `mdp.body_angular_velocity_penalty` | `0.0` | [声明][reward-source-267] | [rewards.py][reward-source-268] |
| `angular_momentum` | `mdp.angular_momentum_penalty` | `0.0` | [声明][reward-source-269] | [rewards.py][reward-source-270] |
| `dof_pos_limits` | `mdp.joint_pos_limits` | `-1.0` | [声明][reward-source-271] | [rewards.py][reward-source-253] |
| `action_rate_l2` | `mdp.action_rate_l2` | `-0.1` | [声明][reward-source-272] | [rewards.py][reward-source-33] |
| `air_time` | `mdp.feet_air_time` | `0.0` | [声明][reward-source-273] | [rewards.py][reward-source-274] |
| `foot_clearance` | `mdp.feet_clearance` | `-2.0` | [声明][reward-source-275] | [rewards.py][reward-source-276] |
| `foot_swing_height` | `mdp.feet_swing_height` | `-0.25` | [声明][reward-source-277] | [rewards.py][reward-source-278] |
| `foot_slip` | `mdp.feet_slip` | `-0.1` | [声明][reward-source-279] | [rewards.py][reward-source-280] |
| `soft_landing` | `mdp.soft_landing` | `-1e-05` | [声明][reward-source-281] | [rewards.py][reward-source-282] |

说明：`mdp.py` 中 `pose_target_match` 有两个同名定义，Python 实际使用后面的第 2538 行定义，前面的定义被覆盖。

### 去重后的奖励实现

共找到 172 处 RewardTermCfg 显式函数注册，关联 110 个去重实现位置（包含一个被后续同名定义覆盖的位置）。

- [joint_torques_l2][reward-source-256] — `examples/microduck/microduck_rl-develop/.venv/lib/python3.12/site-packages/mjlab/envs/mdp/rewards.py`
- [joint_vel_l2][reward-source-258] — `examples/microduck/microduck_rl-develop/.venv/lib/python3.12/site-packages/mjlab/envs/mdp/rewards.py`
- [action_rate_l2][reward-source-33] — `examples/microduck/microduck_rl-develop/.venv/lib/python3.12/site-packages/mjlab/envs/mdp/rewards.py`
- [joint_pos_limits][reward-source-253] — `examples/microduck/microduck_rl-develop/.venv/lib/python3.12/site-packages/mjlab/envs/mdp/rewards.py`
- [track_linear_velocity][reward-source-260] — `examples/microduck/microduck_rl-develop/.venv/lib/python3.12/site-packages/mjlab/tasks/velocity/mdp/rewards.py`
- [track_angular_velocity][reward-source-262] — `examples/microduck/microduck_rl-develop/.venv/lib/python3.12/site-packages/mjlab/tasks/velocity/mdp/rewards.py`
- [upright][reward-source-264] — `examples/microduck/microduck_rl-develop/.venv/lib/python3.12/site-packages/mjlab/tasks/velocity/mdp/rewards.py`
- [self_collision_cost][reward-source-14] — `examples/microduck/microduck_rl-develop/.venv/lib/python3.12/site-packages/mjlab/tasks/velocity/mdp/rewards.py`
- [body_angular_velocity_penalty][reward-source-268] — `examples/microduck/microduck_rl-develop/.venv/lib/python3.12/site-packages/mjlab/tasks/velocity/mdp/rewards.py`
- [angular_momentum_penalty][reward-source-270] — `examples/microduck/microduck_rl-develop/.venv/lib/python3.12/site-packages/mjlab/tasks/velocity/mdp/rewards.py`
- [feet_air_time][reward-source-274] — `examples/microduck/microduck_rl-develop/.venv/lib/python3.12/site-packages/mjlab/tasks/velocity/mdp/rewards.py`
- [feet_clearance][reward-source-276] — `examples/microduck/microduck_rl-develop/.venv/lib/python3.12/site-packages/mjlab/tasks/velocity/mdp/rewards.py`
- [feet_swing_height][reward-source-278] — `examples/microduck/microduck_rl-develop/.venv/lib/python3.12/site-packages/mjlab/tasks/velocity/mdp/rewards.py`
- [feet_slip][reward-source-280] — `examples/microduck/microduck_rl-develop/.venv/lib/python3.12/site-packages/mjlab/tasks/velocity/mdp/rewards.py`
- [soft_landing][reward-source-282] — `examples/microduck/microduck_rl-develop/.venv/lib/python3.12/site-packages/mjlab/tasks/velocity/mdp/rewards.py`
- [variable_posture][reward-source-266] — `examples/microduck/microduck_rl-develop/.venv/lib/python3.12/site-packages/mjlab/tasks/velocity/mdp/rewards.py`
- [neck_action_rate_l2][reward-source-35] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [feet_air_time_upright][reward-source-245] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [upright_progress][reward-source-228] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [height_progress][reward-source-230] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [fallen_state_penalty][reward-source-247] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [recovery_success][reward-source-249] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [servo_stall_penalty][reward-source-242] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [servo_acc_spike_penalty][reward-source-240] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [action_rate_l2_fallen_scaled][reward-source-235] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [joint_torque_rate_l2_fallen_scaled][reward-source-233] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [body_upright_linear][reward-source-78] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [body_upright_gaussian][reward-source-54] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [upright_gaussian_at_height][reward-source-80] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [body_ang_vel_at_height][reward-source-112] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [standing_composite_score][reward-source-82] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [com_upward_velocity][reward-source-74] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [wheel_glide_reward][reward-source-58] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [is_alive][reward-source-56] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [com_height_target][reward-source-187] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [forward_speed_reward][reward-source-46] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [crouch_glide_pose_by_phase][reward-source-42] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [crouch_glide_pose_l1][reward-source-44] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [crouch_forward_lean][reward-source-48] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [feet_flat_penalty][reward-source-31] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [neck_joint_pos_l2][reward-source-64] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [joint_torques_l2][reward-source-37] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [joint_torque_rate_l2][reward-source-84] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [feet_grounded_reward][reward-source-29] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [body_impact_cost][reward-source-40] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [wheel_speed_reward][reward-source-198] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [braking_reward][reward-source-200] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [joint_deviation_l1][reward-source-196] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [pose_target_match][reward-source-8] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [pose_target_match][reward-source-9] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [pose_l1_penalty][reward-source-68] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [height_target_gaussian][reward-source-12] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [height_l1_penalty][reward-source-72] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [trunk_vertical_accel_penalty][reward-source-76] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [trunk_downward_velocity_penalty][reward-source-129] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [upright_while_tall][reward-source-135] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [mouth_ground_proximity_phased][reward-source-16] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [mouth_perpendicular_phased][reward-source-18] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [ground_pick_return_pose_phased][reward-source-20] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [ground_pick_return_upright_phased][reward-source-23] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [neck_vel_descent_penalty][reward-source-25] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [apply_mouth_payload_force][reward-source-27] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [heading_tracking_reward][reward-source-217] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [skating_air_time_reward][reward-source-202] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [single_support_reward][reward-source-206] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [glide_reward][reward-source-204] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [leg_symmetry_reward][reward-source-213] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [grounded_reward][reward-source-215] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [gait_symmetry_penalty][reward-source-208] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [heading_hold_reward][reward-source-60] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [action_over_limit_penalty][reward-source-194] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [forward_lean_reward][reward-source-210] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [variable_posture_body_relaxed][reward-source-221] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [track_yaw_rate_fine][reward-source-225] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [upright_body_cmd_relative][reward-source-223] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [head_pose_tracking][reward-source-118] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [head_pose_bias_penalty][reward-source-165] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [body_pose_tracking_6d][reward-source-184] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [body_pose_tracking_locomotion][reward-source-177] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [ball_forward_velocity][reward-source-2] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [ball_speed_overshoot_penalty][reward-source-4] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [single_foot_grounded_reward][reward-source-6] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [spin_rate_track][reward-source-146] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [spin_rate_l1][reward-source-148] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [spin_stay_in_place][reward-source-150] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [spin_wheel_differential][reward-source-152] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [spin_grounded][reward-source-156] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [leg_antisymmetry][reward-source-154] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [posture_pose_match][reward-source-116] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [posture_pose_l1][reward-source-120] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [posture_height_gaussian][reward-source-122] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [posture_height_l1][reward-source-125] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [posture_composite][reward-source-139] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [posture_stillness][reward-source-137] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [posture_rise_bootstrap][reward-source-127] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [trunk_upward_velocity_penalty][reward-source-131] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [neck_torque_when_tilted][reward-source-143] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [roulade_progress][reward-source-86] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [roulade_head_pivot][reward-source-90] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [roulade_landing_composite][reward-source-92] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [roulade_upright_after_roll][reward-source-94] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [roulade_height_after_roll][reward-source-96] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [roulade_landing_sharp][reward-source-98] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [roulade_stand_tax][reward-source-100] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [roulade_rise_velocity][reward-source-102] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [roulade_overspeed_penalty][reward-source-88] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [roulade_flatness_penalty][reward-source-108] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [roulade_sagittal_penalty][reward-source-104] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`
- [roulade_lateral_velocity_penalty][reward-source-106] — `examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py`

- [target_angle_tracking][reward-source-251] — 舵机测试台本地奖励。

[reward-source-1]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_ball_kick_env_cfg.py:238
[reward-source-2]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:6153
[reward-source-3]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_ball_kick_env_cfg.py:243
[reward-source-4]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:6179
[reward-source-5]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_ball_kick_env_cfg.py:253
[reward-source-6]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:6201
[reward-source-7]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_ball_kick_env_cfg.py:262
[reward-source-8]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:2201
[reward-source-9]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:2538
[reward-source-10]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_ball_kick_env_cfg.py:274
[reward-source-11]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_ball_kick_env_cfg.py:290
[reward-source-12]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:2582
[reward-source-13]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_ball_kick_env_cfg.py:306
[reward-source-14]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/.venv/lib/python3.12/site-packages/mjlab/tasks/velocity/mdp/rewards.py:162
[reward-source-15]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_ground_pick_env_cfg.py:187
[reward-source-16]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:3129
[reward-source-17]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_ground_pick_env_cfg.py:205
[reward-source-18]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:3147
[reward-source-19]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_ground_pick_env_cfg.py:222
[reward-source-20]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:3165
[reward-source-21]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_ground_pick_env_cfg.py:238
[reward-source-22]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_ground_pick_env_cfg.py:255
[reward-source-23]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:3186
[reward-source-24]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_ground_pick_env_cfg.py:270
[reward-source-25]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:3203
[reward-source-26]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_ground_pick_env_cfg.py:284
[reward-source-27]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:3244
[reward-source-28]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_ground_pick_env_cfg.py:312
[reward-source-29]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:1767
[reward-source-30]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_ground_pick_env_cfg.py:324
[reward-source-31]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:1605
[reward-source-32]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_ground_pick_env_cfg.py:339
[reward-source-33]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/.venv/lib/python3.12/site-packages/mjlab/envs/mdp/rewards.py:58
[reward-source-34]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_ground_pick_env_cfg.py:344
[reward-source-35]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:437
[reward-source-36]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_ground_pick_env_cfg.py:349
[reward-source-37]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:1721
[reward-source-38]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_ground_pick_env_cfg.py:354
[reward-source-39]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_ground_pick_env_cfg.py:364
[reward-source-40]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:1786
[reward-source-41]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_roller_crouch_env_cfg.py:175
[reward-source-42]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:1478
[reward-source-43]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_roller_crouch_env_cfg.py:182
[reward-source-44]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:1502
[reward-source-45]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_roller_crouch_env_cfg.py:188
[reward-source-46]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:1400
[reward-source-47]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_roller_crouch_env_cfg.py:195
[reward-source-48]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:1524
[reward-source-49]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_roller_crouch_env_cfg.py:208
[reward-source-50]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_roller_crouch_env_cfg.py:216
[reward-source-51]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_roller_crouch_env_cfg.py:221
[reward-source-52]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_roller_crouch_env_cfg.py:224
[reward-source-53]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_roller_slope_env_cfg.py:132
[reward-source-54]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:852
[reward-source-55]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_roller_slope_env_cfg.py:137
[reward-source-56]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:1261
[reward-source-57]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_roller_slope_env_cfg.py:143
[reward-source-58]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:1235
[reward-source-59]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_roller_slope_env_cfg.py:149
[reward-source-60]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:5117
[reward-source-61]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_roller_slope_env_cfg.py:152
[reward-source-62]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_roller_slope_env_cfg.py:160
[reward-source-63]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_roller_slope_env_cfg.py:167
[reward-source-64]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:1697
[reward-source-65]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_roller_slope_env_cfg.py:170
[reward-source-66]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_roller_standup_env_cfg.py:178
[reward-source-67]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_roller_standup_env_cfg.py:188
[reward-source-68]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:2563
[reward-source-69]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_roller_standup_env_cfg.py:201
[reward-source-70]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_roller_standup_env_cfg.py:210
[reward-source-71]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_roller_standup_env_cfg.py:219
[reward-source-72]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:2596
[reward-source-73]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_roller_standup_env_cfg.py:232
[reward-source-74]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:1042
[reward-source-75]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_roller_standup_env_cfg.py:258
[reward-source-76]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:2609
[reward-source-77]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_roller_standup_env_cfg.py:268
[reward-source-78]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:825
[reward-source-79]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_roller_standup_env_cfg.py:273
[reward-source-80]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:876
[reward-source-81]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_roller_standup_env_cfg.py:289
[reward-source-82]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:952
[reward-source-83]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_roller_standup_env_cfg.py:320
[reward-source-84]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:1743
[reward-source-85]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_roulade_env_cfg.py:232
[reward-source-86]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:7436
[reward-source-87]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_roulade_env_cfg.py:246
[reward-source-88]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:7654
[reward-source-89]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_roulade_env_cfg.py:254
[reward-source-90]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:7465
[reward-source-91]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_roulade_env_cfg.py:267
[reward-source-92]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:7499
[reward-source-93]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_roulade_env_cfg.py:284
[reward-source-94]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:7533
[reward-source-95]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_roulade_env_cfg.py:289
[reward-source-96]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:7554
[reward-source-97]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_roulade_env_cfg.py:305
[reward-source-98]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:7572
[reward-source-99]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_roulade_env_cfg.py:323
[reward-source-100]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:7602
[reward-source-101]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_roulade_env_cfg.py:336
[reward-source-102]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:7629
[reward-source-103]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_roulade_env_cfg.py:353
[reward-source-104]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:7689
[reward-source-105]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_roulade_env_cfg.py:357
[reward-source-106]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:7700
[reward-source-107]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_roulade_env_cfg.py:361
[reward-source-108]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:7673
[reward-source-109]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_roulade_env_cfg.py:371
[reward-source-110]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_roulade_env_cfg.py:372
[reward-source-111]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_roulade_env_cfg.py:383
[reward-source-112]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:905
[reward-source-113]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_roulade_env_cfg.py:402
[reward-source-114]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_roulade_env_cfg.py:410
[reward-source-115]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_sitstand_env_cfg.py:269
[reward-source-116]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:6776
[reward-source-117]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_sitstand_env_cfg.py:283
[reward-source-118]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:5513
[reward-source-119]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_sitstand_env_cfg.py:290
[reward-source-120]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:6792
[reward-source-121]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_sitstand_env_cfg.py:303
[reward-source-122]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:6807
[reward-source-123]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_sitstand_env_cfg.py:313
[reward-source-124]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_sitstand_env_cfg.py:326
[reward-source-125]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:6821
[reward-source-126]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_sitstand_env_cfg.py:340
[reward-source-127]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:6943
[reward-source-128]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_sitstand_env_cfg.py:370
[reward-source-129]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:2637
[reward-source-130]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_sitstand_env_cfg.py:378
[reward-source-131]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:6970
[reward-source-132]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_sitstand_env_cfg.py:386
[reward-source-133]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_sitstand_env_cfg.py:397
[reward-source-134]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_sitstand_env_cfg.py:402
[reward-source-135]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:2691
[reward-source-136]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_sitstand_env_cfg.py:416
[reward-source-137]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:6899
[reward-source-138]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_sitstand_env_cfg.py:440
[reward-source-139]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:6839
[reward-source-140]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_sitstand_env_cfg.py:466
[reward-source-141]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_sitstand_env_cfg.py:467
[reward-source-142]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_sitstand_env_cfg.py:481
[reward-source-143]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:7081
[reward-source-144]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_sitstand_env_cfg.py:492
[reward-source-145]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_spin_env_cfg.py:138
[reward-source-146]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:6338
[reward-source-147]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_spin_env_cfg.py:144
[reward-source-148]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:6360
[reward-source-149]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_spin_env_cfg.py:157
[reward-source-150]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:6381
[reward-source-151]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_spin_env_cfg.py:167
[reward-source-152]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:6436
[reward-source-153]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_spin_env_cfg.py:177
[reward-source-154]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:6491
[reward-source-155]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_spin_env_cfg.py:187
[reward-source-156]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:6466
[reward-source-157]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_spin_env_cfg.py:197
[reward-source-158]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_spin_env_cfg.py:205
[reward-source-159]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_spin_env_cfg.py:210
[reward-source-160]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_spin_env_cfg.py:213
[reward-source-161]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_spin_env_cfg.py:218
[reward-source-162]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_standup_env_cfg.py:235
[reward-source-163]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_standup_env_cfg.py:250
[reward-source-164]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_standup_env_cfg.py:266
[reward-source-165]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:5614
[reward-source-166]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_standup_env_cfg.py:284
[reward-source-167]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_standup_env_cfg.py:302
[reward-source-168]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_standup_env_cfg.py:311
[reward-source-169]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_standup_env_cfg.py:324
[reward-source-170]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_standup_env_cfg.py:351
[reward-source-171]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_standup_env_cfg.py:373
[reward-source-172]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_standup_env_cfg.py:393
[reward-source-173]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_standup_env_cfg.py:414
[reward-source-174]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_standup_env_cfg.py:425
[reward-source-175]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_standup_env_cfg.py:441
[reward-source-176]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_standup_env_cfg.py:466
[reward-source-177]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:5790
[reward-source-178]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_standup_env_cfg.py:504
[reward-source-179]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_standup_env_cfg.py:505
[reward-source-180]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_standup_env_cfg.py:514
[reward-source-181]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py:323
[reward-source-182]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py:710
[reward-source-183]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py:717
[reward-source-184]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:5705
[reward-source-185]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py:741
[reward-source-186]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_velocity_rollers_env_cfg.py:189
[reward-source-187]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:1274
[reward-source-188]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_velocity_rollers_env_cfg.py:194
[reward-source-189]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_velocity_rollers_env_cfg.py:203
[reward-source-190]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_velocity_rollers_env_cfg.py:211
[reward-source-191]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_velocity_rollers_env_cfg.py:214
[reward-source-192]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_velocity_rollers_env_cfg.py:217
[reward-source-193]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_velocity_rollers_env_cfg.py:226
[reward-source-194]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:5152
[reward-source-195]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_velocity_rollers_env_cfg.py:239
[reward-source-196]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:2113
[reward-source-197]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_velocity_rollers_env_cfg.py:253
[reward-source-198]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:1816
[reward-source-199]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_velocity_rollers_env_cfg.py:259
[reward-source-200]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:1882
[reward-source-201]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_velocity_rollers_env_cfg.py:277
[reward-source-202]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:4896
[reward-source-203]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_velocity_rollers_env_cfg.py:292
[reward-source-204]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:4987
[reward-source-205]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_velocity_rollers_env_cfg.py:310
[reward-source-206]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:4944
[reward-source-207]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_velocity_rollers_env_cfg.py:323
[reward-source-208]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:5086
[reward-source-209]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_velocity_rollers_env_cfg.py:334
[reward-source-210]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:5187
[reward-source-211]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_velocity_rollers_env_cfg.py:344
[reward-source-212]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_velocity_swizzle_env_cfg.py:44
[reward-source-213]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:5035
[reward-source-214]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_velocity_swizzle_env_cfg.py:50
[reward-source-215]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:5065
[reward-source-216]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_velocity_swizzle_env_cfg.py:79
[reward-source-217]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:4877
[reward-source-218]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_velocity_swizzle_env_cfg.py:134
[reward-source-219]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_velstand_env_cfg.py:571
[reward-source-220]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_velstand_env_cfg.py:584
[reward-source-221]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:5417
[reward-source-222]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_velstand_env_cfg.py:590
[reward-source-223]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:5477
[reward-source-224]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_velstand_env_cfg.py:617
[reward-source-225]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:5455
[reward-source-226]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_velstand_env_cfg.py:627
[reward-source-227]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_velstand_env_cfg.py:689
[reward-source-228]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:591
[reward-source-229]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_velstand_env_cfg.py:702
[reward-source-230]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:620
[reward-source-231]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_velstand_env_cfg.py:710
[reward-source-232]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_velstand_env_cfg.py:727
[reward-source-233]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:814
[reward-source-234]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_velstand_env_cfg.py:733
[reward-source-235]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:793
[reward-source-236]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_velstand_env_cfg.py:744
[reward-source-237]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_velstand_env_cfg.py:749
[reward-source-238]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_velstand_env_cfg.py:754
[reward-source-239]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_velstand_env_cfg.py:759
[reward-source-240]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:760
[reward-source-241]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_velstand_env_cfg.py:764
[reward-source-242]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:738
[reward-source-243]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_velstand_env_cfg.py:770
[reward-source-244]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_velstand_env_cfg.py:780
[reward-source-245]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:572
[reward-source-246]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_velstand_env_cfg.py:788
[reward-source-247]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:649
[reward-source-248]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/microduck_velstand_env_cfg.py:808
[reward-source-249]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/mdp.py:691
[reward-source-250]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/testbench_env_cfg.py:206
[reward-source-251]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/testbench_env_cfg.py:99
[reward-source-252]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/testbench_env_cfg.py:215
[reward-source-253]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/.venv/lib/python3.12/site-packages/mjlab/envs/mdp/rewards.py:81
[reward-source-254]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/testbench_env_cfg.py:219
[reward-source-255]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/testbench_env_cfg.py:223
[reward-source-256]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/.venv/lib/python3.12/site-packages/mjlab/envs/mdp/rewards.py:32
[reward-source-257]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/src/mjlab_microduck/tasks/testbench_env_cfg.py:227
[reward-source-258]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/.venv/lib/python3.12/site-packages/mjlab/envs/mdp/rewards.py:42
[reward-source-259]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/.venv/lib/python3.12/site-packages/mjlab/tasks/velocity/velocity_env_cfg.py:276
[reward-source-260]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/.venv/lib/python3.12/site-packages/mjlab/tasks/velocity/mdp/rewards.py:27
[reward-source-261]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/.venv/lib/python3.12/site-packages/mjlab/tasks/velocity/velocity_env_cfg.py:281
[reward-source-262]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/.venv/lib/python3.12/site-packages/mjlab/tasks/velocity/mdp/rewards.py:47
[reward-source-263]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/.venv/lib/python3.12/site-packages/mjlab/tasks/velocity/velocity_env_cfg.py:286
[reward-source-264]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/.venv/lib/python3.12/site-packages/mjlab/tasks/velocity/mdp/rewards.py:67
[reward-source-265]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/.venv/lib/python3.12/site-packages/mjlab/tasks/velocity/velocity_env_cfg.py:294
[reward-source-266]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/.venv/lib/python3.12/site-packages/mjlab/tasks/velocity/mdp/rewards.py:385
[reward-source-267]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/.venv/lib/python3.12/site-packages/mjlab/tasks/velocity/velocity_env_cfg.py:307
[reward-source-268]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/.venv/lib/python3.12/site-packages/mjlab/tasks/velocity/mdp/rewards.py:184
[reward-source-269]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/.venv/lib/python3.12/site-packages/mjlab/tasks/velocity/velocity_env_cfg.py:312
[reward-source-270]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/.venv/lib/python3.12/site-packages/mjlab/tasks/velocity/mdp/rewards.py:196
[reward-source-271]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/.venv/lib/python3.12/site-packages/mjlab/tasks/velocity/velocity_env_cfg.py:317
[reward-source-272]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/.venv/lib/python3.12/site-packages/mjlab/tasks/velocity/velocity_env_cfg.py:318
[reward-source-273]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/.venv/lib/python3.12/site-packages/mjlab/tasks/velocity/velocity_env_cfg.py:319
[reward-source-274]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/.venv/lib/python3.12/site-packages/mjlab/tasks/velocity/mdp/rewards.py:209
[reward-source-275]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/.venv/lib/python3.12/site-packages/mjlab/tasks/velocity/velocity_env_cfg.py:330
[reward-source-276]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/.venv/lib/python3.12/site-packages/mjlab/tasks/velocity/mdp/rewards.py:241
[reward-source-277]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/.venv/lib/python3.12/site-packages/mjlab/tasks/velocity/velocity_env_cfg.py:341
[reward-source-278]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/.venv/lib/python3.12/site-packages/mjlab/tasks/velocity/mdp/rewards.py:271
[reward-source-279]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/.venv/lib/python3.12/site-packages/mjlab/tasks/velocity/velocity_env_cfg.py:352
[reward-source-280]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/.venv/lib/python3.12/site-packages/mjlab/tasks/velocity/mdp/rewards.py:326
[reward-source-281]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/.venv/lib/python3.12/site-packages/mjlab/tasks/velocity/velocity_env_cfg.py:362
[reward-source-282]: /home/ubuntu/work/style3d-sim/examples/microduck/microduck_rl-develop/.venv/lib/python3.12/site-packages/mjlab/tasks/velocity/mdp/rewards.py:356
