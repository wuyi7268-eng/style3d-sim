# MicroDuck / SynReal 项目介绍与 AI 交接

> 历史恢复备份：开头敏感信息已移除，来源与恢复限制见同目录 README.md。









本文是本目录唯一维护的项目说明、使用向导和 AI 交接正文。
合并了顶层 README、AGENTS、ALL_TASKS、OFFICIAL_PORT_STATUS、VALIDATION 和 OFFICIAL_REWARDS_INDEX。
旧顶层正文已归并，部分文件（含 `VALIDATION.md`）已删除；`AGENTS.md` 指向本文。
两个官方子项目中的 Markdown 保持原样，不属于本次合并范围。

接手当前软垫任务，先读下方最新状态、第 3.1 节入口关系、第 5 节命令、第 11.0 节待办。
第 1～12 节同时保留原训练项目背景；第 13 节及附录为历史证据和源码索引。

快速导航：[目的](#purpose) · [接手规则](#rules) · [代码导航](#architecture) · [任务配置](#configuration) ·
[训练与评估 UI](#workbench) ·
[独立软垫入口与当前参数](#soft-pad-handoff) · [软垫训练待办](#soft-pad-next) ·
[运行命令](#commands) · [物理契约](#contracts) · [移植差异](#alignment) · [Velocity](#velocity) ·
[VelStand](#velstand) · [回放续训](#checkpoints) · [实验与待办](#next) · [验证](#validation) ·
[运行中任务与日志](#active-runs) · [SynReal 排查交接](#synreal-velocity-handoff) ·
[起步与停步新发现](#velocity-start-stop) · [SynReal 静止策略诊断与下一步](#synreal-stall) ·
[历史问题](#history) · [来源维护](#sources) · [奖励源码索引](#reward-index)

文档与代码不一致时核对代码；解释某次实验时以其 `config.json`、checkpoint 和对应代码版本为准。

### StandUp / SynReal 训练入口验收（2026-10-30）

用户已训练好 Velocity，下一步先独立训练站起专家，再考虑走路摔倒后自动恢复。
官方路线为 Velocity、StandUp 两个专家 → 从 Velocity 热启动 VelStand → PPO + 双教师 BC；
最终是一个 VelStand 策略。当前本地仍没有双教师 BC 或跨任务 warm-start，不能直接用
`--resume velocity/model.pt --task standup/velstand` 代替。此次仅完成 StandUp 训练入口整理和验收。

**已有移植核对：** 本地 `standup` 已包含官方奖励和出生课程，本次复用，没有另写一套起身奖励。
使用官方副本的独立 Python 实际构造 `Mjlab-StandUp-MicroDuck` 配置，与快照的奖励、事件、课程、
终止、观测、动作、指令、PPO 和回合长度逐项相等。20 个奖励均有适配实现；Actor 61 维、Critic 68 维。
仍是 SynReal + 共享 PD，不是官方 BAM 的完整复现；Critic 双脚力通道、部分动力学随机化未适配，
默认关闭自碰撞，因此 self_collisions 奖励没有有效信号。启用自碰撞会改变碰撞资产，须另外验证。

**训练配置：** `configs/rtx4090_synreal_standup.json` 选择 SynReal / standup / official 平地，15000 轮。
不覆盖任务 PPO：rollout 24、每 250 轮保存、adaptive LR 初值 0.001、entropy 0.01。
环境数、物理参数仍来自 `env.py`；不改已训练 Velocity 的奖励、物理默认值或 checkpoint。

```bash
# 正式独立训练；不传 velocity 的 --resume 或 --init-from。
.venv/bin/python -u -m examples.microduck.train \
  --training-config examples/microduck/configs/rtx4090_synreal_standup.json \
  --num-envs 1024 --iterations 15000

# 小规模训练链路检查。
.venv/bin/python -u -m examples.microduck.train \
  --training-config examples/microduck/configs/rtx4090_synreal_standup.json \
  --num-envs 64 --iterations 5 --run-name standup_smoke

# 查看训练所得站起策略；模型路径替换为实际运行目录。
.venv/bin/python -m examples.microduck.play --solver synreal \
  --checkpoint logs/rsl_rl/microduck_synreal/<standup运行目录>/model_14999.pt \
  --num-envs 1 --viewer gl --num-frames 3000

# 四类出生分别评估，8环境 × 每类3批 × 每回合6秒。
.venv/bin/python -m examples.microduck.evaluate_standup --solver synreal \
  --checkpoint logs/rsl_rl/microduck_synreal/<standup运行目录>/model_14999.pt \
  --num-envs 8 --viewer null --episodes 3 --output logs/standup_evaluation.json
```

官方回合长 6 秒（300 控制步），倒地不会立即终止。出生课程按站立/坐姿/俯卧/仰卧的概率为：
初期 40/40/20/0%，第 600 轮 25/30/35/10%，第 1500 轮 20/25/30/25%，第 2500 轮 15/20/30/35%。
仰卧保留官方 ±90° 扰动，包含接近侧躺的起点；不是覆盖任意关节姿态的全面恢复保证。

`evaluate_standup` 可用 `--start-pose standing|sitting|face_down|face_up` 单独检查。
强制出生类别但保留官方高度、关节和倾角噪声；关闭逐帧观测噪声和自动推扰，使用零 twist/head/body 指令，
保留训练的传感器延迟/偏置。判据：躯干离地 ≥0.10 m、倾角 ≤20°、双脚几何接触，连续保持 0.5 秒。
分别报告达到判据的比例、结束时仍满足连续保持的比例、首次满足保持判据的平均时间。
超时后自动复位到站姿不能计为成功；评估给末帧保留一个步长后再超时，以读取真实结束姿态。
此几何判据不是严格接触力/承重测量，也不能替代视觉验收。

**本次验证：** 50 项相关回归通过；新增 StandUp 套件开启 GPU 后 13 项通过，包含真实 SynReal
8 环境、最终课程下四类起点各 20 步；Ruff 通过。真实 SynReal 64 环境 × 5 轮 PPO 已完成，
共 7680 个环境步，保存 model_0/model_4。四类起点各 4 环境 × 6 秒 checkpoint 评估完成，
该短测权重坐姿/俯卧/仰卧起身均 0/4，不能声称已学会站起。未启动 15000 轮正式训练。
证据：`.validation/microduck_standup_port/` 下 provenance.json、smoke.log、late_gpu.log、evaluation.json。

### 回放边缘复位更新（2026-10-30）

同日检查点筛选：完整TensorBoard曲线到13603轮，约3000轮后速度误差/奖励进入波动平台，
未通过软垫行走收敛验收。重新按训练保存的1.2m场景评估100/300/600/1200/6000/9300/13600，
100/300/13600各两seed，其余单seed，共10次16环境/12秒真实回放。
前进稳定性与平地保留优先时，暂推荐本次混合目录model_300.pt；100更快但前进/停步仍有摔倒。
300两seed的软垫前进档位无摔倒，转向共2次失败，头颈异常仍在，不代表已训练成功。
0.3指令下300的平地/软垫vx均值约0.177/0.116，13600约0.062/0.063m/s。
该排序保留训练边缘复位、统一成熟课程，不可推广为当前3m场景或跨边缘最优；未穷举137份权重。
约601轮起头部偏差惩罚生效、后续课程加重，早期奖励下降不可直接视为退化。
图表、原始轨迹与筛选说明：`.validation/microduck_selection_20261030/REPORT.md`。

随后用户将env.py改为3m/24×24×3、base_z=0（12.5cm水平单元、凸起6cm）。
对“走几步低头摔倒”沿用9000轮/0.3m/s做13秒、各5平地/5软垫对照：原策略软垫3次失败/平地0次；
仅将头颈动作替换为姿态指令偏移、保留原PD/力矩限制后，头颈极端姿态被抑制，
但软垫13次失败/平地2次，速度更差。此干预未合入正式回放，不能把“固定头”当作修复。
部分回合身体先倾斜，部分头颈先偏离；全身控制因果尚未定位。当前3m场景不同于下文1m凹槽场景。
证据`.validation/microduck_head_control_20261030/REPORT.md`；未修改用户场景参数或训练。

同日后续场景核查：用户当前9000轮/10环境/0.3m/s回放读取env.py的1m垫，训练config保存1.2m，
两者均24×24×3网格。取消边缘复位后，真实诊断记录到脚与凹槽硬边角接触及向垫内法向，
部分轨迹随后减速/倒退、头颈异常和摔倒。扩大到2m/48×48×3保持当前单元间距后无周边地板接触，
仍有软垫内部失稳：两组各5软垫环境/13秒，均3次失败、平地0次、变形保护0次。
因此不能说场景完全无问题，也不能归结为只有边缘问题；接触法向不等于已测接触力或唯一根因。
未改用户场景参数，证据`.validation/microduck_edge_scene_20261030/REPORT.md`。

用户要求取消回放时到海绵边缘重置：`play.main()`构建环境后将混合地形的
`pad_edge_timeout_margin`设为`None`，允许离开及重新进入软垫，只保留距场景中心2.5m的外边界。
训练仍使用原10cm内缩边缘规则；已运行的训练与回放不热更新。摔倒、回合时限、变形保护仍保留。
独立`soft_pad_scene`原本就没有自动episode复位。此更改仅表示允许观察跨边缘，不能说明策略已学会跨越。
`mixed_terrain.py`属于配方指纹，旧checkpoint诊断回放可能需要显式`--allow-recipe-mismatch`；
`--use-checkpoint-environment`仍恢复保存的物理参数，但play按本次要求关闭边缘复位。
验证：边缘四方向/重新入垫/场景外界、原训练边界及固定分组等定向22项通过，Python编译及差异空白检查通过。
未执行真实SynReal跨边缘步态验收。只取消边缘复位，不代表已修复头颈失稳或跨边缘碰撞问题。

### 本次接手必读（2026-10-29 17:52，Asia/Shanghai）

本节为当前状态，优先于下方旧摘要与历史记录。本轮用户要求“完成交接”，仅更新本文，不改训练代码、
不停止/重启现有训练、不自动开启额外GPU实验。工作树有大量历史修改、删除和未跟踪模块，未提交；
不要reset/clean，不要只保留已跟踪文件。环境参数集中在env.py，奖励/事件集中在official_snapshot.json。
认证信息不得输出、复制到报告或写入日志。生成Git diff日志时需避免夹带已有敏感内容。

**正在运行的训练（17:51:52只读快照）：**

- PID `1266000`，用户自行启动；操作前重新核对进程，不能根据本文旧PID发信号。
- 128环境、`--pad-probability 0.3`、30000轮、viewer=null；固定38个软垫环境＋90个纯平地环境。
- 运行目录：`logs/rsl_rl/microduck_synreal/2026-10-29_17-46-16_491657_velocity_inset_pad_finetune/`。
- 来源：`logs/rsl_rl/microduck_synreal/2026-10-28_10-04-52_798619_velocity_PCG/model_13500.pt`，使用init-from。
- TensorBoard最后读到迭代编号80，采样4.117 s、更新0.052 s，累计压缩保护2次、翻转/非有限0次。
  训练组速度误差pad约0.233、flat约0.151 m/s；这些不是固定指令评估，也不能证明策略收敛。
- 核查时仅`model_0.pt`，每100轮保存；后续请重新列目录。已生成一次compression_limit诊断。
- GPU计算进程仅此训练，约1226 MiB；前文“没有训练进程”均为更早快照。

当前进程命令（仅记录，不要重复启动）：

```bash
.venv/bin/python -u -m examples.microduck.train \
  --training-config examples/microduck/configs/rtx4090_mixed_soft_pad.json \
  --init-from /home/ubuntu/work/style3d-sim/logs/rsl_rl/microduck_synreal/2026-10-28_10-04-52_798619_velocity_PCG/model_13500.pt \
  --num-envs 128 --pad-probability 0.3 --iterations 30000 \
  --run-name velocity_inset_pad_finetune --viewer null
```

**当前磁盘与本次运行的物理配置：**

| 项目 | 当前值 |
| --- | --- |
| PhysicsConfig | dt=0.005、decimation=4（控制周期0.02 s）、非线性5/线性15 |
| 软垫 | 1.2×1.2×0.06 m、cells=(24,24,3)、base_z=-0.06，固定底面 |
| 材料 | density=80、shear_modulus=2000、lame_lambda=10000、damping=5 |
| 接触 | contact_stiffness=5000、particle_radius=0.0015 |
| 分组 | fixed_groups_center；仅软垫组创建软体，每次复位在中心 |
| 边界 | 软垫组根节点|x|或|y|>0.50 m时timeout（边缘内缩10 cm），不算摔倒 |
| 地板 | 真正嵌入凹槽：软垫组4周边块＋底板，纯平地组1块，无覆盖垫面的z=0硬盖板 |
| 正常训练 | Actor观测噪声开启、摩擦随机化开启、每3–6秒X/Y±0.3 m/s随机速度推扰 |
| PPO | 随机动作采样、权重更新开启；rollout24、save100、LR=1e-4 fixed、entropy=0.001 |

JSON默认30000轮；之前建议的64环境/10%软垫/3000轮只是较早命令，不能当作当前实际运行。
EnvironmentConfig仍默认1024环境/50%，因此命令必须显式指定目标规模。pad数量为floor(N*p+0.5)，
比例表示固定分组，不是每回合重新抽样。材料lambda增大改善短测压缩次数，但改变了可压缩性，未做真实海绵标定。
0.25保护是单个四面体剩余体积比，不是整垫厚度减少25%，也不能把所有正体积阈值越界都称为数值翻转。

**训练/回放差异与新诊断开关：**

- 普通train保留动作探索、观测噪声、推扰、摩擦随机化及学习；play使用均值动作、关闭每步观测噪声/推扰，
  默认关闭摩擦随机化。两者共用物理实现，但指令采样、头部指令、复位课程和归一化更新也可能不同。
- 用户要求在train入口对齐排查，不能仅叫用户换用play。已实现
  `--diagnostic-no-exploration --num-frames 30000 --command 0 0 0`：仍直接走训练env.step，
  关闭探索与PPO更新、冻结归一化、不保存权重，并对齐play的指令和checkpoint复位阶段。
  它不会关闭摩擦随机化；与默认play对照须另加`--no-randomize-foot-friction`。
- 用户随后明确恢复正常训练，上述诊断开关默认false；当前进程未使用它。
  `--command`仅在play/诊断模式控制指令，对普通PPO训练无效。
- 13500权重的动作std约0.10–0.15 rad，init-from同时恢复它与归一化；仅改init_noise_std会被加载覆盖，
  entropy_coef=0也不是关闭探索。不要将std设零后继续概率型PPO更新。
- 当前训练GUI路径在env.step中按viewer是否为空决定是否update_commands，这一点与无窗口训练的采样行为
  不完全一致，应作为后续审计项；本次未修改，不要笼统声称打开窗口只影响渲染。

**已验证与尚未解决：**

- 1.2×1.2 m/10 cm边界改动定向29项通过；新增训练诊断定向19项通过、Ruff/空白检查通过。
  更早完整195通过/9跳过属于历史版本，不代表本次重新跑全套。本次交接只做只读状态检查及文档检查。
- 真实SynReal4环境半垫/100帧：训练诊断与play配置相同、首帧动作相同，物理步进后出现小差异并放大；
  同一play重复运行也有近似幅度差异，未定位原生内部原因，不能声称两路逐位等价或训练入口有独有错误。
  证据：`.validation/microduck_train_play_diagnostic/`（含train.pt、play_first.pt、play.pt及比较JSON）。
- 原平地13500在软垫上给速度后低头/摔倒尚未解决。`.validation/microduck_pad_bowing/`：
  旧2×2 m配置、8环境、5/10、先站1秒再vx=0.3，400步内4个pad均摔倒而flat未摔，0次变形保护/翻转；
  身体倾角增大早于大幅颈部动作。5/50对照连flat也失稳，不能推荐“迭代越大越稳”。
  原始result.json的pad键被指标覆盖，5/50子目录保留pad_config；不要把该数据当现1.2 m垫的验收。
- 软垫边缘弹回：低位碰撞探针确认凹槽竖直侧壁法向向垫内；不是完整鸭脚轨迹根因证明。
  用户选择恢复边缘复位，未改倒角/斜坡。证据`.validation/microduck_edge_contact/`及`microduck_edge_reset/`。
- 2×2 m网格40×40×3导致每垫28800四面体，当前24×24×3为10368，数量减少64%，耗时不保证同比下降。
  前次128环境2m垫5/30独占采样约24秒/轮；同时运行的GPU诊断也曾拖慢训练，做性能比较须隔离进程。
- 旧软垫微调策略平地退化、软垫行走质量未验收。后续应评估当前新保存权重与原13500，固定物理配置和指令，
  分别观察平地/软垫的站立、行走、倾角、失败率；不能从训练窗口探索抖动直接认定策略退化。
- 暂不再改奖励/材料/求解器以免干扰现有运行；如用户要求排查先读当前进程对应config，磁盘不会热更新旧进程。

### 历史接手快照（2026-10-29 10:22，Asia/Shanghai；下列“当前”均指当时）

用户当前目标是同一 MicroDuck 策略在 SynReal 平地及海绵垫上稳定运动，混合训练目标规模 1024。
用户明确要求环境及求解参数集中在 `env.py`，不要再拆出新的环境配置文件。
用户喜欢终端直接输出训练进度，可用 `python -u`；不要默认重启长训练或反复要求已授权操作的确认。
本次请求仅做交接，下面是只读核查结果和此前已完成的实现。

**当前进程与权重：** 本次核查没有 MicroDuck train/play/soft_pad_scene 或 authenticated_worker 进程，
`nvidia-smi` 没有计算进程，GPU 利用率约 3%、总显存约 1175 MiB（瞬时快照）。
后文 PID 741258、678612、467823 及“训练继续运行”均是历史状态，不能复用或据此发送信号。
当前没有已验证可复用的认证 worker；真实 SynReal 运行需正常终端登录，凭据不能进入聊天、文件或日志。

| 用途 | 路径（仓库相对） |
| --- | --- |
| 用户曾认为有一定软垫行走能力的基线 | `logs/rsl_rl/microduck_synreal/2026-10-28_10-04-52_798619_velocity_PCG/model_11100.pt` |
| 混合训练实际迁移来源、平地目录最新权重 | `logs/rsl_rl/microduck_synreal/2026-10-28_10-04-52_798619_velocity_PCG/model_13500.pt` |
| 用户反馈退化的混合运行 | `logs/rsl_rl/microduck_synreal/2026-10-28_21-40-08_772789_velocity_flat_pad_finetune/` |
| 混合目录最新保存权重 | 上述混合目录中的 `model_2900.pt`；也保留 `model_2000.pt` 等每 100 轮权重 |

**历史元数据已发生变化：** 前次审计读取该混合运行 `solver_settings.linear_iterations=50`，
本次磁盘 `config.json` 为 **15**；原审计证据 `.validation/microduck_finetune_regression/audit.json` 仍保存 50。
本次未修改该运行配置，也不推断由谁修改；不能拿现在的 15 反推当时训练使用 15。
`--use-checkpoint-environment` 只读取当前磁盘元数据，不能恢复被改写的历史；重建 50 次对照需显式设置并留证。
实际该混合运行训练环境数为 **256**，不等于当前源码默认的 1024。平地来源配置没有保存求解次数，
旧策略现在回放默认 15，不等于已从其历史元数据证明训练当时使用 15。

**最高优先未解决问题：** 用户明确反馈混合训练约 2000 轮后，平地上零指令也站不稳、给指令走路也不稳；
不是仅因 `--command 0 0 0` 不前进。该问题尚未修复，也没有真实 SynReal 同条件新旧权重对照。
不要以“已经训练成功”“再加轮数即可”或“必然是灾难性遗忘”作为结论。
审计确认迁移来源 model_13500 的 SHA256 匹配，检查的 model_0/2000/2900 参数有限、Actor 已更新，
Actor/Critic 归一化统计与来源逐位一致，排除这几份 checkpoint 的归一化漂移。

| 训练日志窗口平均 | 0–99 轮 | 1900–1999 轮 |
| --- | ---: | ---: |
| 平地 upright | 0.9984 | 0.9991 |
| 平地速度误差（m/s） | 0.1543 | 0.1180 |
| 垫面速度误差（m/s） | 0.2367 | 0.1271 |
| 当前垫面世界占比 | 11.95% | 32.35% |

这些是带采样动作、混合指令的训练统计，不能代替固定指令确定性回放；日志与用户现象的矛盾需要排查。
每次 reset 50% 垫面出生不保证 50% 训练帧。迁移使用固定学习率 1e-4、entropy=0.001、全网络更新，
没有保持旧策略的蒸馏/KL 锚定；当前 RSL-RL 仅 adaptive schedule 分支使用 desired_kl 调学习率。
这说明保留旧能力没有保证，不足以证明这就是失稳根因。

**性能问题仍未解决：** 用户反馈旧软垫回放约 15–20 FPS；调查时曾有 256 环境训练与回放争用 GPU。
统一环境后，交互每帧增加了 4 次接触容量同步检查、8 次 GPU 计数回读和一次帧末同步；尚未优化或测出各自贡献。
先前“GPU 100%”不是当前状态。不能把迭代数变化、争用、重构同步三个因素合并为已验证单一根因。

**崩溃后完整性复查已完成：** 用户担心 Codex 崩溃未保存；磁盘源码、两份训练 JSON 和新增测试均完整。
重新执行完整回归 **136 passed / 9 skipped**（53.75 s）、Ruff、`git diff --check`，
并解析 44 个本地 Python 文件和 5 个 JSON，通过；训练与回放配置解析一致。
上轮 CUDA 预览构建/状态检查证据为 `.validation/env_settings_preview.log`（无动力学）。
未执行重构后的真实 SynReal 行走或 PPO；CPU/配置测试不能当作步态验收。
工作树仍有大量历史修改、删除、未跟踪文件，**已保存但未提交 Git**；不要 reset/clean 或只转交已跟踪文件。

证据目录：`.validation/microduck_env_unification/`（含 before 冻结源码、几何/CPU/策略对照），
`.validation/microduck_finetune_regression/audit.json`（训练统计及权重审计），
`.validation/microduck_mixed/`（历史故障隔离/训练记录；worker 状态文件可能已过期）。
这些通常被 Git 忽略，跨机器交接须另行保留需要的证据。

### 最新交接摘要

- **2026-10-29 用户要求恢复正常训练配置。** 恢复velocity Actor的`enable_corruption=true`、
  `push_robot.velocity_range.x/y=[-0.3,0.3]`（原3–6 s间隔），以及训练默认`randomize_foot_friction=True`。
  正常train默认仍使用随机动作探索和PPO更新；对照开关保留为显式可选，正常训练不要传
  `--diagnostic-no-exploration`。回放仍关闭每步观测噪声、自动推扰及默认摩擦随机化。
  本轮未改软垫尺寸/材料、边缘复位、求解或PPO超参数，未启动或重启训练；新进程生效。

- **2026-10-29 训练入口增加无探索对照模式。** 用户要在训练环境中对齐回放，而不是被要求改用play。
  `train --init-from ... --diagnostic-no-exploration --num-frames 30000 --command 0 0 0`：
  仍直接调用训练`env.step`（奖励、传感器、终止、复位均保留），使用均值动作，冻结网络/归一化，
  不调用PPO learn、不写权重或训练目录。速度键和头部指令与play共用处理，复位课程步数恢复为checkpoint阶段，
  观测噪声及interval events关闭；摩擦/材料/求解参数仍取训练CLI配置，摩擦默认值见最新接手摘要。
  不带该开关仍为正常PPO训练；`--command`对普通训练不控制其指令采样，仅用于play及此诊断。
  19项定向测试通过、Ruff/空白检查通过。真实SynReal4环境半垫、vx=0.3、100帧对照均完成且无复位；
  配置相同、首帧动作逐位相同，但首帧物理步进后q差约2.3e-5并逐渐放大，不能宣称轨迹逐位一致。
  重复运行同一play也首帧动作相同、q差约2.4e-5，100帧最大动作差约0.68（train/play约0.55），
  所以这份跨进程轨迹差异不能单独归因于训练入口；未定位原生重复运行差异的内部原因。
  证据`.validation/microduck_train_play_diagnostic/`，训练/首次play与重复play采样均保留。
  此模式用于隔离动作探索/更新/指令差异，不等于已证明用户原抖动的唯一原因；未启动长训练。

- **2026-10-29 按用户要求关闭默认脚底摩擦随机化。**
  `env.py::EnvironmentConfig.randomize_foot_friction=False`，训练和回放默认均关闭。
  显式CLI开启或恢复checkpoint环境仍可能覆盖默认值；重启训练后生效。

- **2026-10-29 用户为降低开销将软垫改为1.2×1.2 m，边缘10 cm复位。**
  `env.py::PAD_CONFIG`现为`size=(1.2,1.2,0.06)`、`cells=(24,24,3)`，保留5 cm水平间距和嵌入几何。
  `MixedTerrain`边缘内缩改为0.10 m，软垫组根节点|x|或|y|>0.50 m时timeout并重置中心；训练/play共用。
  单垫10368个四面体，比上一版2×2 m的28800减少64%，不承诺时间同比下降。
  材料、求解参数未改；运行中的进程不热更新，重启后生效。此条覆盖下文旧尺寸及边界值。

- **2026-10-29 按用户最终指定将软垫扩大为2×2 m。** `env.py::PAD_CONFIG.size=(2.0,2.0,0.06)`、
  `cells=(40,40,3)`，保持5 cm水平网格间距、厚度6 cm及嵌入深度，材料不变。
  默认中心出生，四边内缩0.12 m复位阈值随尺寸变为|x|或|y|>0.88 m；凹槽自动随尺寸扩大。
  单垫四面体数量由6912增为28800（约4.17倍），这会增加软体计算量，不承诺原训练速度。
  `SoftPadConfig`历史默认未改；显式恢复旧checkpoint环境仍使用记录的旧尺寸。
  凹槽测试按实际尺寸选取边界及外部查询点；仅几何/配置回归，不宣称大网格长期动力学稳定。
  此条尺寸覆盖下文旧1.2×0.8 m及0.48/0.28 m阈值；新启动的训练/回放生效。

- **2026-10-29 按用户要求恢复软垫边缘复位。** 此条覆盖下文“跨边缘连续行走”：嵌入几何保持原样，
  `MixedTerrain.out_of_bounds`恢复垫边内缩0.12 m的回合截断，默认垫面组根节点超过|x|=0.48或|y|=0.28 m
  就timeout并在正常reset路径回到中心，不作为摔倒失败；纯平地组仍以2.5 m场景边界为限。
  元数据恢复`pad_edge_timeout_margin=0.12`、`boundary_curriculum=surface_locomotion_before_edge_crossing`。
  42项定向回归通过，真实SynReal四环境半垫测试确认四个边界仅重置软垫组且XY回到0；
  证据`.validation/microduck_edge_reset/`。未改用户当前5/10求解设置或材料，未启动长训练。
  新启动的训练及普通策略play共用该边界逻辑；已打开的进程需重启。

- **2026-10-29 齐平软垫续训的频繁压缩保护排查。** 用户64环境运行的首个记录为控制步110、world48、
  体积比0.2467，属于局部压缩，不是已证实的翻转/CUDA故障。5/15、64环境真实PPO短测在第200次控制步
  因100步窗口8次压缩失败停训（32个软垫世界，8/3200=0.25%，阈值0.1%），0次翻转。
  失败单元主要位于垫面内部；只禁用软垫与凹槽地板碰撞仍在200步停训、9次压缩，
  只将dt减半到0.0025/8子步仍在200步停训、16次压缩，因此未采用这两个实验修改。
  单独将材料Lamé lambda从2000提高至10000，64环境5/15完成50轮/1200控制步，仅2次压缩保护、0次翻转。
  这支持提高抗体积压缩能力，但不定位闭源FEM内部唯一原因，也不证明长期收敛或真实海绵标定。
  `env.py::PAD_CONFIG.lame_lambda=10000`；剪切模量2000、接触刚度5000、网格和保护阈值保留。
  此改动会减少压缩、改变材料特性，不应称为完全相同的海绵。`SoftPadConfig`历史默认及旧元数据保持原值。
  用户在排查期间将线性迭代改为50，现保留5/50；用户自行启动的PID1204040未由本轮停止或重启，
  新参数只影响新进程，运行状态必须再次查询。测试期间GPU有争用，耗时不作为性能对照。
  `SoftPadGuard.finish_step`新增`guard_stop.json`，在清零窗口前记录实际失败数、分母、翻转数和原因；
  异常文本包含比例和路径，阈值及终止行为未放宽。证据与实验脚本在`.validation/microduck_inset_failure/`。
  最终256环境5/50完成12轮/73728条环境转移，0次压缩保护、0次翻转；64环境50轮的上述结果为5/15。
  完整回归195 passed/9 skipped；新增单步报错测试后guard定向9 passed；材料修改后配置/物理/迁移/凹槽
  定向32 passed/1 skipped；Ruff及差异空白检查通过。短测通过不代表30000轮均不会触发保护。
  未启动正式长训练；GPU短测为临时内存PPO更新，不输出可续训权重。

- **2026-10-29 齐平软垫允许跨边缘连续行走。** 用户要求离开垫面不复位：`MixedTerrain.out_of_bounds`
  在 embedded 场景不再按软垫边缘或靠近软垫判断 timeout，允许出垫与重新进入；只保留整个场景距中心 2.5 m
  的外边界截断，以防走出有限硬地板。回合时长、跌倒和变形故障终止不变，真正重置后软垫组仍回到中心。
  元数据记录 `pad_edge_timeout_margin=null`、`boundary_curriculum=cross_surface_locomotion`、
  `scene_timeout_radius=2.5`；历史凸起软垫仍保留原边缘规则。Terrain/pad 与 flat 日志仍按出生组统计，
  pad 组离垫后的硬地面步数也会计入该组，不能将出生比例当成实际软接触采样比例。
  定向回归36 passed，Ruff与空白检查通过；证据 `.validation/microduck_pad_crossing/`。
  真实GPU4环境将状态放在垫面四边外再步进，均不done/timeout，回到中心仍连续；2.6 m外边界正常timeout。
  这是边界/状态步进验证，不是策略自主走过边缘的成功率评估；未启动训练。

- **2026-10-29 默认软垫已改为嵌入式齐平场景。** 用户要求垫面不高于地面：
  `env.py::PAD_CONFIG.base_z=-0.06`，厚度仍 0.06 m，未变形顶面 z=0；固定底面、材料、接触刚度 5000 和 5/15 保留。
  不能只移动粒子：旧共享 z=0 地面会挡住软垫。因此嵌入模式按世界构建有限刚性地板：纯平地组一整块，
  软垫组四块周边地板与低于底面碰撞皮肤的底板，软垫上方没有 z=0 刚性碰撞层。每侧留 2×particle_radius
  （当前 3 mm）碰撞间隙，防止静态侧壁预压缩；该近似不是无缝或真实泡沫校准模型。
  地板边界至少覆盖 ±3 m，回合半径 2.5 m 以内；可视化世界间距加大避免邻居地板遮住洞口。
  顶面查询在凹槽内以凹槽底初始化，允许测到负 z 下陷；其他地方仍返回地面0。传感器按最终 shape/body
  归属识别机器人与地板，不再假设所有世界形状数相同。刚性接触材料已逐块对照旧地面保持一致。
  SoftPadConfig 历史默认仍为 base_z=0.0015，显式恢复旧 checkpoint 的材料会保留旧凸起垫＋共享平面。
  新默认训练/play/独立 soft_pad_scene 共用嵌入几何；固定分组、中心出生和边缘 timeout 未改，不代表已实现上下垫课程。
  验证包括负高度查询、中心没有硬盖板的真实碰撞检查、旧材料恢复、CPU/配置回归，以及4环境GPU运行和中心重置。
  原平地13500策略16环境180步（2880 transitions）短测数值正常，0次翻转，1次过度压缩保护、5次episode重置；
  不宣称站立后倾或长期稳定性已解决。OpenGL headless渲染已实际查看，灰色地面与黄色垫面齐平；
  `.validation/microduck_inset_pad/preview.png` 为预览。其余证据、修改前源码及报告在同目录；未启动训练。
  最终回归 **190 passed / 9 skipped**（55.86 s），Ruff 与空白检查通过；全平地2环境、全软垫1环境GPU短测也通过。

- **2026-10-29 固定纯平地/软垫分组与垫面中心重置已实现。**
  新默认在 `env.py::EnvironmentConfig.pad_sampling="fixed_groups_center"`，`pad_probability` 现在表示软垫环境比例：
  `floor(num_envs * probability + 0.5)` 个末尾环境创建软垫，前面的环境仅含机器人和硬地面；例如 256×0.5 为 128/128。
  两组共享策略、刚体/关节排列和奖励；软垫组每次 reset 将根节点 XY 设为垫面中心 (0,0)，其他姿态、关节和指令随机化保留。
  纯平地组不再在不存在的垫边触发 timeout；软垫组仍按边缘边界终止，未新增上下垫课程。
  `MixedTerrain` 对缺少软体的世界使用显式 -1 查询哨兵，所有状态写入都过滤哨兵；高度查询、体积保护、回滚、
  局部重置与渲染偏移已适配不同粒子数。保护故障率分母改用实际有垫环境数，避免平地组稀释异常率。
  所有机器人仍创建 SDF：当前全表面软接触管线会验证所有参与的网格，不能只给软垫组创建 SDF 后关闭精确接触。
  元数据保存 sampling、pad_world_count、pad_world_ids；`--use-checkpoint-environment` 对旧记录恢复
  `independent_per_episode` 及旧随机出生行为；严格 resume 检查分组方式，跨配方继续使用 init-from。
  新旧分组都保留 env.py 当前物理设置和软垫材料。原 PID 1150168 的用户 256 环境训练本轮未停止/重启，
  内存中继续运行旧场景；新分组仅对重新启动的进程生效。运行中源码已经改变，后续切勿把磁盘源码当成该运行的精确版本。
  验证：完整回归 185 passed / 9 skipped；最后增加稀疏垫渲染测试后针对性 28 passed；Ruff/空白检查另见证据。
  GPU 真正求解通过 4 环境半垫、4 环境全平地、1 环境全垫，各 24 控制步，并检查 reset 后中心位置。
  256 环境固定半垫实际构建 884736 个四面体，原全垫为 1769472；不宣称步态/长期稳定性已经改善。
  性能限制：异构世界无法使用当前原生适配层的同构共享模板；用户训练同时占用 GPU，短测速度不能当独占性能。
  256 环境固定 HOME 动作短测：预热 24 步后测 3×24 步，固定半垫中位 32.46 s，旧每环境一垫中位 48.45 s，
  本次条件下采样耗时减少约 33%。两次顺序运行且与用户训练共享 GPU，出生方式也不同，不能推断独占 PPO 精确耗时。
  真实 SynReal 4 环境半垫强制翻转一个软垫世界后，保护回滚验证所有粒子/刚体状态恢复且邻居逐位不变，中心重置正常。
  证据与修改前源码：`.validation/microduck_fixed_groups/`，含 groups/guard/render 测试、GPU smoke 与短时 benchmark。

- **2026-10-29 混合软垫行走奖励优化已接入，尚未用新奖励训练策略。**
  用户先要求停止训练，原 PID 1101789 已 SIGINT 退出；最后完成日志编号 115（116 轮），
  最后保存 `2026-10-29_13-01-07_044272_20261029_130049_contact_fixed256_5_50/model_100.pt`。
  本轮仅做代码优化与无梯度策略回放，不恢复长训练。磁盘 `env.py` 当前为用户修改后的 5/15、
  `contact_stiffness=5000`；本轮未改物理、材料、PPO、观测维度或指令采样。
  `configs/official_snapshot.json → tasks.velocity.rewards` 集中维护两项 local 奖励：
  `foot_slip` 权重仍 -0.1，函数改为 `feet_surface_slip`；软接触使用重心插值粒子速度与刚体接触点速度
  （含 COM/角速度修正）的相对切向速度平方，每只脚按有效接触数平均，避免相反滑动方向抵消。
  混合边缘按软/硬接触计数混合，硬地面保留原世界 XY 脚速；纯平地任务直接调用原公式。
  新增 `command_progress=-0.25`，仅在 mixed_soft_pad 启用，两类出生环境都适用：
  沿有符号平移指令方向计算不足量，容差 0.02 m/s，归一化速度下限 0.05 m/s，最大不足比 2；
  指令速度须 >0.03 m/s、倾角 ≤45°；指令变化/重置后先免罚 0.3 s，再用 0.3 s 线性渐入。
  达标和超速不获得额外正奖励，超速/侧向误差仍交给原速度跟踪；零指令和纯转向不启用推进惩罚。
  新公式在 `locomotion_rewards.py`，已加入配方指纹；局部导出保留两项配置，严格 resume 不放宽。
  新日志增加 `Metrics/world_slip_velocity_mean`、`Metrics/pad_relative_slip_velocity_mean` 和
  `Metrics/command_progress_deficit`，便于分别判断移动、相对滑动与推进不足。
  无梯度 CUDA 回放使用原平地 model_13500、16 环境、180 控制步（2880 transitions），1910 个转移存在软垫脚接触；
  同状态未加权脚滑成本旧均值 0.01956、新均值 0.008336，推进项实际生效，0 次变形保护/翻转，3 次普通 episode 重置。
  该成本差异同时包含接触点/切向定义与表面速度修正，不能全归因于垫面运动，更不代表策略能力提高。
  测试覆盖 CPU/CUDA 接触插值、COM 偏移、法向运动排除、相反滑动不抵消、零/转向/倾倒门控、换向缓冲、
  局部重置、纯平地公式等价及导出保留。证据与修改前备份：`.validation/microduck_reward_optimization/`。
  最终完整回归 **178 passed / 9 skipped**（56.43 s），所改 Python 的 Ruff 和 `git diff --check` 通过。
  最终步态收益、遗忘程度与长时间稳定性尚需固定物理设置的训练对照验证。

- **2026-10-29 已修正导致 256 环境快速停训的软接触刚度，并重新启动训练。**
  保持用户指定 5/50、dt=0.005×4，未改变软垫网格、密度、Lamé/剪切模量、阻尼及变形保护阈值。
  新参数为 `env.py::PAD_CONFIG.contact_stiffness=5000`，原软接触罚刚度为 50000；该参数影响刚体与
  软垫接触的柔顺性，不是把软垫变硬，也不是修复闭源 FEM 后宣称永不翻转。
  完整 256 世界故障子步输入、接触和 Python 驱动历史已保存在
  `.validation/microduck_training_launch/full_substep.pt`。重新初始化求解器后的重复基线仍在 world 178
  翻转（最小体积比约 -0.21）；缩小步长、块预条件器及法线归一化未消除该故障。
  单独将接触刚度降到 5000 后，该世界体积比约 0.81，整批无翻转。
  重放存在原生内部历史与浮点差异，不是逐位完全复现；不能据此定位闭源代码的唯一内部缺陷。
  排除刚体代理粒子后，故障子步脚—垫接触穿透 p95 从约 0.21 mm 变为 0.53 mm，最大约 6 mm（旧值约 6.3 mm）。
  256 环境单因素 PPO 短测正常完成 8 轮 / 49152 条环境步：0 次翻转、3 次局部压缩保护，未触发停训。
  原有保护、故障重置及失败惩罚保留；长时稳定性和步态质量仍需持续观察。

  参数通过 SoftPadConfig 写入新元数据；`--use-checkpoint-environment` 可恢复旧独立 `soft_contact.ke`
  字段，缺失时采用历史 50000。严格 resume 会拒绝接触刚度不一致，不修改任何历史 config.json。
  已更新配置接线、历史恢复、严格续训和参数合法性测试，并移除测试中对已被用户调整的默认 15 次的硬编码。
  完整回归 **172 passed / 9 skipped**；Ruff 与差异空白检查通过，跳过项不作为 GPU 验证。

  正式训练 PID **1101789**（本条写入时完成 7 轮、约 23.3 秒/轮、0 次翻转、2 次局部压缩保护），
  256 环境、计划 10000 轮、50% 垫面出生，从原平地 `model_13500.pt` 重新初始化；操作前须重新核实 PID。
  目录：`logs/rsl_rl/microduck_synreal/2026-10-29_13-01-07_044272_20261029_130049_contact_fixed256_5_50/`。
  日志快捷路径：`.validation/microduck_training_launch/train_fixed.log`；进程/命令信息：`train_fixed.json`；
  证据汇总：`contact_fix_report.json`（均位于同一 `.validation/microduck_training_launch/` 目录）。
  启动包装仅负责正常认证和对生成的 Git diff 日志去除凭据，训练执行正式 `train.main()`，无物理 monkeypatch。

- **2026-10-29 12:15 左右，按用户要求实跑 1024→256 环境、非线性 5 / 线性 50。**
  用户明确要求从平地 `model_13500.pt` 开始混合海绵垫训练；已将 `env.py` 的线性默认从 15 改成 50，
  实际运行 config.json 确认 5/50、dt=0.005×4、pad_probability=0.5，使用已更新的仓库原生二进制。
  1024 短测完成两轮，80.51/79.90 秒、305/307 条环境步每秒，因速度慢主动 SIGINT 停止。
  改用 256，从同一原平地权重新启动计划 10000 轮；前四轮 23.60/23.44/23.38/23.29 秒，
  约 262 条环境步每秒。每轮更快但采样量也减为四分之一，不能称总采样吞吐更高。
  **256 训练在完成四轮后因 `Too frequent soft-pad failures` 自动停止，目前这两次进程均已退出。**
  诊断：控制步 20、world 8 的最小体积比 0.1399；控制步 38、world 58 的最小体积比 -0.2506，
  保存的状态数值有限，但确有软垫单元翻转。没有放宽保护或修改垫材质，也没有继续自动重启训练。
  256 运行目录：`logs/rsl_rl/microduck_synreal/2026-10-29_12-11-55_714832_20261029_121140_train256_5_50/`，
  保留 `model_0.pt`、config.json、TensorBoard 及 diagnostics；后续三轮更新没有到达每 100 轮保存点。
  启动记录、进程 PID、原生哈希、完整日志和结果汇总见 `.validation/microduck_training_launch/result.json`。
  当前阻塞是此配置下的软垫变形失败，不能宣称训练正在后台继续或把 CUDA 依赖修复视为该问题的修复。

- **2026-10-29 修复混合软垫策略回放右键拖拽只有线、没有施力。**
  `step_playback()` 的混合/完整步进分支此前只关闭训练 interval events，未向 `step()` 传递 viewer forces，
  因而每个子步清空外力后直接推进求解器；纯平地快速回放和独立交互 scene 原有施力路径正常。
  现完整回放显式启用 `apply_viewer_forces`，在每子步清力后调用 viewer，再进入求解器。
  该 Python 内部参数独立于训练自动推扰，未新增 CLI 开关；普通训练仍保持原行为，回放自动推扰仍关闭。
  新回归在修复前复现“快速回放通过、混合路由外力为零”；修复后共享物理定向回归 **12 passed**，
  验证每子步外力送达、释放后清零、普通训练不继承鼠标力及 interval events 契约；Ruff/差异空白检查通过。
  测试用真实 CPU MuJoCo 步进验证调用链，混合路由在进入动力学前切回刚体模式；不是软垫动力学或 GL 手动验收。
  已打开的回放进程须重启后生效。

- **2026-10-29 已删除完整步进 CLI 开关，并定位混合场景平地退化的原生二进制版本问题。**
  按用户要求删除该命令行参数、Python 分支参数及工作台复选框/命令生成；软垫仍自动执行必要的
  传感器与变形保护，训练保留推扰、所有策略回放关闭训练 interval events。
  用户提供正常认证后完成真实 SynReal 对照，凭据仅用于临时进程，不写入文件或日志。
  在 30 世界、原平地 `model_13500.pt`、5/15、dt=0.005、固定摩擦 1 下，构建核对确认
  两场景机器人质量/惯量、PD、刚性接触材质一致；CPU 20 状态的两条 Actor 观测路径差值为零。
  所有机器人在平地时，12 秒实际回放的纯平地/混合场景速度接近；让后 15 只踩软垫后，
  前 15 只平地鸭子也明显减速，且在首次重置之前已经出现。固定动作对照中同样复现，
  因而不能再将其只归因于微调遗忘、可视化或随机推扰。

  实际加载 `.venv/lib/python3.12/site-packages/style3d_sim/_core.cpython-312-x86_64-linux-gnu.so`
  的 SHA256 原为 `4c223854a5a554140ed6cf0e99d5687aa5846bbe49465a85a5819079f3db34e6`，
  与当前仓库二进制 `3fded7227b402d76e763d772f8c85aebae0b59a01b6688e875a31807b79b9838` 不同。
  仓库 e74e136 将二进制子模块更新到 23fb797，提交说明为逐世界独立 CG 系数；
  未获取原生源码，不能仅凭提交说明确定原安装文件的确切内部缺陷或来源。
  隔离进程加载仓库二进制后，12 秒实际回放（站立 2 秒，再 vx=0.3，后 9 秒统计）结果：

  | 原生版本 | 纯平地前 15 世界平均 vx | 半数踩垫时前 15 平地世界平均 vx |
  | --- | ---: | ---: |
  | 原安装版本 | 0.1335 m/s | 0.0801 m/s |
  | 仓库版本 | 0.1837 m/s | 0.1837 m/s |

  两种版本各自对照共享实际出生姿态、偏差/延迟初始化、权重、指令和求解参数；平地组没有重置。
  软垫组有重置，不能据此宣称软垫步态稳定或速度跟踪达到 0.3 m/s。
  相同动作的无重置对照也支持原版本存在明显跨世界影响，仓库版本消除了所测的严重减速/后退。
  已将旧扩展备份后原子替换为仓库版本，新进程导入哈希一致；需重启训练/回放/工作台子进程，
  正在运行的旧进程不会自动更新。未修改旧 checkpoint、历史 config.json 或启动长训练。
  提高线性迭代到 50 的旧版本单因素对照反而出现平地倒地，不作为推荐修复；参数保持 5/15。
  证据与独立诊断脚本在 `.validation/microduck_flat_mixed_comparison/`：`model_audit.json`、
  `isolation_30.json` / `isolation_repo_30.json`、`playback_half_v03.json` / `playback_half_repo_v03.json`、
  `binary_replacement.json`（含原扩展备份路径）。这些为本机证据，默认不跟踪 Git。
  二进制对照汇总为 `binary_comparison_summary.json`。替换后完整 MicroDuck 回归 **162 passed / 9 skipped**；
  真实 SynReal 对照为上述独立诊断，未通过测试跳过项来冒充 GPU 验证。
  当前最优先重新评价旧平地与混合权重；历史混合训练是在什么原生版本下运行，需保留证据后判断。

- **2026-10-29 新增本地中文训练工作台。** 入口 `python -m examples.microduck.ui`，默认地址
  `http://127.0.0.1:8765`。训练、训练效果预览、模型评估和 TensorBoard 由独立子进程管理。
  不修改环境、奖励、指纹或原训练入口；环境默认值仍由 `env.py` 提供。操作说明见下方 UI 小节。

- **2026-10-29 同一平地权重在混合场景退化：已修正一项回放协议差异，动力学原因仍待实测。**
  用户观察 `model_13500.pt` 在混合场景的平地上也不如原平地场景。源码确认纯平地快速回放不执行
  interval events，而混合回放委托完整 `step()`，此前会每 3～6 秒触发一次水平速度随机推扰
  （Velocity 当前 x/y 各为 ±0.3 m/s 的增量），因此不能直接把视觉差异归因于微调遗忘或软垫物理。
  `step_playback()` 现对完整回放显式传入 `apply_interval_events=False`；`step()` 训练默认仍开启。
  此前完整步进回放同样关闭训练 interval events；对应 CLI 开关现已按用户要求删除。
  相关回归 39 项通过，并补充验证回放后继续训练仍执行推扰；Ruff 与差异空白检查通过。
  未执行新的真实 SynReal 动力学对照，不能宣称这是全部根因或步态已恢复。
  混合场景仍有边界 timeout、额外低高度终止、软接触管线与不同出生随机数消费等差异。
  下一步固定原权重、`--pad-probability 0`、相同世界数/指令/摩擦，比较两 profile；
  更严格物理对照还须统一实际出生状态和动作序列，并在任何重置前记录轨迹。

- **2026-10-29 多世界软垫显示错位已修复。** 用户的 30 世界回放中，鸭子分散显示但软垫叠在原点，
  因而部分鸭子看似浮空。Newton 1.5 的刚体渲染应用 world_offsets，三角网格渲染未应用该偏移。
  `MicroDuckEnv.render()` 现在通过 `MixedTerrain.display_state()` 使用独立粒子渲染缓冲区，
  让软垫与机器人采用相同的世界偏移；不修改物理状态、碰撞或出生高度。
  混合场景显示间距改为垫宽 + 2 m、垫长 + 1 m（当前 3.2 × 1.8 m），避免垫子重叠。
  定向回归 19 项通过，包含 1/4/30 世界、不同显示间距、逐世界变形及两份物理状态不变检查；Ruff 通过。
  CUDA/OpenGL 四世界初始布局截图为 `.validation/microduck_terrain_rendering/four_worlds.png`，
  仅验证构建和显示，未执行新的 SynReal 动力学回放。修改涉及指纹文件，旧权重诊断回放仍需
  `--allow-recipe-mismatch`，严格续训仍须保留原配方。运行中的窗口须重启才能使用修复。

- **2026-10-29 用户进一步要求：迭代次数和环境参数集中编辑 `env.py`，不新增配置文件。**
  `PhysicsConfig` 管理策略求解次数和物理步长，`PhysicsConfig.home()` 管理无策略 HOME 模式；
  `PAD_CONFIG` 管理软垫尺寸、网格、密度、Lamé 参数、阻尼和固定底面；
  `EnvironmentConfig` 管理训练/回放环境数、出生概率、交互出生位置/间隙、训练与回放摩擦随机化。
  `ENVIRONMENT_CONFIGS` 中 `official` 对应原平地/任务地形，`mixed_soft_pad` 对应平地＋软垫；
  两者默认共用 PhysicsConfig 和 PAD_CONFIG，可在同一表内显式覆盖。
  两份 rtx4090 训练 JSON 只选择 `environment_profile`，已移除重复的环境数、地形、概率和摩擦字段。
  这些字段若再次写入训练 JSON 会给出迁移提示，临时覆盖仍可直接使用 CLI。

  **当前优先级：显式 CLI > env.py；旧权重 config.json 不再默认覆盖求解参数和软垫材质。**
  普通 `train`、`--init-from`、`.pt play`、`soft_pad_scene --checkpoint` 共用当前设置；策略默认仍为 5/15。
  回放会从 checkpoint 的地形种类选择默认 profile，但参数来自 env.py；可显式 `--environment-profile` 选择。
  `--use-checkpoint-environment` 才恢复历史物理/地形参数（交互 scene 始终保留一个垫子，恢复已保存的垫材质）。
  `--resume` 仍检查配方指纹，并拒绝物理、地形、垫材质、出生概率或已记录的摩擦设置不一致；
  要改变环境使用 `--init-from`。切勿修改历史日志的 config.json 冒充原训练环境。
  启动会打印来源、实际求解参数、环境数量、地形/出生信息和垫材质，新训练元数据记录实际参数。
  `soft_pad_scene` 仍限制 1 个交互世界，多世界使用 `play --num-envs N`；设置大于 1 会提示正确入口。
  修改源码后须重启对应进程，不会热更新正在运行的训练或回放。

  本轮验证：完整测试 **136 通过、9 跳过**；Ruff、128 列检查和 `git diff --check` 通过。
  新测试覆盖当前参数优先、显式历史恢复、严格续训不一致拒绝、训练/两种回放共用参数、CLI 覆盖及历史元数据不改写。
  实际 CUDA 软垫预览 1 帧及状态检查通过，日志 `.validation/env_settings_preview.log`。
  用户混合 `model_2900.pt` 的两种回放参数解析均为 5/15，混合训练解析为 1024 世界、pad_probability=0.5。
  未执行需登录的 SynReal 动力学回放/PPO；以上证明配置接线和构建，不是步态恢复验证。

- **2026-10-29 用户要求统一环境：`env.py::MicroDuckEnv` 是唯一仿真环境实现。**
  `train.py`、`play.py` 和 `soft_pad_scene.py` 均使用它；机器人/软垫构建、求解器、饱和 PD、物理子步和重置集中管理。
  `soft_pad_scene.SoftPadScene` 现在是持有 `env` 的显示/CLI/键盘包装器，不再创建求解器或自行调用 solver.step。
  `soft_pad_policy.py` 仅加载冻结权重、处理交互指令并推理动作，已移除自己的运动学/PD 实现。
  `play.PolicyPlayback.step` 也委托给环境；共享物理循环为 `MicroDuckEnv.advance_physics()`。
  `soft_pad.py` 仍负责软垫几何与材质，`mixed_terrain.py` 负责混合出生、变形高度和局部故障隔离；它们不是第二套 env。

  参数入口为 **`env.py::PhysicsConfig`**：保留用户本轮前已修改的策略默认非线性 **5** / 线性 **15**、
  primal=1、block=False、空气密度=1.225、dt=0.005、4 子步；新训练与迁移训练共用这套默认值。
  HOME 的明确覆盖在 `PhysicsConfig.home()`：5/30、primal=3、block=True、空气密度=0、dt=1/480、8 子步。
  初次统一实现曾默认恢复 checkpoint 的求解参数；此行为已由上条用户要求更新。
  现在普通回放使用 env.py，显式 `--use-checkpoint-environment` 才恢复记录的 50 次等历史设置。
  `--init-from` 是迁移初始化，使用当前新任务默认动力学。训练/查看启动日志都会明确打印实际步长和迭代次数。

  训练保持每次 reset 50% 平地/50% 垫面抽样、原奖励、故障终止与局部重置；交互模式保持中心/垫前固定出生、
  R 手动重置、不自动隐藏摔倒、鼠标施力和原画面。交互单世界保留原始拓扑；批量训练复制同一个机器人＋垫子模板。
  当前运行进程不会热更新，未停止或重启用户正在运行的训练。重构改变配方指纹，旧模型回放仍需显式
  `--allow-recipe-mismatch`；严格续训检查未放宽。旧命令入口保持可用。

  验证：完整回归 128 项通过、9 项需登录 GPU 用例跳过，随后新增单世界 world=-1 适配测试通过（合计 129 项）；
  Ruff 与 `git diff --check` 通过。CPU MuJoCo 新旧环境连续 20 步的状态、观测、奖励、done 逐位一致；
  训练与 fast playback 的物理轨迹也逐位一致。真实 CUDA 构建的平地/垫面模型及初态与旧实现逐位一致。
  旧 PCG `model_11100.pt` 初始 Actor 观测、动作、饱和目标与手动重置逐位一致；此项没有推进软体动力学。
  `soft_pad_scene --preview --viewer null --num-frames 1 --test` 通过；CPU 1 环境 / 1 轮 PPO、checkpoint 有限性和
  保存参数续训解析通过，目录 `logs/rsl_rl/microduck_mujoco/2026-10-29_08-53-05_218998_unified_env_cpu_smoke/`。
  对照报告保存在 `.validation/microduck_env_unification/`。本轮没有可复用的 SynReal 登录会话，
  **尚未重跑重构后的真实 SynReal 物理步进或 1024 软垫 PPO**，不能将几何/CPU 验证当作软垫步态验收。

- **2026-10-28 混合训练翻转诊断与隔离：这是训练域约束，不是原生求解器翻转问题已修复。**
  首次正式运行在完成 2 轮 PPO 后，第三次 rollout 因 world 317 的四面体翻转停止。
  使用相同来源模型和 1024 世界复现，故障世界/时间会变化；独立故障前快照捕获 world 634、控制步 26、子步 3，
  最小体积比从 0.2311 降至 -0.2337。机器人当时直立，故障在垫内部、未发生 reset，局部垫节点速度约 4 m/s。
  同一故障前状态提高非线性迭代 5→15 仍翻转；把 5 ms 拆成 2×2.5 ms 或 4×1.25 ms 也未消除。
  这些对照只说明已压缩状态下的局部改参无效，**没有证明所有步长/材料配置均无效，也未定位闭源原生内部缺陷**。
  `NewtonSolver.step` 明确会覆盖输入和输出的粒子位置，不能把 `states[1]` 当成真实上一时刻。
  正确独立快照在 `.validation/microduck_mixed/inversion_state.pt`，对照在 `inversion_dt_probe.json` 等相邻文件。

  `mixed_terrain.SoftPadGuard` 每控制步独立备份粒子（含 affine proxies）、刚体、关节及驱动角历史。
  每物理子步检测 J≤0.25、翻转/非有限位置或速度；只恢复故障世界的两个缓冲，剩余子步继续隔离，步末失败重置。
  不让异常几何进入传感器/观测，不在同一个动作中间开启新 episode；其他世界状态和驱动历史按位保持。
  失败帧屏蔽正常奖励，使用集中配置 `velocity.rewards.soft_pad_failure`（local，weight=-50，乘 dt 后为 -1），
  done=true、time_out=false，不让 RSL-RL 添加 timeout bootstrap。其他物理爆炸与碰撞容量溢出继续 fail-fast。
  单步失败超过 max(4, 5% 世界)，或 100 控制步内至少 8 次且占转换总数 >0.1%，停止训练。
  训练目录 `diagnostics/` 每类保存首次故障快照；TensorBoard 分别记录危险变形及翻转/非有限的次数、比例。
  所有材质、网格和物理步长保留；J=0.25 是保守训练边界，**不是已标定的泡沫压缩极限**。
  真实 SynReal 4 世界强制翻转测试通过：两缓冲恢复、其他世界/驱动历史按位不变、失败奖励/终止正确，
  有效 Actor/Critic 观测且重置后再运行 40 控制步通过；证据 `.validation/microduck_mixed/guard_gpu_validation.json`。
  本轮完整回归 **119 项通过、9 项默认 GPU 用例跳过**；Ruff 和 `git diff --check` 通过。
  已从原平地 `model_13500.pt` 重新启动 1024 环境、10000 轮迁移训练，运行目录：
  `logs/rsl_rl/microduck_synreal/2026-10-28_21-09-07_508979_velocity_flat_pad_finetune_guarded/`。
  登录 worker PID 678612；保持本机登录终端打开。日志继续使用
  `.validation/microduck_mixed/training_1024.log`，首次失败日志另存 `training_1024_inversion.log`。
  `model_0.pt` 已验证参数有限、17 个模型张量发生更新、优化器状态非空；`source/` 32 个源文件及哈希已保存核对。
  截至本轮交接，连续完成 **10 轮 / 240 控制步 / 245760 环境样本**，两个 100 步异常率窗口均通过。
  累计隔离 90 次危险变形/数值异常，其中 5 次为翻转或非有限状态；整体发生率约 0.0366%，全部已记录为失败。
  所有已写入 TensorBoard 的 scalar 有限，约 57～59 秒/轮，GPU 总显存约 11.5 GiB；训练继续运行。
  第 10 轮 pad 帧占比约 13.8%，平地/软垫 upright 均值约 0.9982/0.9547、速度误差约 0.1617/0.2653 m/s。
  每次 reset 的 50% 抽样不保证平衡采样帧：软垫 episode 更短，后续应监测软垫数据占比；本次未改抽样规则。
  持续指标检查脚本为 `.validation/microduck_mixed/inspect_training.py`，输出 `guarded_training_metrics.json`；
  从每轮精确 failure fraction × 24576 累加还原故障次数，控制台的 count 字段是 rollout 内均值，不能当作整数累计值。
  后续须看异常率及两种地面的独立运动表现，不能只因训练不再退出便宣称软垫稳定或策略收敛。

- **2026-10-28 新目标实现：SynReal 平地/海绵垫混合训练，1024 环境，从用户提供的平地模型迁移。**
  `train --terrain mixed_soft_pad --init-from <checkpoint 或运行目录>` 已实现；固定来源为
  `logs/rsl_rl/microduck_synreal/2026-10-28_10-04-52_798619_velocity_PCG/model_13500.pt`，
  SHA256 `3935882fdd57b424a10474f6f07727e815840d2d1feca4966d35099e5f23c4f4`。
  `--init-from` 加载 Actor/Critic、已学习的动作标准差及两套观测归一化，优化器和新任务轮数重置。
  这是显式迁移，不放宽 `--resume` 的同配方检查；新运行元数据记录来源、哈希、来源指纹、地形及求解参数。
  `configs/rtx4090_mixed_soft_pad.json` 为新配置：1024 环境、10000 轮、每轮 24 步、每 100 轮保存，
  固定学习率 1e-4、entropy 0.001；策略周期 0.02 s、物理 dt=0.005 s。
  运行目录保留在 `logs/rsl_rl/microduck_synreal/`，不覆盖来源权重。

  每个独立物理世界复制原 1.2×0.8×0.06 m / 24×16×3 四面体垫与机器人，所有世界共享刚性无限平面。
  每次 episode 独立以 50% 概率选择垫面出生，否则在垫外平地出生；出生比例不等于实际采样帧比例。
  `mixed_terrain.py` 根据 finalize 后的拓扑恢复垫子索引，按环境重置两份粒子位置/速度/力，
  原有选择性 FK/affine proxy 同步负责机器人与驱动历史，关节目标也按环境重置。
  `sensors.py` 合并刚性接触及软体顶点/边/面重心接触，过滤正间隙、不同世界与 ABD proxy；
  离地高度和躯干支撑高度来自当前变形的顶面三角形。每子步检查正体积与接触容量。
  TensorBoard 新增 `Terrain/flat/*`、`Terrain/pad/*`、垫面出生比例和最小体积比。
  第一阶段在垫边内 12 cm 或平地接近垫边时以 timeout 重置，学习两种支撑面的运动，**尚不训练跨越 6 cm 垫边**。
  同一策略/观测维度同时用于两种地面；共享 `play --checkpoint` 自动恢复保存的混合地形，走完整环境步。
  固定 RSL-RL 2.3 的 `learn()` 首帧漏归一化通过 `ppo.InitialObservationEnv` 局部适配，覆盖迁移及混合任务续训。

  **本轮离线验证范围：** 最终完整回归 **112 项通过、9 项需登录 GPU 测试跳过**；
  新增混合地形及 checkpoint 测试覆盖世界隔离、变形高度、软接触记录、翻转检测、迁移权重/归一化及真实 PPO 首帧。
  CPU MuJoCo 1 环境 / 1 轮 PPO 入口短测通过，并核对保存元数据保留 ActorCritic/PPO 类名及续训配置可加载；
  产物在 `logs/rsl_rl/microduck_mujoco/2026-10-28_18-24-23_673235_mixed_integration_cpu/`，
  不是软垫动力学验证。Ruff（保留原有 PLC0414 导出别名例外）与 `git diff --check` 通过。
  4 世界 GPU 网格/碰撞构建通过；1024 世界实际构建通过，1802240 粒子、7077888 四面体，
  构建约 39.7 s，GPU 总占用约 5461 MiB（含桌面，**不包含原生求解器/PPO**）。
  该规模探针使用默认刚性接触容量 10240；实际混合训练已显式预留 131072 并检查溢出。
  证据在 `.validation/microduck_mixed/geometry_1024.json`、`source_audit.json` 及相邻日志。
  **2026-10-28 20:21 首次启动正式 1024 环境 SynReal 混合迁移训练（随后因翻转停止，见上条）。** 用户在本机终端完成登录，
  4 环境真实物理 160 控制步（含重置后再运行）、脚接触信号及双缓冲/驱动历史局部重置隔离通过；
  2 轮 PPO 短训也通过，产物 `2026-10-28_20-17-15_379955_mixed_pad_transfer_smoke/model_1.pt`。
  正式运行目录为
  `logs/rsl_rl/microduck_synreal/2026-10-28_20-20-53_623170_velocity_flat_pad_finetune/`，
  已完成第 0 轮 PPO 并保存 `model_0.pt`，验证模型张量实际更新、状态有限、优化器状态非空。
  首轮共 24576 样本，约 58.1 s / 423 samples/s；垫面出生比例 0.5107，平地/垫面两组均有奖励及跟踪指标，
  首轮未发生失败重置，体积检查逐子步通过；GPU 总占用约 11.3 GiB。以上只是启动证据，不是收敛证明。
  本轮修复临时认证 worker 部分清除 sys.modules 却保留父包属性导致的新旧 TerrainHeightSensor 类冲突；
  完整包重载的离线复现已通过，不改变奖励/物理参数。首次初始化错误日志另存为
  `.validation/microduck_mixed/training_1024_initialization_error.log`。
  当前运行 PID 678612，需保持登录终端打开；原排队调度 PID 683939 已结束，重试通过同一登录 worker 运行。
  状态在 `.validation/microduck_mixed/launch_status.json`，实时训练日志为同目录 `training_1024.log`。
  运行目录 `source/` 保存 32 个 Python/JSON 源文件及 SHA256，包含未跟踪的训练模块；原 checkpoint 不覆盖。
  凭据不得写入交接文档、代码、配置或日志；本轮已移除后来新增的明文凭据段落，训练快照前完成清理。
  旧平地训练 PID 467823 已退出；本轮未向它发送终止/暂停信号。

以下软垫记录属于此前独立场景阶段；“尚未接入 PPO”等表述保留其历史范围，当前代码状态以上条为准。

- **最新未解决问题：鼠标拖拽后脖子、脚出现不符合实物的扭转。** 用户使用 `soft_pad_scene`、PCG `model_11100.pt`、
  `--allow-recipe-mismatch --command 0 0 0`，默认垫面出生，明确反馈在鼠标拖拽后出现。
  已核对 14 个 hinge 都启用并导入有限角度范围，`joint_limit_ke/kd≈2500/100`，适配器传给原生 SynReal；
  例如 hip_roll ±22°、neck_pitch -90°～60°、ankle ±90°、head_yaw ±170°。不是关闭了关节角度限位。
  角度限位由有限刚度的约束力实现，不是每步硬裁剪实际角度；策略端 `HOME + action` 不裁剪目标角，只饱和 PD 力矩。
  当前策略契约关闭自碰撞，肢体之间不具备完整的实体阻挡；单轴角度限位不保证整个连杆姿态符合真实机器人。
  无操作零指令 600 帧 / 12 秒复测未越限，最大轴偏差 3.47°；向前 0.1 m/s 的 600 帧测试出现最大越限 3.03°、
  最大轴偏差 30.20°，不能再仅凭躯干倾角和垫体积宣称关节约束正常。
  复用 Viewer 的真实 Picking 内核，前 100 帧无拖拽，随后拉头部 100 帧，再松开 200 帧：默认参数下
  最大拉力 36.16 N（0.737 kg 整机的 5g），最大角度越限 2.82°、轴偏差 27.21°、关节锚点分离 2.47 mm。
  颈部实际角约 -91°～60°，目标角最低约 -368°；未复现单个关节连续转满 360°，不能混淆目标角和实际角。
  降低 Picking 到 stiffness=1 / damping=0.1 / acceleration=0.5g 后，最大拉力 3.62 N，仍有最大越限 4.21°、
  轴偏差 20.03°，不是有效修复。两条拖拽轨迹在施力前也有差异，不作为严格的单因素改善比例证据。
  诊断 JSON 在 `logs/microduck_soft_pad/diagnostics/joint_pad_zero_before.json`、`joint_pad_forward_before.json`、
  `joint_drag_head_default.json`、`joint_drag_head_gentle.json`；脚本 `joint_audit.py` 保留在同目录，不含凭据。
  本轮仅诊断并更新本文，**没有修改场景、动作、求解或限位参数**，诊断进程已退出。
  后续优先逐子步检查 hinge 轴对齐/锚点误差、连续实际角与驱动力，区分角度限位、连接约束、自碰撞和策略异常响应；
  更改约束后须同时重测原平地策略。不要用简单裁剪动作、打开自碰撞或增加迭代来冒充已验证修复。

- **用户已确认代码关系：软垫是新增的独立场景，没有覆盖原 `env.py`。**
  `train.py` / `play.py` 仍使用 `MicroDuckEnv`；`soft_pad_scene.py::SoftPadScene` 自行构建、步进、显示单个刚柔场景，
  `soft_pad.py` 提供四面体软垫，`soft_pad_policy.py` 加载已有策略做推理。
  当前完成预览、HOME 承重与策略回放，**尚未接入 PPO、批量软体环境、软接触奖励或混合地形训练**。
  最新目标仍是同一场景同时训练平地和局部海绵垫运动；本次“做好交接”只更新文档，没有启动训练或改动 Python。

- **交接时重新核对磁盘：两种模式的迭代次数现在均为非线性 5 / 线性 15。**
  当前代码是 `5 if checkpoint is not None else 5`、`15 if checkpoint is not None else 15`，条件分支结果相同。
  `checkpoint is not None` 表示加载策略，与机器人脚下是平地还是软垫无关；其他模式差异见第 3.1 节。
  此前 GPU 承重/重置测试用的 HOME 配置为 **15 / 30**，不能把那次通过当作当前 HOME **5 / 15** 的验证。
  策略分支仍为 5 / 15，先前的 12 秒平地回放证据对应这一分支。本次未重新执行物理测试。

- **已修正软垫场景中“未碰垫子、在平地先摔倒”的回放配置错误。** 不是用“策略没学过软垫”解释这个现象。
  原场景把机器人和硬地面的刚度/阻尼从 `5e6/10000` 改成 `5e4/100`，物理 dt 从训练的 `0.005` 改成
  `0.0025`，并启用了不同的刚体块预条件器及迭代配置。现在保留原硬接触材质，软接触仍用全局
  `soft_contact_ke/kd/mu=5e4/100/0.8`；策略模式从 checkpoint 读取物理周期（本模型为 50 Hz、4 子步），
  使用原 `play` 的非线性 5 / 线性 15 / primal 1 / 关闭刚体块预条件 / 空气密度 1.225。
  修复验证当时 HOME 用 60 Hz、8 子步、非线性 15 / 线性 30；交接时 HOME 迭代数已改，见上条。
  `--randomize-foot-friction` 复用共享脚底摩擦事件。
  对照固定 `model_11100.pt`、单环境、向前 0.1 m/s，机器人远离垫子：原 `play` 150 帧最大倾角 4.27°；
  旧场景 100 帧已摔倒（最大 174.36°）；完整恢复后的 150 帧最大 3.90°。
  在恢复配置上**只把 dt 改回 0.0025 即再摔倒**；**只开启刚体块预条件器也再失稳**。
  单独恢复旧场景的刚度、阻尼或步长均不足以修好，不能宣称只有一个材质参数是根因；未定位原生求解器内部缺陷。
  最终代码开启脚底摩擦随机化后通过 **600 帧 / 12 秒平地行走**，前进约 0.78 m、最大倾角 4.06°、无重置，
  软垫无翻转。该对照将出生 x 设为 -2 m，避免接触垫边；轨迹在
  `logs/microduck_soft_pad/diagnostics/flat_scene_fixed_600.json`，其余 `flat_*` 为消融证据。
  正常 `--spawn ground`（x=-0.9）也通过 150 帧 / 3 秒检查，最大倾角 4.00°；见 `flat_scene_fixed_spawn.json`。
  修复后垫面出生的策略另测 150 帧 / 3 秒：最大倾角 9.54°、最小体积比 0.6509，无翻转；见 `pad_policy_fixed_150.json`。
  修复版本曾通过 **20 项离线测试及 1 项 GPU 承重/完整重置/重置后再运行测试**，Ruff 和差异空白检查通过。
  GPU 用例为 HOME 15 / 30、0.9 × 0.55 × 0.06 m 小垫，运行 120 帧后重置、再运行 30 帧；不是策略重置测试。
  本次没有训练新策略；平地通过不能作为已学会 6 cm 上垫动作或软垫行走的证据。

- **已修正局部软垫场景的平地显示。** 原来真实碰撞平面的可见性关闭，替代的纯色地面缺少视觉参照，像空背景。
  现在直接显示真实平面，使用浅灰棋盘格并移除替代平面网格；只改显示，不改碰撞、求解或网格数量。
  初始布局渲染已检查，截图为 `logs/microduck_soft_pad/ground_visible.png`（`preview_only=true`，不是新的动力学验证）。

- **局部海绵垫场景已接入 SynReal Velocity checkpoint 回放。** `soft_pad_scene --checkpoint` 接受 `.pt` 文件或运行目录，
  目录按保存轮次选最高的 `model_<轮次>.pt`，启动时输出实际文件。`soft_pad_policy.py` 复用共享回放的 Actor 观测处理，
  恢复 RSL-RL Actor 与保存的观测归一化，保留 61 维观测、14 个 HOME 偏移动作、每子步饱和 PD 和 50 Hz 策略频率。
  Actor 使用推理均值；保留编码器/IMU 偏差和观测延迟，逐帧噪声关闭，与共享 `play.py` 一致。
  策略模式每帧 0.02 秒，子步数默认取自 checkpoint 的 physics_dt；HOME 模式仍为 60 Hz、物理 dt=1/480 秒。
  当前仅接受 SynReal / Velocity / position-PD / 无自碰撞 / `official_asymmetric_pd_v2`；校验增益、力矩上限及策略周期。
  配方指纹不同默认拒绝，诊断回放需显式 `--allow-recipe-mismatch`，不会修改 checkpoint 或共享训练入口的检查。
  此入口保留硬地面回放契约并加入软接触；新软地形仍属于迁移测试。
  本次固定使用 `2026-10-28_10-04-52_798619_velocity_PCG/model_11100.pt`，未重训或更换权重。
  早期接入时 13 项离线测试通过、1 项 GPU 重置测试跳过。初始三次 primal 配置下约 0.7 秒跌倒撞击后出现软垫翻转；
  减小步长至 0.002 秒未消除。早期改用训练的单次 primal 更新、保留 block 预条件；小测 150 帧 / 3 秒无翻转，
  **机器人仍跌倒，不能宣称已稳定在软垫行走**。轨迹为 `logs/microduck_soft_pad/diagnostics/policy_primal1.json`。
  以上为旧配置历史，当前平地回放修复见首条；HOME 模式仍使用三次 primal。
  `--command VX VY YAW` 设置速度；I/K 前后、J/L 横移、U/O 转向、X 零速、R 重置软垫和机器人及策略历史。
  默认从垫面中心开始，`--spawn ground` 从垫子前方平地开始。跌倒不自动重置，便于观察迁移失败。

- **最新目标是在同一场景学习平地和海绵垫上的运动，当前先调整场景。** 新增 `soft_pad.py` 的连续四面体软垫和
  `soft_pad_scene.py` 的独立可视化入口，复用 MicroDuck 资产与 HOME PD；没有改共享训练环境或启动 PPO。
  用户反馈 4 × 4 m 全软地面太卡，默认已缩小为 **1.2 × 0.8 × 0.06 m**，均匀网格 `(24, 16, 3)`、1700 节点、
  **6912 四面体（比原 115200 个减少 94%）**。水平间距仍为 5 cm、厚度方向仍分 3 层，底面粘附固定。
  周围保留带碰撞的刚性平地，默认 overview 镜头同时显示两种地面；垫面高于平地约 6 cm，尚未验证上下垫子的行走。
  材料仍为可压缩弹性近似而非已标定泡沫；密度 80 kg/m³、Lamé 参数 2000/2000 Pa、阻尼 5 保持原值。
  完整表面刚柔接触已启用，机器人接触凸包在 finalize 前配置 SDF；渲染/状态索引取自 finalize 后的拓扑。
  当前尺寸已通过 **7 项离线测试**（GPU 重置回归本轮跳过），正式 SynReal 入口另运行 **300 帧 / 5 秒**，逐帧检查通过。
  末帧最小体积比 0.8727、顶部最大下陷 5.89 mm、躯干高度 0.17455 m；受力后截图及 JSON 在
  `logs/microduck_soft_pad/local_pad.png` / `local_pad.json`。四面体减少比例不等于已实测的整体加速倍数。

- **此前启动失稳修复及大垫验证（以下为旧尺寸的证据，不代表当前默认尺寸）。**
  **已真实登录 SynReal，复现并修复当前场景的启动失稳。** 旧设置接触后约 0.2～0.35 秒角速度急增、软垫翻转；
  场景设置改为 `joint_constraint_primal_iterations=3`、`affine_body_block_preconditioner=True`，仍使用非线性 15、
  线性 30 次、每帧 8 子步（物理 dt=1/480 秒）。单独提高 primal 次数或开启 block 预条件均通过小垫 2 秒对照；
  排除 affine proxy 接触、将其半径归零、关闭碰撞 Hessian 非对角项均未消除旧配置失稳。
  证据指向该刚柔接触场景的关节/接触求解收敛问题，尚不能精确归因到某个原生实现缺陷；未改求解器核心。
  **4 × 4 m 地面已通过 600 帧 / 10 秒动态复核**：逐帧最小四面体体积比 0.7737、最大节点位移 11.34 mm，
  无翻转；末帧最大位移 5.54 mm、躯干高度 0.17457 m，机器人保持站立。
  完整轨迹在 `logs/microduck_soft_pad/diagnostics/large_ground_600.json`，摘要在 `logs/microduck_soft_pad/validation.json`。
  该次共享 GPU 负载下约耗时 253 秒；这不是实时性能或大批量训练吞吐的证明。
  **8 项软垫测试通过**：7 项离线检查，以及真实 SynReal 小垫 120 帧、双缓冲/关节目标完整重置、再运行 30 帧的回归。
  GPU 测试需显式设置 `MICRODUCK_TEST_SOFT_PAD_GPU=1` 并使用正常登录；默认跳过，未伪造认证。
  正式入口 `--test` 另通过大地面 120 帧，受力后截图为 `logs/microduck_soft_pad/loaded_ground.png`；
  同名 JSON 记录 `preview_only=false`、2 秒物理、顶部最大下陷 6.18 mm。该图不是 10 秒轨迹的末帧。
  `--preview` 仍然只显示初始布局、不运行物理。不传 checkpoint 时保持 HOME 关节目标；已有策略回放不等于软地面训练完成。
  新入口命令见第 5 节。后续需接软接触传感器、两种地面的支撑高度/接触奖励、混合出生采样、批量软体重置及上下垫子课程。

- **2026-10-28 顶层代码归并：29 个 Python 文件整理为 22 个。** 只调整本地入口、模块归属和引用；
  不修改两个官方子目录、奖励/课程/物理参数或生成的官方公式。已保留整理前的未提交改动。
  原 `config.py` → `ppo.py`；`tasks.py` 的场景构建 → `policy_spec.py`、旋转工具 → `adapter.py`；
  原启动随机化模块 → `events.py`；原官方 walk/本地 playback 实现 → `play.py`；
  原 mjlab 转发和 ONNX 回放入口 → `mjlab.py train/play`；原地形导出入口 → `export_recipe.py terrain`。
  合并后的旧文件已删除，不保留空转发模块。共享 `train` / `play` / `validate` 命令不变，见第 3、5、14 节。
  文件内容和指纹清单变化会使配方指纹变化；未放宽 checkpoint 检查，旧 checkpoint 不能直接按旧指纹续训。
  整理前基线为 83 通过、8 跳过；按用户后续要求，整理后仅检查语法、导入和 CLI 帮助，不执行训练或仿真测试。

- **此前问题模型是 17:52 运行的 SynReal `model_19500.pt`，增加回放环境数未解决站立失败。**
  这是原训练任务的历史问题，不是当前软垫回放使用的 PCG `model_11100.pt`。
  用户最新观察：30 和 1024 环境都容易摔倒。既有诊断的 32 环境也全部发生失败重置；
  1024 环境在不同测试条件下仍有大量失败，不能表述为“环境多就能稳定站立”。
  固定动作实验确有环境数敏感现象，但尚未证明它是该策略训练失败的主因；
  也未证明该 19500 轮模型陷入了“只站不走”的局部解。详见第 11.8 节。
- **局部 reset/push 跨环境物理状态同步已修复（2026-10-28）。** 只对被编辑环境执行 FK/affine proxy 同步；
  SynReal 驱动历史也只重置这些环境。command-only 事件不再重算物理状态。4 环境 GPU 回归覆盖两后端
  的 reset、平面推力和坐姿翻倒；本地全套 MicroDuck 测试 83 通过、8 个 GPU 用例跳过。
  环境/事件源码属于配方指纹，当前值已变；旧 checkpoint 不能按旧指纹直接续训，尚未运行修复后的长训。
- **已将低速/启停采样扩展到六个方向，尚未验证长训练效果。** 前进/后退、左右横移、左右转向分别有低中高速采样。
  每次 episode 重置约 20% 环境执行站立 2～3 秒、指定方向移动/转向 3～4 秒、停止、同方向同速再次启动。
  只对 `velocity` 启用，由 MuJoCo/SynReal 共用。六项本地奖励及速度跟踪奖励未改；详见第 8.4 节。
  当前指纹已更新，旧进程不热更新，旧 checkpoint 不能直接 `--resume` 切换该采样。
- **最新 MuJoCo 反馈已转为低速不起步。** 新站立课程运行的 1200 轮模型会在行走后归零时继续踏步，
  2100 轮在本次短测中已能停稳，但低速仍站着不动。单环境逐档 0.1/0.2/0.3 m/s 基本不动，
  到 0.4 才起步；用户观察约 0.5。不能再把“600 轮初始不踏步”当作起停都学好了。
  已查到该旧运行纯前进采样下限 0.3、低速不走仍有高跟踪奖励；详见[第 11.6 节](#velocity-start-stop)。
- **历史反馈（10-27）：用户曾认为“env=1 易倒、env=30 不易倒”问题解决，转而关注不跟指令。**
  当时未独立重做环境数量验收，也未得到具体修复版本；不能当作当前模型已通过验收。
  第 11.5 节保留旧物理诊断，当前状态以第 11.8 节及用户最新反馈为准。
- **历史静止策略诊断针对 13:17 的 `linear120_noise015/model_600.pt`，不是当前 19500 轮模型。**
  后续交叉回放发现：它在 SynReal 和 MuJoCo 都站立；会走的 MuJoCo 策略在 SynReal 能前进/后退。
  指令到达 Actor、执行器有响应；训练学习率长期触底，尚无因果消融。详见[第 11.7 节](#synreal-stall)。
- **16:10 检查已有新的 SynReal 训练进程 PID 109234。** 新运行目录为
  `logs/rsl_rl/microduck_synreal/2026-10-27_16-08-19_468974_velocity_standing_curriculum/`。
  保存元数据确认使用当前六项奖励、六方向采样、5%～25% 课程、初始噪声 1.0；当时仅见 `model_0.pt`。
  本次交接未启动、停止或重启训练；后续先核对该运行，避免重复启动，见第 11.4 节。
- **历史目标为用 SynReal 复现官方 MicroDuck。** 当前目标已转为柔性支撑面运动；MuJoCo 保留为刚性地面基线。
- **此前用户确认内八已改善、零指令踏步仍存在。** 当时明确反馈模型为
  `microduck_mujoco/2026-10-27_12-26-36_958920_velocity_stand_contact/model_1500.pt`。
  髋侧摆约 -4°/+3.75°、站距约 8.7 cm；停止后仍约每脚每秒 2.3 次触地。证据详见第 11.3 节。
- 内八改善出现在加入 `hip_roll_neutral=-1` 和 `action_over_limit=-0.5` 后重新训练的模型中。
  此前提高站立采样比例发生在该反馈之后，**不能把内八改善归功于新站立课程**，也未做完整因果消融。
- 当前为六项本地奖励 + 本地站立课程 + 低速/启停采样。奖励权重没有随采样改动增大，髋侧摆/目标超限约束应保留。
  按用户最新要求，课程由 `20/25/25/30/30/35%` 下调为 `5/10/15/20/25/25%`，详细含义见第 8.3 节。
  官方也是按阶段增加，参数为 `2/5/10/15/20/25%`；两者节点均约 0/500/750/1000/1500/2000 轮。
  当前比官方更早达到 25%，并额外包含 20% 启停组；不能把参数当作全体静止时间占比。
- 用户此前启动运行 `2026-10-27_14-19-58_532936_velocity_standing_curriculum`，该运行的旧配方指纹为
  `06c466c95e0946d2dc349a3d9e5c7f711d621266a3cd323623865def8418011b`。
  现已有 600/1200/2100 轮短测，**没有证明低速跟踪或完整收敛**。
- 此前全方向采样版本完整回归 **73 项通过、2 项 GPU 测试默认跳过**，CPU 两轮 PPO 短测完成。
  站立课程下调后定向回归 **22 项通过、1 项 GPU 测试默认跳过**，`git diff --check` 通过。
  本次仅整理交接，没有重跑物理/PPO 测试；当前采样与课程尚无长期行为验收结论。
- 用户最新反馈 3300 轮的前进、后退、转向均有起步门槛。已核对旧 `velocity_standing_curriculum/model_3300.pt`
  仍是 `06c466...` 配方、没有 `local_commands`，因此它没有使用本次或上一轮的低速采样改进。
- `init_noise_std` 主模板、默认值、旧 MuJoCo 运行实际值均为 **1.0**，与官方 Velocity `init_std=1.0` 一致。
  交接记录中的 **0.15 是此前独立 SynReal 试验覆盖值**，不是当前主模板；本轮核对后保持 1.0。

<a id="purpose"></a>

## 1. 项目目的与当前结论

**当前主要目的是让 MicroDuck 在 SynReal 中同时学习平地和局部海绵垫上的运动。**
先验证可压缩软垫与机器人接触，再开展两种地面的站立、行走、转向和启停训练。此前官方任务适配与 MuJoCo 基线作为已有基础。
海绵垫采用体积软体；薄布面和吊床属于后续不同支撑条件，不能只换名称便当作同一种材料模型。

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

1. 先执行 `git status --short` 并看相关 diff，不要 reset 或覆盖用户工作。
   当前工作区有大量此前归并带来的修改、删除和未跟踪文件，不是本次文档交接新产生的代码改动。
   `soft_pad.py`、`soft_pad_scene.py`、`soft_pad_policy.py` 及两个软垫测试文件尚未跟踪；
   `velocity_commands.py` 等归并后的新文件也未跟踪。后续提交或转交须保留完整依赖，不要只带已跟踪文件。
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
10. 在已授权范围内直接完成检查和修改，不反复请求已有许可；不要默认启动无关的 1024 环境长训练。
    此前已执行 SynReal 从零、1024 环境、每 300 轮评估实验；现有新运行见第 11.4 节，不自动重复启动。
11. 不泄露 `STYLE3D_ACCOUNT` / `STYLE3D_PASSWORD`，不把凭据写入文档、脚本、日志或运行元数据。
12. 进入官方子目录修改前分别阅读其 `AGENTS.md`；本文描述本地适配层，不替代官方目录内部规则。
13. 此前 SynReal 停滞诊断在隔离副本内完成，未改项目训练源码/配置；用户随后已授权低速/启停和站立采样调整。
    本次“做好交接”仅更新本文，不额外合入实验补丁、调整奖励或启动训练。所有策略模型保存到
    `/home/ubuntu/work/style3d-sim/logs/rsl_rl/`，提供模型时明确文件路径，不能只给目录或 `/tmp` 下的别名。

<a id="architecture"></a>

## 3. 代码导航与运行路径

| 文件 / 目录 | 职责 |
| --- | --- |
| [train.py](train.py) | 本地共享 PPO 新训练、保存、同任务续训 |
| [common.py](common.py) | CLI、JSON 默认值、SynReal 登录、checkpoint 契约检查 |
| [ppo.py](ppo.py) | PPO 默认值、任务参数映射、Roulade 镜像损失 |
| [env.py](env.py) | 唯一环境、PhysicsConfig、后端/场景创建、PD、物理步进、重置和训练调度 |
| [policy_spec.py](policy_spec.py) | 机器人资产、14 舵机顺序、HOME、增益、碰撞过滤、球体场景与出生状态 |
| [soft_pad.py](soft_pad.py) / [soft_pad_scene.py](soft_pad_scene.py) | 软垫几何材质 / 共享环境的显示、CLI、预览与回放入口 |
| [mixed_terrain.py](mixed_terrain.py) | PPO 混合出生、批量软垫重置、变形支撑高度、体积检查与分地形指标 |
| [checkpoint.py](checkpoint.py) | 显式迁移初始化、来源权重与归一化校验、可追溯哈希 |
| [soft_pad_policy.py](soft_pad_policy.py) | 冻结 checkpoint 加载、交互指令及 Actor 推理；物理交给 env.py |
| [task_registry.py](task_registry.py) | 33 任务名称及官方任务 ID 映射；保持无仿真依赖 |
| [recipe.py](recipe.py) | 配方加载、指令采样、奖励适配、课程与指纹 |
| [velocity_commands.py](velocity_commands.py) | 本地 Velocity 六方向低速分段采样、episode 内连续启停 |
| [reward_functions.py](reward_functions.py) | 提取的官方任务奖励函数 |
| [velocity_rewards.py](velocity_rewards.py) | 提取的 mjlab 步态奖励函数 |
| [observations.py](observations.py) | Actor/Critic、噪声、偏差、延迟、缩放和缓存 |
| [events.py](events.py) | 启动时脚底摩擦随机化、重置、终止课程和推扰适配 |
| [reset_functions.py](reset_functions.py) | 提取的官方重置/倒地判断公式 |
| [sensors.py](sensors.py) / [adapter.py](adapter.py) | 几何接触计时、角动量、状态约定转换 |
| [terrain.py](terrain.py) | Rough/Slope 几何、出生原点、地形高度查询 |
| [stability.py](stability.py) | 子步物理异常检查与故障快照 |
| [play.py](play.py) | 本地 `.pt` 回放、快速 Actor 路径及原官方 `walk` 演示 |
| [mjlab.py](mjlab.py) | `train` 转发官方 mjlab 训练；`play` 运行官方 ONNX CPU MuJoCo 推理，可选择 BAM |
| [export_recipe.py](export_recipe.py) | `recipe` / `terrain` / `all`：官方配方/函数与固定地形导出 |
| [validate.py](validate.py) / [tests](tests/) | 站立检查与回归测试；不是技能收敛证明 |

不要混淆以下路径：

- `python -m examples.microduck.train --solver mujoco|synreal`：本项目共享 PD 训练。
- `python -m examples.microduck.mjlab train`：官方 mjlab 训练；使用官方任务 ID 和独立 Python。
- `python -m examples.microduck.play --checkpoint ...`：本地 PD checkpoint 回放。
  不传 checkpoint 时默认进入原 `walk` 演示，并不是刚训练的模型。
- `python -m examples.microduck.soft_pad_scene --checkpoint ...`：独立的平地＋局部软垫策略回放；无 PPO 更新。

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

<a id="soft-pad-handoff"></a>

### 3.1 统一环境与软垫查看入口

`env.py::MicroDuckEnv` 统一构建、配置、推进和重置环境；`SoftPadScene` 只持有它并负责显示与交互。
训练与查看共享同一个机器人＋软垫模板和物理循环，episode 规则仍按用途区分。

```text
train.py ──────────────┐
play.py ───────────────┼─→ env.MicroDuckEnv / PhysicsConfig
soft_pad_scene.py ─────┘      ├─ policy_spec：机器人
  显示 / CLI / 键盘          ├─ soft_pad / mixed_terrain：软垫、出生与状态检查
                             ├─ apply_policy_targets：统一饱和 PD
soft_pad_policy：冻结推理 ───→└─ advance_physics：唯一物理子步循环
```

| 参数 | 新训练 / 迁移训练 / 未记录该字段的旧策略 | HOME 保持 |
| --- | --- | --- |
| 控制帧周期 | 0.02 秒（50 Hz） | 1/60 秒 |
| 默认子步数 / 物理 dt | 4 / 0.005 秒 | 8 / 1/480 秒 |
| 非线性 / 线性迭代 | 5 / 15 | 5 / 30 |
| `joint_constraint_primal_iterations` | 1 | 3 |
| `affine_body_block_preconditioner` | False | True |
| `air_density` | 1.225 | 0.0 |

修改运行参数只维护 `env.py`：PhysicsConfig（求解/步长）、PAD_CONFIG（垫材质）、EnvironmentConfig（数量/出生）。
普通训练与回放使用当前 profile；保存为 5/50 的旧模型也不会默认覆盖当前 5/15。
显式 `--use-checkpoint-environment` 可恢复保存参数；缺失字段仍回退当前默认值，不能声称恢复了未保存的参数。
严格 `--resume` 要求实际参数与保存记录一致；改变环境应使用 `--init-from`。
`ENVIRONMENT_CONFIGS` 的 official/mixed_soft_pad 分别服务原任务与混合任务，命令行可临时覆盖数量/出生概率等设置。
`--substeps` 仍是查看入口的显式步长覆盖，会输出动力学差异提示。查看时物理运行仍需要正常 SynReal 登录。

硬接触保留模型原材质：脚/地面的 `ke=5e6`、`kd=10000`，默认 `mu=1`；脚底摩擦可用共享事件采样。
软接触单独使用 `soft_contact_ke=5e4`、`kd=100`、`mu=0.8`。
`soft_pad_scene.metrics()` 会输出实际材质、步长、求解配置、checkpoint 路径、指纹、躯干状态及最小四面体体积比。
`--test` 检查有限状态、固定底面和正体积；它不判断步态是否稳定、跟踪是否达标。

默认 `--spawn pad` 在垫面中心，`--spawn ground` 在 x=-0.9 m 的垫前平地，后者距垫边约 0.3 m。
12 秒平地回归曾在诊断脚本中把出生 x 移到 -2 m，以便持续前进而不碰垫边；默认 CLI 出生点不是那次长测位置。
当前单场景 `R` 重置整个机器人和垫子、双缓冲、关节目标及策略历史；跌倒不自动重置。
`--preview` 无动力学、无登录；完整机器人 SDF 构建仍依赖 CUDA，不能根据旧 CPU 提示宣称 CPU 完整预览已通过。

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
| `ppo.py` 默认值 → 任务覆盖 → 模板 `ppo` | 新训练 PPO 配置的覆盖顺序 |
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

`ppo.py` 的保存间隔 500 是默认值，会被任务和模板覆盖。
模板也会覆盖某些任务的官方参数，例如 entropy 0.01 覆盖 Rollers 的官方 0.03。
`config.json` 不等于所有执行源码的完整存档；必须同时保留对应代码版本。

动作噪声核对（2026-10-27）：本地 `init_noise_std=1.0` 对应官方 Actor
`distribution_cfg.init_std=1.0`，两者均使用 scalar 参数化；当前所有任务的官方快照初始值也是 1.0。
`Policy/mean_noise_std` 是训练中学出来的标准差，Velocity 2100 轮约 0.151 不代表初始化参数被改为 0.15。
推理取策略均值；续训加载 checkpoint 中已学习的噪声参数，不会因修改模板而自动重新初始化。
此前独立 SynReal 排查有显式使用 0.15 的实验，见第 11.4/11.5 节，不将其混同于主模板。

<a id="commands"></a>

## 5. 日常命令与依赖

<a id="workbench"></a>

### 5.0 中文训练与评估工作台

**2026-10-29 多实验更新：** 工作台支持同时启动多个训练。每次启动生成独立 ID、进程和日志目录，
即使实验名称相同也不会覆盖。上方“训练实验”列表显示状态、任务、设备、环境数、轮数和 PID；点击选择后，
指标、“停止所选训练”和预览对应这个实验。下方“启动新实验”表单始终用于创建新的训练，修改它不会改变运行中的实验。
每个训练拥有独立预览窗口；切换选择不会关闭已有窗口。停止训练、关闭预览只作用于指定进程，不影响其他实验。
日志下拉列表可单独选择任意训练或预览；轮询只传当前选择的日志，页面刷新后重新加载本次控制台的实验列表。
同一 GPU 的多个训练会共享显存和算力，环境数按每个实验分别计算；可在每次启动时指定不同 CUDA 设备。
不提供自动显存分配、GPU 排队或多个策略自动融合；TensorBoard 和独立模型评估仍各保留一个进程。

初始化方式旁已加入常显说明：

| 方式 | 读取旧权重 | 优化器和轮数 | 典型用途 |
| --- | --- | --- | --- |
| 从零开始 | 否，策略和价值网络随机初始化 | 全新 | 新任务、从零基线 |
| 严格续训 | 策略、价值网络、归一化及优化器等 | 恢复，追加指定轮数 | 同一实验中断后继续，保留配方/环境校验 |
| 迁移初始化 | 策略、价值网络、动作标准差及归一化 | 重置，可设新学习率和环境 | 平地策略学习软垫，当前仅 SynReal Velocity |

严格续训不恢复完整仿真状态与随机数状态；迁移不保证旧能力不退化，也不支持任意跨任务迁移。
本次多实验验证：23 项 UI 定向测试通过；Firefox 中两个 CPU 训练并行运行、独立日志和保存目录、
分别打开预览、停止 A 后 B 继续迭代、关闭 A 预览后 B 预览保留、刷新页面和窄屏布局均已验证。
证据在 `.validation/microduck_ui_multi/`。未验证 SynReal 多训练登录或 GPU 吞吐/显存上限。

旧版控制台进程不会热更新 Python 后端。待现有训练妥善停止后，重启控制台并刷新网页才能使用新版；
新版页面检测到旧后端会给出重启提示，避免错误地调用单实验接口。此次开发保留用户已有控制台进程。

在仓库根目录、现有 `.venv` 中启动；Web 服务使用 Python 标准库，不需要 Node、前端构建或额外 UI 依赖：

```bash
# MuJoCo，或终端已经设置了 MicroDuck 登录环境变量
.venv/bin/python -m examples.microduck.ui

# SynReal：在终端隐藏输入账号、密码，再打开控制台
.venv/bin/python -m examples.microduck.ui --login

# 不自动打开浏览器 / 自定义端口
.venv/bin/python -m examples.microduck.ui --no-browser --port 8765
```

浏览器访问 `http://127.0.0.1:8765`。3D 窗口在运行 Python 的机器桌面显示，浏览器内不传输仿真画面。
服务器仅监听本机；远程使用需要用户自己的桌面或端口转发。`--log-root` 可切换模型扫描和保存根目录。

- **训练工作台**：选择 33 个注册任务、MuJoCo/SynReal、任务默认地形或平地＋软垫、计算设备、环境数和轮数。
  高级设置支持学习率、熵系数、采样步数、保存间隔、epochs、mini-batches、学习率调度、种子及摩擦选项。
  空白参数不覆盖任务或 `env.py` 默认；UI 提交的临时 PPO 覆盖记录在 `logs/rsl_rl/.workbench/`。
  它不是新的环境默认配置。默认从零训练 1000 轮，仅点击开始才运行。
- **新训练 / 严格续训 / 迁移初始化**：分别调用原 `train`、`--resume` 和 `--init-from`。
  续训忽略表单 PPO 覆盖，保留原配方与环境校验；轮数表示本次追加量。迁移仅支持现有 SynReal Velocity 契约。
  平地＋软垫同样仅支持 SynReal Velocity。当前 MuJoCo CPU 路径只支持 1 个环境，多环境使用 CUDA。
- **查看训练效果**：无渲染训练持续运行，另开 `play` 子进程加载当前实验最新完整保存的 `model_N.pt`。
  检查 ZIP 尾部，跳过正在写入的权重；没有权重时显示等待提示，不会误播官方 walk 策略。
  预览使用该训练保存的环境和 CLI 覆盖，默认 1 个环境、零速度；显示实际 checkpoint 路径。
  这是保存时刻的策略快照，不是 rollout 的实时镜像，也不自动热更新。关闭再打开可加载最新保存权重。
  窗口内 I/K 前后、J/L 侧移、U/O 转向。关闭预览不停止训练，释放该回放的 GPU/渲染开销；提速幅度依负载而定。
- **模型与评估**：扫描现有训练目录，可按实验名、任务、后端搜索，选择具体轮次或手动输入 `.pt` 路径。
  自动使用权重任务和后端；可配置评估环境数、设备、初始指令、历史环境恢复及诊断选项。
  默认不绕过指纹；旧权重诊断需用户勾选允许配方差异。权重旁必须保留 `config.json`。
  “填入训练来源”只填写表单，不自动启动。可视化评估是交互回放，不计算成功率，也不证明策略收敛。
- **监控和日志**：显示已完成迭代数（原日志编号 + 1）、采样速度、episode 平均奖励（首次 episode 结束后才出现）。
  页面保留最近 1200 行，可下载当前尾部日志；完整输出写到界面显示的 `output.log`。
  TensorBoard 使用本机 6006 端口，启动后点击打开；端口占用或进程失败会显示日志。

关闭网页不会结束子进程，重新打开可继续管理。停止训练会先发中断信号，超时再终止；只保证保留已保存权重，
不额外承诺保存当前迭代。终端 Ctrl+C 会停止本工作台拥有的子进程；不接管或停止外部终端启动的训练。
工作台重启后可扫描历史模型，但不恢复上一次进程状态。日志、模型与 `.workbench/` 元数据保存在忽略的 logs 目录。

SynReal 凭据只在进程内存和子进程环境中使用，不经过网页、不写配置或日志。
沿用 MicroDuck 的 `STYLE3D_ACCOUNT` / `STYLE3D_PASSWORD`；`--login` 负责隐藏输入，实际登录在子进程初始化时完成。
UI 对缺少凭据直接报错，避免后台进程等待不可见的登录输入。

源码：`ui.py`（本机 HTTP 服务）、`ui_backend.py`（参数和进程管理）、`ui_assets/`（中文界面）。
回归：`tests/test_ui.py` 覆盖默认参数、权重选择、未完成保存过滤、进程隔离和接口访问限制。
本轮完整回归 **162 passed / 9 skipped**；真实 MuJoCo CPU 单环境 2 轮训练和 GL 20 帧预览已通过。
Firefox 浏览器交互检查通过：33 任务加载、表单启动 CPU 训练、预览关闭并重开、选择权重评估、填入续训来源、
窄屏布局、完成轮数显示和 TensorBoard 启停。短测进程与测试浏览器均已关闭。
该短测仅验证 UI 调用链，不证明步态质量；未执行需要登录的 SynReal 动力学或长训练。

短测权重：`logs/rsl_rl/microduck_mujoco/2026-10-29_11-09-37_784902_ui_cpu_smoke/model_1.pt`。
检查与页面截图保存到 `.validation/microduck_ui/`，不属于交付依赖。

平地/海绵垫混合迁移训练（先完成真实 SynReal 小批量验证，正常隐藏输入登录）：

```bash
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"
.venv/bin/python -u -m examples.microduck.train \
  --training-config examples/microduck/configs/rtx4090_mixed_soft_pad.json \
  --init-from logs/rsl_rl/microduck_synreal/2026-10-28_10-04-52_798619_velocity_PCG/model_13500.pt \
  --num-envs 1024
```

`--init-from` 支持目录并在启动时选择最高编号 checkpoint，正式实验建议固定文件路径。
同一混合运行中断后用 `--resume <新运行/model_N.pt>`；当前环境和配方指纹必须匹配。
需要恢复历史环境参数时显式加 `--use-checkpoint-environment`，不会绕过配方检查。
评估可用 `play --checkpoint <新运行/model_N.pt> --solver synreal --pad-probability 0|1` 分别检查平地/垫面；
出生概率覆盖只允许回放，严格续训须保持原分布。初始软垫课程不包含上下垫技能。

海绵垫场景使用根目录环境；以下命令不启动训练：

```bash
# 平地 + 1.2 × 0.8 m 局部海绵垫，SynReal 固定 HOME 姿态承重；终端隐藏输入登录。
.venv/bin/python -m examples.microduck.soft_pad_scene

# 近距离观察机器人；窗口也可用 Terrain view / Robot view 按钮切换。
.venv/bin/python -m examples.microduck.soft_pad_scene --camera close

# 运行 5 秒物理，逐帧检查状态、固定底面和四面体体积，并保存受力后的截图及同名 JSON。
.venv/bin/python -m examples.microduck.soft_pad_scene --headless --num-frames 300 \
  --test --screenshot logs/microduck_soft_pad/local_pad.png

# 仅初始布局预览，不需要 SynReal 登录；物理帧数为 0，不是动态验证。
.venv/bin/python -m examples.microduck.soft_pad_scene --preview
```

在该场景回放用户指定的 PCG 运行（固定文件可保证复现；传运行目录则选择当时最高保存轮次）：

```bash
.venv/bin/python -u -m examples.microduck.soft_pad_scene \
  --checkpoint logs/rsl_rl/microduck_synreal/2026-10-28_10-04-52_798619_velocity_PCG/model_11100.pt \
  --allow-recipe-mismatch --randomize-foot-friction --spawn ground --command 0 0 0 \
  --viewer gl --num-frames 30000
```

用户确认能走的原平地回放入口，保留作对照：

```bash
.venv/bin/python -u -m examples.microduck.play --solver synreal \
  --checkpoint logs/rsl_rl/microduck_synreal/2026-10-28_10-04-52_798619_velocity_PCG/model_11100.pt \
  --num-envs 1 --randomize-foot-friction --viewer gl --num-frames 30000 \
  --diagnostics --allow-recipe-mismatch
```

该运行的保存指纹为 `1ae1713d...`，与当前代码不同，所以上例显式开启诊断覆盖；不是恢复原配方的等价回放。
`--spawn ground` 从平地开始，上例零速等待键盘指令。I/K 前后、J/L 横移、U/O 转向、X 零速；垫子边缘约有 6 cm 高差。
策略回放默认使用 env.py 的步长；`--substeps 8` 会改变动力学，只有明确做对照时才覆盖。

R 重置机器人与软垫，空格暂停，鼠标/WASD 调整视角。默认 overview 显示软垫及周围平地，close 用于观察机器人。
可用 `--size 长 宽 厚`、`--cells NX NY NZ`、`--mu`、`--lambda`、`--density`、`--damping` 调整软垫。
尺寸单位为米，密度 kg/m³，`mu` 与 `lambda` 为 Lamé 参数（Pa）；默认 2000/2000，尚未进行真实泡沫材料标定。

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

`--viewer null --benchmark --num-frames 300` 可做无窗口测速；步进路径由任务和地形自动选择。
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
官方独立环境缺失时，`mjlab.py train` 提示使用 `uv sync --project examples/microduck/microduck_rl-develop --locked`。

以下是官方参考路径，不是共享 SynReal 训练：

```bash
python -m examples.microduck.mjlab train --list-tasks
python -m examples.microduck.mjlab train --num-envs 64 --iterations 5
python -m examples.microduck.mjlab play --solver mujoco --policy velstand --keyboard
python -m examples.microduck.play --solver synreal --policy walk --num-frames 3000
```

`mjlab.py play` 当前只支持 MuJoCo，默认 `--actuator metadata`；显式 `--actuator bam` 才选 BAM 分支。
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
- Torch/Warp 交界显式同步；reset 或外力事件只对被改动 articulation/body 执行 FK 与 affine proxy 同步，
  并且只重置这些 SynReal 驱动的角度历史。

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
官方/旧本地纯前进采样参数 20%；原地转向覆盖概率 Velocity 15%、VelStand 20%，会清除 standing 标记。
当前精确 `velocity` 已改为第 8.4 节的有符号平移/转向采样，其他任务仍沿用原分支。

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
通过共享 `recipe.py` 计算；髋侧摆和目标超限两项复用 `reward_functions.py` 中已有官方函数。
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
实测静止奖励已触发；目前没有证据表明继续增加惩罚就能解决。此前将站立课程提高到 `20/25/25/30/30/35%`。
用户随后要求减少站立采样，当前改为 `5/10/15/20/25/25%`，沿用原控制步阈值，六项奖励权重不变。
只覆盖精确 `velocity`，配置位于 `tasks.velocity.curricula.standing_envs`，标记 `local: true`。

| 控制步阈值 | 约 PPO 轮数 | 官方原 standing 参数 | 当前 standing 参数 | 普通组零 twist 概率约 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0 | 2% | 5% | 4.25% |
| 12000 | 500 | 5% | 10% | 8.5% |
| 18000 | 750 | 10% | 15% | 12.75% |
| 24000 | 1000 | 15% | 20% | 17% |
| 36000 | 1500 | 20% | 25% | 21.25% |
| 48000 | 2000 | 25% | 25% | 21.25% |

官方配置见 `microduck_rl-develop/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py` 的 `standing_envs`。
源码注释说明先建立行走能力，再增加站立比例；`mdp.py → standing_envs_curriculum` 实际只按累计步数切换，
不会检测机器人是否已学会走路。表中“官方”指仓库内的官方副本，本次没有联网更新上游。
此前本地 20%～35% 课程实际只在约 500/1000/2000 轮上涨；当前在约 500/750/1000/1500 轮各增 5 个百分点。

`step` 是配方累计控制步计数，不是物理子步数或全部并行环境的样本数；每 PPO 轮累加 24，不能再乘 1024。
`rel_standing_envs` 是每次命令重采样时选中站立的概率，不是奖励，也不是固定留出某几个机器人始终站立。
`commands.twist.rel_standing_envs=0.02` 保留上游初值；本地 `update_commands()` 使用课程阶段值覆盖它。
选中后 twist 为 `(0, 0, 0)`，不冻结关节动作；通常 3～8 秒后重新采样。相邻阶段比例相同就是继续保持。
阶段按 `>` 选取；随机采样分支的原地转向仍优先覆盖约 15% 样本。
因此普通组零 twist 概率约为参数 ×0.85；官方原课程初期/最终阶段约 1.7%/21.25%，当前约 4.25%/21.25%。
此概率指一次命令重采样的期望结果，不是按时间加权的精确训练数据占比，也不是专门保证停走转换次数的课程。
此次下调旨在增加普通组的移动指令采样机会；尚未训练验证，不能据此认定旧站立比例就是不起步的根因。
`official_alignment.local_curricula` 记录实际本地课程，导出器与本地奖励一样保留其 `local: true` 配置。
运行中的旧进程不会热更新；旧 checkpoint 的课程仍以对应运行的代码版本为准。

第 8.4 节加入启停分组后，本节表格的零 twist 概率只适用于普通随机指令分组；
全体环境还包含有固定阶段时长的启停组，不可继续用 standing 参数 ×0.85 代表全体占比。

本次下调验证：`test_gait_rewards.py` 与 `test_velocity_commands.py` 共 22 通过、1 项 GPU 测试默认跳过。
覆盖六阶段实际采样、六方向低速/启停分组、课程导出保留及 MuJoCo CPU 奖励/重置检查；未启动长训练。

### 8.4 当前全方向低速与连续启停采样

用户授权优化低速不起步、行走后停步问题；本次先调整已确认覆盖不足的指令分布，
不同时改变速度跟踪奖励、Z/静止/内八约束、动作平滑课程或初始探索噪声。
配置在 `tasks.velocity.commands.twist.local_sampling`，整个 twist 配置标记 `local: true`：

| 参数 | 值 | 意义 |
| --- | --- | --- |
| `axis_velocity_bins.lin_vel_x` | `[0.05,0.15]`、`[0.15,0.25]`、`[0.25,0.4]` m/s | 前进/后退，正负号各半 |
| `axis_velocity_bins.lin_vel_y` | `[0.03,0.1]`、`[0.1,0.2]`、`[0.2,0.3]` m/s | 左右横移，正负号各半 |
| `axis_velocity_bins.ang_vel_z` | `[0.05,0.2]`、`[0.2,0.5]`、`[0.5,1.0]` rad/s | 左右转向，正负号各半 |
| `pure_translation_fraction` | 0.5 | 普通组约 50% 候选指令为纯 x/y 平移，两个轴各半，随后仍可能被站立/转向覆盖 |
| `start_stop_fraction` | 0.2 | 每次 reset 重新抽选约 20% 环境，episode 内使用启停序列 |
| `standing_time_range_s` | `[2,3]` | 每段站立/停止保持时间 |
| `moving_time_range_s` | `[3,4]` | 每段移动/转向保持时间 |

每个轴的三个速度段等概率、段内均匀；用幅值采样后随机取正负号，所有指令保持在任务原范围内。
启停组在 x/y/yaw 三轴中等概率选择一个，再独立选正负号和速度段，六方向期望等频。
从零指令开始，在同一物理 episode 内交替站立和运动，停止后用同一方向、同一速度再次启动；
不重置身体、不清空策略动作、不强迫脚落地。各段时长独立采样，运动指令每个 episode 抽取一次。
局部 reset 只重选对应环境的组别/速度/阶段；下一 episode 重新分组，避免固定摩擦或编码器偏差绑定某一指令类型。

其余约 80% 环境先采一般随机速度，再将约 50% 候选改为纯 x/y 有符号平移；
随后约 15% 原地转向优先覆盖，并清除对应 standing 标记，余下仍遵循 standing 课程。
纯平移和纯转向都使用上述速度分段，混合多轴指令仍保留原范围均匀采样。
普通组沿用 3～8 秒重采样；站立概率仍约为 standing 参数 ×0.85，头部课程不变。
这些是有覆盖关系的条件概率，不能直接相加；中期 standing 参数 0.2 时，普通组约 34% 为混合指令。
旧字段 `rel_forward_envs=0.2`、`turn_in_place_min_frac=0.4` 为上游参考，启用本地采样时不再决定前进或转向幅度；
`rel_turn_in_place_envs=0.15` 仍控制普通组的纯转向覆盖概率。
启停组站立时间占比理想情况下约 2.5/(2.5+3.5)=41.7%；全体占比还受 reset、阶段起点和 episode 长度影响。

只有精确 `velocity` 启用；VelStand、Rough、Backlash 等保持原采样。
两后端经同一 `OfficialRecipe` 使用该逻辑，无引擎改动；`config.json → official_alignment.local_commands`
记录实际配置。导出器保留 `local: true` 的 commands，避免重新导出时丢失本地调整。

此前验证（站立课程下调前）：全方向版本完整测试 73 通过、2 默认跳过。CPU 两轮 PPO 已完成，运行目录：
`logs/rsl_rl/microduck_mujoco/2026-10-27_15-53-49_229326_velocity_all_directions_smoke/`。
测试覆盖六方向各速度段、混合指令/站立/转向比例保留、同方向同速重启、计时保持、局部 reset、
空选择与单环境到期隔离、幅值不能超过任一正负方向的任务限值、导出保留及任务作用域。
本次 GPU 不可访问，未新增 GPU/SynReal 物理验证；采样实现为两后端共享 Torch 代码。
这是功能短测，不能作为可展示的技能模型；尚未用新采样训练到行为收敛。

前一轮只覆盖纯前进的版本曾通过 69 项测试、MuJoCo CPU/GPU 接入检查及 CPU 两轮 PPO，
对应 `2026-10-27_15-42-14_018673_velocity_low_speed_smoke/`，不是本次六方向版本的 GPU 验证。
该前进专用版没有正式长训；用户反馈的旧 `velocity_standing_curriculum/model_3300.pt` 仍没有使用它。

新训练命令示例（先检查第 11.4 节已有运行，勿重复启动）：

```bash
source .venv/bin/activate
python -u -m examples.microduck.train \
  --training-config examples/microduck/configs/rtx4090_synreal_official_adapted.json \
  --solver mujoco --task velocity --run-name velocity_all_directions_start_stop
```

切换 `--solver synreal` 使用同一采样；16:08 的新 SynReal 运行已保存当前配方，见第 11.4 节。
用户反馈环境数量问题已解决；若再次出现物理异常，再使用第 11.5 节历史复现包核对版本。
新配方需要新训练；不要修改旧运行指纹或放宽 `--resume` 检查来伪装同配方续训。
验收应分开测六个方向，包含各轴低中高速的从静止启动、持续跟踪、停止及再次启动，并复核站距/脚尖方向。
推荐 x 轴 ±0.05/0.1/0.2/0.3/0.4 m/s、y 轴 ±0.03/0.1/0.2/0.3 m/s、yaw ±0.05/0.1/0.2/0.5/1.0 rad/s。
若低速仍站着不动，再单独对跟踪误差尺度做对照；不把本次采样覆盖通过称为问题已解决。

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

本地 `ppo.py` 目前保持普通 PPO，未加载 BC 或教师。
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
- 当前六项本地奖励 + 站立课程 + 低速/启停采样 + 局部物理同步修复指纹为
  `1ae1713dd93a3d67770d167f98f3463dee4c99c871c627758b6264f57e0192bb`（六方向、站立课程 5%～25%）。
  同步修复前该组合指纹为 `f69e030e1c68de0e5a527fa8b52952df39514de0ab7ecb0903f62f5fe0e7a475`。
  此前六方向、站立课程 20%～35% 版为 `2e155eeb8f81d0fd16501d6109462a63f86f14ac871dcdae7eb455488f420995`。
  前进专用采样版为 `3c654528b7e338d55719b58ef43c6987a533a05e51889b5fae1683cccb6e8379`。
  此前六项奖励 + 站立课程版为 `06c466c95e0946d2dc349a3d9e5c7f711d621266a3cd323623865def8418011b`。
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
  混合软垫自动走完整环境步，关闭训练 interval events，仍受回放噪声/头部指令设置影响。
  快速专用物理路径主要覆盖 velocity/velstand/groundpick/ballkick，其他任务回退完整 `env.step()`。
- 当前共享 `.pt` 配方无 CUDA graph；`--disable-cuda-graph` 主要影响原 walk 演示。
- CUDA 保存的 checkpoint 做 CPU 诊断时，当前 RSL-RL `load()` 未显式设置 `map_location`；历史诊断曾临时适配。
  不要把 CPU 环境可步进等同于 CLI 已支持任意 CUDA checkpoint 的 CPU 加载。
- 同一 Viewer 不说明同一引擎；相同代码、种子也不保证独立接触轨迹逐位一致。

<a id="next"></a>

## 11. 近期实验与下一步工作

<a id="soft-pad-next"></a>

### 11.0 当前优先：定位混合训练后的站立/行走退化

软垫批量环境、接触传感器、变形支撑高度、混合出生、局部故障隔离和 PPO 接口均已实现。
此前已启动过真实 1024 训练，之后用户运行 256 环境到至少保存 model_2900；不要重复按“尚未接入训练”实施。
当前首要待办是顶部“本次接手必读”的退化问题，其次是交互性能；跨垫边、长期稳定性尚未验收。

1. 固定来源，分别保留平地 model_11100（用户基线）、model_13500（实际迁移来源）、混合 model_2000/2900。
   不用“目录最高权重”代替可复现的文件路径；核对磁盘元数据的 50→15 改写，保留审计原件。
2. 在同一当前环境中比较零速站立及 vx=0.1 m/s 行走，分别从平地、垫面出生；固定种子、摩擦、
   出生点/姿态、材质、dt、4 子步、求解器、外力及重置规则。先固定 5/15，再单因素比较 5/50。
   使用同一命令条件比较所有权重，记录实际命令、倾角、躯干高度、速度误差、首次倒地时间、重置和软垫故障。
3. 对照交互固定出生与训练随机出生、确定性均值动作与训练采样动作、1 世界与批量世界。
   `soft_pad_scene` 不自动重置，`play` 混合环境会按训练规则故障隔离/重置，两者不是相同评估协议。
   小样本先排除回放/物理条件差异，再判断策略是否遗忘；不要先改奖励、学习率或线性迭代并宣称修复。
4. 如果证实策略本身退化，再评估较小更新、旧平地策略行为约束、固定平地/软垫评估集及最佳模型选择。
   只有在来源基线和短训回归通过后才重启长期混合训练。用户要求目标 1024；历史 256 是实际运行覆盖值。
5. 性能对照应保持相同 GPU 负载、权重、迭代和渲染条件；确认瓶颈后再减少同步，保留接触溢出与数值检查。
   当前没有运行的训练需要暂停；未来若出现新进程，应核对用户授权而不是套用旧 PID 的暂停许可。

真实 SynReal 测试需终端登录；通过 CPU 测试或 CUDA 几何构建不代表完成动力学验证。
当前仅参数集中和保存完整性已验收；训练退化、关节拖拽真实性、重构后的动力学一致性仍待验证。

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

### 11.2 已确认的运行文件与配方关系

以下为 2026-10-27 14:29 文件检查结果；checkpoint 编号不等于当前迭代数，不表示行为已评估。
进程状态另见第 11.4 节；SynReal 行已按 14:39 核查更新。

| 日志根下的运行目录 | 当次看到的最新 checkpoint | 配方关系 |
| --- | --- | --- |
| `microduck_mujoco/2026-10-27_10-26-04_633842_velocity_z` | model_2100.pt | 仅 Z 惩罚，与当前指纹不同 |
| `microduck_mujoco/2026-10-27_11-09-59_978957_velocity_still_heading` | model_3000.pt | 三项本地奖励，指纹 `d1a354...` |
| `microduck_mujoco/2026-10-27_12-26-36_958920_velocity_stand_contact` | model_1500.pt | 六项奖励、旧站立课程，指纹 `86d328...`；用户已确认内八改善 |
| `microduck_mujoco/2026-10-27_14-19-58_532936_velocity_standing_curriculum` | model_0.pt | 六项奖励、新站立课程，指纹 `06c466...`，与当前源码一致 |
| SynReal 实验 D，完整目录见第 11.5 节 | model_600.pt | 批量跟踪未通过，单环境评估中 |

根目录是 `logs/rsl_rl/`；下一位 AI 应重新检查 checkpoint 和进程，不假定上表仍是最新。
更早 SynReal 运行的文件状态本次未重新核查，不沿用早期“最新 model_1200.pt”的说法。

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

**以下是当时短暂微调的结论，不覆盖后续从零训练的改善：该阶段仍未解决内八/踏步。**

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

当时保留 `hip_roll_neutral=-1`、`action_over_limit=-0.5`，理由是约束已测到的限位利用行为；
该批短微调没有证明其改善从零训练。随后 `velocity_stand_contact/model_1500.pt` 已显示内八明显改善，见下文。
因此不要继续用该批旧策略短微调的失败来声称当前从零训练也没有改善；仍需检查停稳和指令跟踪。
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
  当时假设是站立及停走经验不足，故提高站立课程；用户随后要求下调，当前值见第 8.3 节，尚无因果验证。

本轮 **35 项测试通过、1 项 GPU 测试跳过**，包括 4 个阶段各 20,000 次真实 CPU 指令采样，
核对零命令概率、原地转向概率、任务范围和导出保留；CPU 单环境 2 PPO 轮/48 样本训练及保存通过。
没有追加 GPU 训练，没有修改用户运行中的训练或另一个 SynReal 进程。新课程的训练效果尚未验证。
证据：`.validation/microduck-gait/stand-contact-1500/`，含完整诊断、头部比较、修改前源码/快照和测试/CPU 短测日志。
旧实验脚本断言旧指纹，需使用对应源码复现。

当时建议比较旧运行的 2000/3000 及之后 checkpoint；本次交接核查旧进程已不在、文件只到 model_1500.pt，
不能继续假定该目录会产生新模型。当前优先评估新站立课程运行，保存间隔为 300 轮，应使用真实存在的文件。
1500 轮附近有动作平滑、头部和站立课程变化，但这些轮数不是保证停稳的收敛阈值。
新站立课程只对新启动训练生效，普通 `--resume` 仍禁止跨指纹。

1. 验证新约束从零训练的结果：零指令、行走后停止、重新起步、转向、侧移。
   记录真实 Actor 指令、脚部速度/接触、机身 vz、足部 yaw、速度误差及重置原因。
2. 检查新约束是否导致僵硬、单脚悬停或速度跟踪进一步变差；权重 -1.5 不是已验证最优值。
   当前静止接触项已实现，但仍需新训练对照；特别检查是否出现双脚滑动、单脚长期悬停或失去起步能力。
3. 在相同任务/配置/采样量/物理开关下比较 MuJoCo 与 SynReal；当前两条长期实验指纹不同，不能直接归因引擎。
4. 对 StandUp、BallKick、VelStand 分开评估任务成功。高奖励或能站着不证明会起身/踢球。
5. 用户决定采用官方教师路线时，再做显式 warm-start/双教师接口；或者独立设计无教师课程。
6. BAM、物理随机化、接触力及 Rough/Backlash 全面动力学验证属于进一步官方对齐工作，尚未完成。

<a id="active-runs"></a>

### 11.4 历史运行任务、GPU 争用与终端命令

本节 PID 和运行状态为历史记录；最新状态以顶部带时间的接手核查为准。

**最新只读核对：2026-10-27 16:10（Asia/Shanghai）。** `/proc` 中发现 SynReal 训练 PID `109234`，
入口 `examples.microduck.train`，工作目录为仓库根，task 为 `velocity`，run-name 为 `velocity_standing_curriculum`。
同名标签不能区分历史 MuJoCo 与当前 SynReal 运行；应按完整目录和保存元数据识别。

新运行位于 `/home/ubuntu/work/style3d-sim/logs/rsl_rl/microduck_synreal/` 下：
`2026-10-27_16-08-19_468974_velocity_standing_curriculum/`，核对时已保存 `model_0.pt`，尚无行为评估结果。
`config.json` 确认 solver=SynReal、1024 环境、seed=42、30000 轮预算、从零开始、脚底摩擦随机化开启，
每轮 24 步、每 300 轮保存、初始动作 std=1.0，指纹与第 10 节当前 `f69e030...` 一致。
`local_curricula` 保存 `5/10/15/20/25/25%`；`local_commands` 保存六方向速度段及 20% 启停组。
此目录当前没有 `source_snapshot/`，接手需保留本轮未提交代码；不要用旧 D 的四项奖励/120 线性迭代解释它。
本地 `env.py` 当前默认 5/30 次非线性/线性迭代；新元数据没有单独的 `synreal_solver` 记录。
标准 `train.py` 每 300 轮保存 checkpoint，**不自动执行旧实验脚本的七指令/单环境评估**。

```bash
tensorboard --logdir logs/rsl_rl/microduck_synreal/2026-10-27_16-08-19_468974_velocity_standing_curriculum
ps -p 109234 -o pid,etime,%cpu,comm
```

以下为 **14:29～14:31 的历史进程快照**，不是最新运行状态；本次交接未终止任何进程。
PID 会退出或被复用，之后采取操作前必须重新核对命令行和工作目录。

| PID | 运行内容 | 输出位置 | 本次确认 |
| --- | --- | --- | --- |
| `50259` | `examples.microduck.train` | 用户终端 `/dev/pts/39` | 当时 MuJoCo、1024 环境、六项奖励、20%～35% 课程 |
| `21376` | 根 `.venv/bin/python -u experiment.py` | 下方 `/tmp/.../train.log` | 当时独立 SynReal 实验、1024 环境、601 轮预算 |

两个历史 checkpoint 目录见第 11.2 节。当时 MuJoCo 运行的命令是：

```bash
cd /home/ubuntu/work/style3d-sim
source .venv/bin/activate

# 历史命令，仅供核对；不要据此重复启动训练。
python -u -m examples.microduck.train \
  --training-config examples/microduck/configs/rtx4090_synreal_official_adapted.json \
  --solver mujoco --task velocity --run-name velocity_standing_curriculum
```

SynReal 实验工作目录为 `/tmp/microduck-synreal-velocity-20261027-linear120-noise015`。
其保存元数据确认：线性迭代 **120**（本地默认 30）、非线性迭代 5、block preconditioner 关闭；
初始动作噪声 **0.15**（当前模板 1.0）。本地奖励元数据仅含 Z、脚速、髋侧摆、脚尖四项，
**不含**当前接触项/目标超限项，也未记录新本地站立课程；指纹为
`41052505f33495d9326ce65c31f5be2a8b03a40fbd2f54df3ccccad997949c3d`。
该隔离实验不能与当前 MuJoCo 结果直接做“其余条件完全相同、仅引擎不同”的比较。
后续检查使用对应版本，不直接用当前根目录脚本声称复现这个实验；`/tmp` 不是持久归档。
运行目录的 `source_snapshot/` 已保存 MicroDuck 代码和评估脚本，官方资产/公共 examples 仍有本机符号链接。
完整可转移的最小物理复现包另见第 11.5 节，不要混淆两类归档。

用户已询问“是否有脚本没关”和“如何看日志”，实际可执行命令：

```bash
# 查看并持续跟踪独立 SynReal 实验；Ctrl+C 只退出 tail，不会停止训练。
tail -n 80 -f /tmp/microduck-synreal-velocity-20261027-linear120-noise015/train.log

# 只读检查 GPU 和指定进程；PID 仅适用于本次核查快照。
nvidia-smi
ps -p 21376,50259 -o pid,etime,%cpu,args

# 新 MuJoCo 运行未重定向文本日志，使用原终端或 TensorBoard 查看。
tensorboard --logdir logs/rsl_rl/microduck_mujoco/2026-10-27_14-19-58_532936_velocity_standing_curriculum
```

先前 12:28 左右确实出现 GPU 争用：`physics_variants.py block20`（当时 PID 4192659）
在 `nvidia-smi pmon` 中占 SM 利用率约 79%～99%；另一个 `experiment.py smoke` 当时随后退出。
当时 MuJoCo 训练约 5.75 秒/轮，先前 1024 环境/24 步运行约 1.20 秒/轮；配方不同且没有隔离停进程 A/B，
不能把 4.8 倍精确归因于某个奖励。温度约 56℃、显存约 3.1/24 GB、未报告降频，计算争用是主要疑点。
这些旧 PID 不应当作仍存活；上表仅说明当时的独立 SynReal 实验和 MuJoCo 训练曾同时运行。
用户只要求查看日志，没有要求停止 SynReal 实验；不要因历史 PID 直接执行 kill，也不要读终端设备来抢占输出。

用户已反馈的旧模型回放命令，补上显式零指令便于诊断：

```bash
checkpoint_run=logs/rsl_rl/microduck_mujoco/2026-10-27_12-26-36_958920_velocity_stand_contact
python -u -m examples.microduck.play \
  --checkpoint "$checkpoint_run/model_1500.pt" \
  --solver mujoco --task velocity --viewer gl --num-envs 1 --num-frames 100000 \
  --randomize-foot-friction --command 0 0 0 --diagnostics --allow-recipe-mismatch
```

旧模型与当前新课程指纹不同，因此这只是明确允许不匹配的旧策略诊断；不会应用新奖励来修正旧模型动作。
评估新课程运行时使用其实际已保存 checkpoint，并优先不加 `--allow-recipe-mismatch`。

<a id="synreal-velocity-handoff"></a>

### 11.5 SynReal Velocity 从零训练、环境数量差异与未完成工作

**状态更正：用户后续已明确环境数量相关回放问题解决，当前聚焦站立不跟指令，见第 11.7 节。**
本节保留旧版本的真实失败记录；没有重新验证或指定修复它的原生版本，不应继续把旧故障当作当前已复现事实。

#### 用户要求与实验边界

用户要求 SynReal、`velocity`、随机初始化、`num_envs=1024`，每 300 轮保存并真实评估。
其经验预期是 300 轮基本不倒、600 轮开始跟踪；这不是保证收敛的阈值。若仍经常摔倒应诊断后改进。
最初提供的两份 MuJoCo/SynReal 历史 checkpoint 来自不同日期和修改版本，用户已确认，不能直接当受控引擎对照。
`train.py --solver` 切换物理求解器；PPO、奖励和观测共用代码，但动力学和状态适配不同。
`mjlab.py train` 是另一路官方 BAM/mjlab 训练，不能与本地共享 PD 入口混用。

以下 SynReal 实验均采用当时冻结的四项本地奖励：Z 速度 -1、静止脚速 -1.5、髋侧摆 -1、脚尖朝向 -1.5。
不含后来加入的接触/目标超限奖励和新站立课程，也未为改善 SynReal 而同步修改项目当前快照。
后两次试验将 PPO 初始动作标准差从 1.0 改为 0.15；这也是实验变量，不能声称只改了引擎。
所有运行目录的父目录为 `/home/ubuntu/work/style3d-sim/logs/rsl_rl/microduck_synreal/`。

| 标记 | 该父目录下的真实运行目录 |
| --- | --- |
| A | `2026-10-27_11-56-44_302813_from_zero_local_sync` |
| B | `2026-10-27_12-30-04_332099_from_zero_local_sync_block20` |
| C | `2026-10-27_12-36-56_603000_from_zero_local_sync_noise015` |
| D | `2026-10-27_13-17-13_839037_from_zero_local_sync_linear120_noise015` |

- **A：局部重置修正，5/30 次非线性/线性迭代，初始 std=1。** 300 轮七指令、28 环境评估仅 5/28 未失败，
  累计 43 次失败重置；前进 0.2 指令实测约 0.003 m/s，已停止。
- **B：局部重置修正，20 非线性迭代并开启 affine block preconditioner，初始 std=1。**
  约第 15 轮、控制步 355 出现数值发散，只有 `model_0.pt`；不是可用策略。
- **C：局部重置修正，原 5/30 物理，初始 std=0.15。** 保存到 `model_600.pt` 后停止。
  300/600 轮批量评估均能站住，前进 0.2 实测分别约 0.00286 / -0.0000124 m/s，未学会走路。
  同一 300 轮模型的单环境快速/完整回放均在 1000 控制步内重置 30 次；600 轮模型的单环境
  300 控制步测试，种子 42/123 分别失败 3/5 次。用户复现了增加播放环境数后不易摔倒、但仍不前进。
- **D：局部重置修正，5/120 物理，block preconditioner 关闭，初始 std=0.15。**
  300 轮批量 28 环境/20 秒无失败；单环境两种种子 × 前进/站立/转向六场景，各 12 秒均无失败、无低高度样本。
  单环境前进 0.2 的实测速度仅 0.000055 / 0.000156 m/s，仍未跟踪。
  **14:39 的 600 轮批量评估：七类指令均无失败重置，前进/后退 vx 约 -0.0000695 / -0.0000254 m/s，**
  **未达到 ±0.05 m/s 跟踪门槛。** 14:39 时单环境评估尚未完成。
  本次 16:10 核查：旧 `experiment.py` 已不在进程列表，仍无 `eval_600_single.json`；
  `status.json` 残留 `evaluating/600`，`stage_result.json` 只记录 300 轮通过，不能当作 600 轮完整验收。
  后续独立静止诊断已完成，结果见第 11.7 节；未补写旧实验缺失的结果。

早先根据 C 的 28 环境结果作出的“300 轮已稳定”判断已撤回；不能将增加播放环境数作为解决方案。
C/D 的运行目录分别有 `config.json`、`source_snapshot/` 和 `eval_*`；更早 `/tmp/.../runs` 路径只是迁移后的符号链接。
本轮已执行的物理/训练试验与本次文档更新应区分：写交接文档没有重新运行这些测试。

#### 已确认：指令到达策略，但策略没有形成有效行走

C 的回放记录显示 `actor[48:51]=[0.2, 0, 0]`；与零指令相比，14 维输出动作差的范数约 0.055。
`--command` 是策略输入，不直接设定身体速度；动作随指令变化也不代表形成了有效步态。
站立可获得姿态等奖励；在线速度其他误差为零时，目标 0.2、实际 0 的跟踪项仍为满分的约 67%
（`exp(-0.2²/0.1)`）。这解释了为什么“较高奖励且站住”不能代替速度评估，
但尚无因果消融证明奖励权重或较小探索噪声是停滞主因；已有 MuJoCo 行走基线与 D 的配方不同，不能视作同配方对照。

#### 局部 reset/push 跨环境同步修复（2026-10-28）

`MicroDuckEnv.sync_selected_state(state, ids)` 由 articulation world id 构造 FK mask；FK 和 SynReal affine proxy
同步均限制在选中环境。向量化 drive mask 同时检查关节父/子 body，只清除连接到已选 body 的 drive 历史。
`_reset(ids)` 在两份 state buffer 上使用局部同步。推力和坐姿翻倒返回实际改过的环境；只有这些 ID 触发同步，
而坐姿头部指令事件只改 command，不投影物理状态。

新增 `test_state_isolation.py` 覆盖 CPU FK mask/共享 body、多事件组合和空选集，并包含可选真实 GPU 两后端测试。
修复前，同一测试先在当前安装的旧原生库上复现 reset/push/tip 对 SynReal 未选环境的改写；
修复后，当前安装库下 4 环境 reset/push/tip × MuJoCo/SynReal 全通过，未选环境状态及 SynReal drive history 不变。
全套 `examples/microduck/tests`：83 passed、8 skipped（opt-in GPU）。未做从零长训练，不能据此声称策略收敛已改善。

本次修改了 `env.py` 与 `events.py`，改变 `recipe_fingerprint`；当前指纹见第 10 节。
已有 checkpoint 不应以旧指纹直接续训；保留其原配方做回放或开始新的配方运行。

#### 历史版本已确认：局部 reset/push 会触碰其他环境

项目原 `_reset(ids)` 和推扰事件在修改选中状态后，对全体执行 FK 和 `sync_affine_body_state()`。
四环境测试中只重置/推扰 world 0，其他世界仍出现约毫米级位移、最高约 1.14 rad/s 角速度变化，
并清除 42 个未选中世界的关节驱动历史标记。
实验中的 `sync_selected_state(state, ids)` 使用 articulation/body mask，只同步实际受影响世界；
修正后六类状态数组和驱动历史的未选中世界变化均为零。13 项相关测试和两轮 PPO 功能短测通过。
**14:36 核查项目源码仍是全局同步，补丁尚未合入。**
补丁在下方诊断目录 `local_sync.patch`；其基于早前版本，后续合入前应对照当前源码，不机械覆盖。

#### 历史版本已确认：不加载策略也能复现环境数量敏感，底层根因未知

最小脚本不导入 `MicroDuckEnv`，没有 PPO、奖励、观测、随机化、推扰或自动重置。
HOME 初态、零速度、固定摩擦；0.5 秒后左踝目标加 0.15 rad，运行到仿真时间 1.5 秒。
两组均完成 **75 控制步 × 4 子步 × 0.005 秒**，读取前等待 GPU 完成；不是多环境算得慢、还没推进到同一时刻。
固定动作可以导致失衡，不能把“没倒”视为更正确，也不能断言多环境更晚不会倒。

| 后端/设置 | 1 环境末尾倾角 | 28 环境中第一环境末尾倾角 | 全程最大机身位置差 |
| --- | ---: | ---: | ---: |
| SynReal 5/30 | 51.369° | 6.581° | 90.834 mm |
| SynReal 5/120 | 176.652° | 13.336° | 227.488 mm |
| MuJoCo 20 迭代 | 142.542° | 142.541° | 0.0233 mm |

先前带环境包装器的 120 线性迭代测试曾更接近，独立最小复测未稳定复现改善。
新进程 28 环境/120 迭代又得到 32.908°，初始化、接触排序、归约、求解收敛等都还需定位。
固定摩擦/shared-world 和轻微材质差异/非共享路径都观察到数量差异；不能声称关闭摩擦随机化就修复了。
第一环境的质量、惯量、COM、armature、PD/被动阻尼等已对齐，但原生压缩视图和每步接触输入仍待完整核验。
增加线性迭代不是已验证修复；没有证据足以指定某个 C++/CUDA 函数或把全部训练失败归因于这一现象。

已否决的另一个候选：仅开启 block preconditioner、保持 5 非线性迭代，在 1024 环境、std=0.15 的
随机动作压力测试中于控制步 197 发散。只增加到 120 线性迭代完成了 250 控制步压力测试，
但仍有正常倒地重置，且最小复现不一致；数值未崩溃不等于物理正确。

#### 最小复现、模型播放与接手操作

诊断目录及压缩包的绝对路径：

```text
/home/ubuntu/work/style3d-sim/logs/rsl_rl/microduck_synreal/diagnostics_world_count_20261027/
/home/ubuntu/work/style3d-sim/logs/rsl_rl/microduck_synreal/diagnostics_world_count_20261027.tar.gz
```

目录包含 `README_诊断.md`、`repro_world_count.py`、`compare_results.py`、XML/38 个网格、
`minimal_results.json`、`mujoco_minimal_results.json`、`isolated_28_linear120.json` 和 `minimal_comparison.png`。
`runtime.json` 保存实际加载的原生库哈希与版本；`block5_instability.pt` 是故障状态，**不是策略 checkpoint**。
其中旧 `STATUS.md`/旧曲线保留历史观察，以 `README_诊断.md` 和 `minimal_*` 的修正结论为准。

```bash
synreal_logs=/home/ubuntu/work/style3d-sim/logs/rsl_rl/microduck_synreal
synreal_python=/home/ubuntu/work/style3d-sim/.venv/bin/python
diagnostic_dir="$synreal_logs/diagnostics_world_count_20261027"

# 实际执行物理测试，需要有效 SynReal 登录；隐藏输入，不把凭据放到命令参数或文件。
# 使用新输出文件，保留原始 minimal_results.json。
"$synreal_python" -u "$diagnostic_dir/repro_world_count.py" --output "$diagnostic_dir/repeat_results.json"
python3 "$diagnostic_dir/compare_results.py" "$diagnostic_dir/repeat_results.json"

# 用户观察到的旧模型：同一 checkpoint 比较 --num-envs 1 与 28，不应把它当已修复策略。
checkpoint_run="$synreal_logs/2026-10-27_12-36-56_603000_from_zero_local_sync_noise015"
cd "$checkpoint_run/source_snapshot"
cuda_runtime_lib=/home/ubuntu/work/style3d-sim/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib
export LD_LIBRARY_PATH="$cuda_runtime_lib:${LD_LIBRARY_PATH:-}"
"$synreal_python" -u -m examples.microduck.play \
  --solver synreal --checkpoint "$checkpoint_run/model_300.pt" --num-envs 28 \
  --randomize-foot-friction --viewer gl --command 0.2 0 0 --num-frames 30000 --diagnostics
```

播放 `frame=1000` 对应仿真 20 秒；`resets` 含超时，不能直接当失败数。D 需使用 **D 自己**的源码快照和配置。
复查历史 D 时读取 `eval_600_fixed.json`；`eval_600_single.json` 在本次交接时仍缺失，不应等待旧进程继续产出。
评估期间 `status.json` 可能停在 `evaluating`，异常退出后也不保证更新；结合进程和日志判断，不只看这个字段。
脚本先保存再评估，轮号 300/600 分别代表 301/601 次 PPO 更新；`model_0.pt` 也已更新一次。

D 的门槛：批量七指令各四环境/20 秒，存活比例至少 75%、低高度样本不足 2%；
单环境两种种子 × 前进/站立/转向六场景各 12 秒，无失败重置和低高度样本。
600 轮另要求批量前进/后退平均 vx 分别至少 +0.05 / 至多 -0.05 m/s；不满足则 `STAGE_REJECTED` 并退出。
错误文案仍写 `failed stability evaluation`，也可能是**跟踪门槛**失败，要读 JSON，不能按文案误判。
批量/单环境评估都关闭逐步 Actor 噪声和推扰，保留偏差/延迟、固定 HEAD_UP；不等价于所有真实播放工况。

用户曾明确没有原生求解器源码，要求最小复现与诊断方案。以下仅在环境数量问题复发时继续；
当前下一步以第 11.7 节和已有新训练为准：

1. 保留失败模型和已完成评估；缺失的 600 轮单环境报告不能视为通过，也不假定旧进程还在运行。
2. 在相同输入下定位首次偏差：预测前后、native solve 前后、affine→body 恢复；核对局部/全局索引和接触归属。
3. 如由引擎维护者进一步定位，提供现成复现包，检查逐世界残差/停止条件/归约和缓存初始化。
   当前没有代码级根因，不能把这些排查方向写成已查明缺陷；未经用户要求不代发消息。
4. 动力学一致性、局部 reset 隔离、单环境稳定性通过后，再对探索/课程/奖励做单变量试验。
   既要验证前后/侧移/转向跟踪，也要验证停走切换；不以调大播放环境数或强制设置身体速度掩盖问题。

<a id="velocity-start-stop"></a>

### 11.6 新站立课程：600/1200/2100 的起步与停步（2026-10-27 15:29）

用户指定运行 `microduck_mujoco/2026-10-27_14-19-58_532936_velocity_standing_curriculum/`，
先反馈 600 轮零指令站稳但大指令才走、1200 轮又踏步，后补充 **2100 轮走后归零已不踏步，但约 0.5 才起步**。
后续诊断应使用这一最新反馈，不能只围绕旧模型加重静止惩罚。

源码与该运行指纹相符。本次仅加载现有模型，使用其归一化与物理开关，没有修改训练代码/配置或继续训练。
完整脚本和结果位于 `.validation/microduck-gait/standing-curriculum-comparison/`：
`REPORT.md`、`diagnose.py`、`results.json`、`transition_results.json`、`results_2100.json`、
`ramp_results.json`、`training_windows.json`。该目录被 Git 忽略，以下保留关键事实。

评估均为 MuJoCo 完整 `env.step`、策略均值、seed=42；关逐步观测噪声/推扰，保留偏差和延迟，
逐帧核对 Actor 指令。批量测试每个速度/头部组合 4 环境，每阶段 8 秒，统计后 6 秒。

| 默认 HEAD_UP、从重置直接给指令 | model_600 | model_1200 | model_2100 |
| --- | ---: | ---: | ---: |
| 0.1 指令的实测 vx | ≈0 | ≈0 | ≈0 |
| 0.2 指令的实测 vx | ≈0 | 0.021 | 0.019 |
| 0.4 指令的实测 vx | 0.027 | 0.157 | 0.288 |
| 从 0.4 归零后的双脚同时几何接触 | 100% | 6% | 100% |

速度单位 m/s。2100 的 0.2 组只有 1/4 环境移动，不能解释为全组都学会慢走。
1200 从重置直接给零指令时仍可站稳；但运动后归零每脚触地约 2.5 次/秒。
此时 `air_time=0`，静止脚速/接触惩罚分别约 -0.0463/-0.47，门控没有失效。
600 本次在 0.4/0.8 前进后都能停稳，未稳定复现其停止后踏步；0.8 超出训练范围，仅作压力检查。
1200 的 0.8 前进压力组有一次重置，其余测试无非超时重置。

**2100 单环境逐档加速更接近用户操作：** 先零指令 3 秒，再逐档 0.1/0.2/0.3/0.4/0.5 各 2 秒，最后归零 6 秒。
0.1～0.3 的实测 vx 均≈0、双脚接触 100%；0.4→0.293、0.5→0.376；最后零指令恢复双脚接触 100%。
移动数据取每档后 1 秒，停止取后 4 秒，无重置。用户的约 0.5 阈值与本次约 0.4 不完全相同，
不能声称代码写死了 0.5。批量重置直接给 0.3 可动，而先站稳再增速到 0.3 仍静止，说明评估必须包含指令历史。

原因分析和证据边界：

1. `recipe.py → update_commands` 的纯前进分支使用 `abs().clamp(min=0.3)`，并清零 vy/yaw。
   一般随机分支仍可采到低 vx，但通常同时有横移/转向；**低速纯前进缺乏专项覆盖**，不是完全没有低速样本。
2. 线速度跟踪权重 2、std²=0.1；理想站着不动时，0.1 指令仍得 1.8097/2≈90.5%，0.2 仍得 1.3406/2≈67%。
   移动则要权衡动作平滑、姿态、头部等目标，支持低速静止局部解的解释。未经训练消融，不能指定某项惩罚为唯一根因。
3. 静止脚部约束只在平移范数≤0.01、yaw≤0.01 且倾角合格时开启；实测 0.1/0.2 下为零。
   因而不是“站立惩罚直接把小速度当作零”。归零也不会清空关节状态、观测中的上次动作或行走后的物理状态。
4. 600/1200/2100 的 standing 参数 25%/30%/35%，考虑转向覆盖后零 twist 预期采样率约 21.25%/25.5%/29.75%。
   后期站立样本没有减少；action_rate 权重则为 -0.2/-0.6/-1，头部范围和 bias 项也在变化，存在多个混杂因素。
5. HEAD_UP 前两通道约 -0.349，超出 600 的 ±0.17 训练范围，但在 1200 的 ±0.39 内。
   改用零头部偏移后低速不走和 1200 停后踏步仍存在；这是回放混杂项，不足以解释全部问题。
6. 训练已有每 3～8 秒重采样，能出现运动→零，不能声称完全没有启停训练。
   当前缺少针对各速度、站稳后起步及停后重启的覆盖保证与分项验收。全局平均奖励上升不等于这些技能通过。

下一步建议：先单独改善低速纯前进采样，并覆盖“站稳→低速→零→再次起步”；保持内八约束和现有静止门控。
若仍不肯走，再对低速跟踪误差尺度做单变量对照；不同时加重 Z、静止脚速和接触惩罚。
0.5 已超过训练 vx 上限 0.4，不应以其能够起步作为正常速度跟踪的验收标准。
该诊断阶段未实施训练改动；用户随后授权后已完成第 8.4 节采样改进，尚未验证新训练效果。
2100 停稳仅有短时、单种子证据，不代表所有扰动/头部指令下均已解决。

<a id="synreal-stall"></a>

### 11.7 SynReal 静止策略诊断、用户最新决策与接手顺序

用户先澄清环境数量导致的回放差异已解决，随后确认当前问题模型就是第 11.5 节 D 的 `model_600.pt`。
该模型在 `/home/ubuntu/work/style3d-sim/logs/rsl_rl/microduck_synreal/` 下，精确相对路径为：
`2026-10-27_13-17-13_839037_from_zero_local_sync_linear120_noise015/model_600.pt`。
诊断使用该运行的 `source_snapshot/`，指纹 `41052505f33495d9326ce65c31f5be2a8b03a40fbd2f54df3ccccad997949c3d`。
它使用四项本地奖励、旧站立课程、初始 std=0.15、5/120 物理；不是当前六项奖励和 5%～25% 课程的训练产物。

已完成的诊断报告：[REPORT.md](../../.validation/microduck-velocity-stall-20261027/REPORT.md)，
图表：[diagnosis.png](../../.validation/microduck-velocity-stall-20261027/diagnosis.png)。
同目录包含 `probe.py`、`exploration_probe.py`、`training_scalars.json`、各组 `summary_*.json` 和逐步 `.npz`。
`.validation/` 被 Git 忽略，换机器时需另行保留产物；本节记录关键结论，避免只留下本机链接。
这次交接仅复核已有文件，没有重新登录、执行物理诊断或训练消融。

**交叉回放证据。** 对照策略是 MuJoCo 运行
`2026-10-27_12-26-36_958920_velocity_stand_contact/model_1500.pt`，父目录为
`/home/ubuntu/work/style3d-sim/logs/rsl_rl/microduck_mujoco/`。
每个策略/后端组合为 6 环境对应 6 类指令、8 秒、seed=123，统计最后 6 秒；关闭逐步观测噪声/推扰，
保留摩擦随机化、传感器偏差/延迟及保存的观测归一化，使用完整 `env.step`。

| 策略 | 后端 | +0.2 指令实测 vx | -0.2 指令实测 vx | +0.4 指令实测 vx |
| --- | --- | ---: | ---: | ---: |
| SynReal 600 | SynReal | -0.000168 | +0.000474 | +0.000189 |
| SynReal 600 | MuJoCo | -0.000211 | +0.000288 | +0.000014 |
| MuJoCo 1500 | SynReal | +0.084632 | -0.082933 | +0.277693 |
| MuJoCo 1500 | MuJoCo | +0.109780 | -0.103233 | +0.290055 |

速度单位 m/s，以上短测均无失败重置。目标策略在两后端都静止，而会走的策略在 SynReal 能产生行走。
因此当前证据指向目标策略未学出步态；不能归因为所有执行器失效或指令没有传入。
MuJoCo 策略在 SynReal 仍有偏航和弱侧移，且两策略训练配方不同，不能宣称跨后端复现或受控训练对照已经通过。

**指令、奖励及训练记录。** 每步核对 Actor `48:51` 与指令一致，归一化已正确加载；
目标策略有微小指令敏感性，关节有物理响应、力矩未饱和，但动作只在固定姿态附近小幅变化。
前进指令下 `standing_foot_motion=0`；改为零头部指令后仍静止，排除了这两种直接解释。
训练第 100/300/600 轮速度误差约 0.2590/0.2589/0.2620 m/s，未持续改善；
600 轮落脚样本的峰值高度均值约 0.848 mm，对应目标 20 mm，有效腾空奖励样本很少。
601 条学习率记录中 546 条为下限 1e-5，约 90.8%；配置初值为 1e-3，但未保存逐 mini-batch KL。
初始 std=0.15，600 轮学到的均值约 0.233；不能描述为探索噪声衰减为零。
短时加噪试验增加了腾空样本，也增加摔倒和惩罚，没有证明某个噪声值能修复速度跟踪。

**对“防原地踏步是否让移动成本太高”的回答边界。** 直接静止脚速/接触项有零指令门控，
不能把移动时不走直接解释成它们误罚正常步态。姿态、脚尖朝向、动作变化等项仍可能影响学习；
0.2 m/s 指令下理想静止仍有约 67% 的线速度跟踪奖励，支持站立局部解的可能性。
但尚未做训练消融，无法排除间接影响或指定某项惩罚为唯一根因；不得声称站立比例下调已经解决不起步。
用户后续只要求减少站立采样，最终指定 5%～25%；本轮没有同时增加速度奖励或改变六项本地奖励权重。
本地课程比官方前中期更高、额外存在启停组，具体数值与概率含义见第 8.3/8.4 节。

接手顺序：

1. 先检查第 11.4 节 16:08 新运行、真实 checkpoint 和保存指纹，区分旧 D 与当前配方，避免重复启动长训。
   当前新运行已确认使用最新采样；只看到 `model_0.pt` 不代表学会站立或行走，更不能作为成功模型交付。
2. 等已有运行产生可评估模型后，分六方向、低中高速记录实际速度、指令误差、失败重置及停走切换。
   包含“站稳→低速→零→同速重启”，同时检查单环境；不能只看总奖励或不摔倒。
3. 若仍停滞，先读实际新运行的学习率、动作噪声、腾空样本和课程；旧 D 的学习率结论不能直接套到新运行。
   再选择单变量对照，例如探索幅度、动作平滑课程或跟踪误差尺度，并固定其他设置和采样预算。
4. 旧物理问题仅在复发时按第 11.5 节排查；不要重新要求用户解决已反馈解决的环境数量问题。
   不擅自用 MuJoCo 策略替代从零复现，也不放宽 `--resume` 指纹检查伪装同配方续训。

### 11.8 当前 SynReal 19500 轮站立复核（2026-10-28）

用户提供的运行目录为 `microduck_synreal/2026-10-27_17-52-06_889360_velocity_stand_contact0.3_synreal/`。
其 `model_19500.pt` 使用当前指纹 `f69e030e...`、1024 环境、`init_noise_std=1.0`、脚底摩擦随机化，
累计约 4.8 亿个环境步。TensorBoard 末段平均回合长度约 995/1000、直立 cos 约 0.998，
所以“训练日志很稳定”和“单环境窗口摔倒”是两种不同条件，不能只按训练轮数判断。
该运行早于局部状态同步修复；新运行指纹已更新，不能据此声称修复改善了这份 checkpoint 或其历史训练。

同一 checkpoint 的新诊断产物在 `.validation/microduck-synreal-standing-20261028/`：
`results/*.json`、`training_metrics.json`、`fixed_control_results.json`、`heterogeneous_fixed_results.json`。
登录只通过隐藏交互输入，凭据没有写入产物。

关键结果：SynReal 单环境零指令 600 控制步，快速回放与完整 `env.step` 都是 12 次失败重置，首次约第 43 步。
完整步进、零指令、无推扰的测试结果如下；600 控制步共 12 秒，同一环境可以重复失败并重置：

| 环境数 | 发生过失败的环境数 | 失败重置总次数 |
| --- | ---: | ---: |
| 1 | 1 | 12 |
| 32 | 32 | 225 |
| 1024 | 476 | 674 |

32 环境快速回放另有 235 次失败；1024 环境使用训练指令、动作采样、观测噪声及推扰时，
829 个环境发生过失败，共 1182 次失败重置。这些短测不能证明策略稳定站立；
改变环境数还会改变随机数分配、初态和重置/同步频率，不能仅凭表格归因于原生求解器。
用户最新反馈：实际窗口中切换到 30 和 1024 环境都容易摔倒，没有观察到增加环境数就能稳定站立。
因此撤回任何“多环境已能稳定站立”的概括；诊断中的 32 环境也不是用户此次测试的精确 30 环境配置。

每步强制全局 FK 和 `sync_affine_body_state` 后，单环境仍有 8 次失败；
该处理改变了轨迹，但没有解决站立失败，尚不能据此确定根因。
MuJoCo 回放该 SynReal 策略单环境也会失败，不能把回放失败全部归因于显示窗口。

最小固定动作复现不加载策略、不含奖励或 PPO：SynReal 在 1 环境、5/30 迭代下 1.5 秒后倾角约 51.4°，
32/1024 环境约 6.5°/7.4°；同一实验 MuJoCo 的 1/32 环境约 142.5°/142.9°，环境数差异只有毫米级。
把 SynReal 线性迭代从 30 提到 120 也没有消除单环境倒地（约 176.7°），因此不能把“调大迭代次数”当作已验证修复。
给每个环境不同的极小脚摩擦以禁用 shared-world 模板后，SynReal 仍保留环境数差异；问题不只是模板共享优化。

已确认的是：此最小固定动作实验存在环境数敏感现象，当前策略在多个环境数下仍会失败。
原生求解的收敛、接触归约、缓存或停止条件都是待查方向，尚未定位到具体函数，
更未通过修复或消融证明环境数敏感是当前策略训练失败的主因。
“19500 轮策略学到只站不走的局部解”也没有得到验证；第 11.7 节旧 600 轮模型的结论不能直接套用。
用户另确认 MuJoCo 训练的策略在 SynReal 仍能站立并跟随移动，这证明 SynReal 可以执行有用策略，
但不能单独证明直接 SynReal 训练或训练/回放条件完全正确。

下一步应分别核对当前 checkpoint 的训练/回放契约与实际动作、重置/推扰的局部同步，
以及固定状态和动作下的求解差异。训练日志的批量平均值不能替代逐环境站立与移动验收；
不把增加回放环境数、盲目调大迭代或改奖励当作已验证解决方案。

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
  examples/microduck/tests/test_recipe.py \
  examples/microduck/tests/test_play_commands.py
```

- `test_gait_rewards.py`：零指令脚部 xyz 运动、头部不误罚、转向/侧移/大倾角门控、yaw 跨 ±π、HOME 轴、注册日志/重置；
  髋侧摆关节选择和大偏差纠正、物理顶限位时策略目标纠正仍有奖励差异、PD 超调余量和任务作用范围；
  静止接触项区分落地/悬空，单脚多个匹配不能替代双脚接触。
  本地站立课程的实际指令分布、转向保留、元数据与导出保留也在该文件验证。
- `test_velocity_commands.py`：六方向速度段覆盖、普通组分布、连续启停及同速重启、局部重置和到期隔离。
  5%～25% 课程调整后，该文件与 `test_gait_rewards.py` 联合运行 22 通过、1 项 GPU 测试跳过；未重跑长训验收。
- `test_vertical_reward.py`：竖直速度正反对称、水平不误罚、权重/dt、任务范围和真实 CPU 环境。
- `test_play_commands.py`：命令持续写入、zero padding。
- `test_state_isolation.py`：局部 FK / proxy / drive-history 状态隔离；真实两后端 GPU 测试需设置 `MICRODUCK_GPU_TESTS=1`。
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

本地适配优先改 `recipe.py` 等适配层；生成公式和官方源保持可追溯。
不得仅为让测试通过而同步改官方源与生成文件。奖励符号要核对函数返回值，有些官方惩罚函数自身为负。

重新导出使用官方独立环境，操作会修改源码/快照和指纹，不要在纯排查或文档整理时顺手运行：

```bash
examples/microduck/microduck_rl-develop/.venv/bin/python examples/microduck/export_recipe.py
examples/microduck/microduck_rl-develop/.venv/bin/python examples/microduck/export_recipe.py terrain
```

配方导出保留快照现有的 `local: true` 奖励、课程和 commands，包括其参数和权重；不代表所有其他手工修改都会保留。
变更后检查实际支持/省略项、公式来源、观测维度、任务表和指纹，再做必要短测。

此前文档归并阶段仅改顶层 Markdown，检查了本地链接、源码索引迁移完整性及当时配方指纹不变；
未重跑训练或声明新的物理验证结果。仓库根的历史知识库不在本次合并范围，读取时按日期记录处理。

2026-10-27 本次交接更新仅修改本文：核对当前配方指纹、5%～25% 课程与官方对照、旧 D 诊断产物，
更正环境数量问题和旧评估的过时状态，记录 16:08 新运行的实际配置及接手顺序。
文档修改不改变配方指纹；本次未启动/停止训练、未重新执行物理/PPO 测试，已有测试数保留对应实现阶段。

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
