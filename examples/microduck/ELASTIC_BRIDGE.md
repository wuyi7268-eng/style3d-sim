# MicroDuck 弹性带桥

2026-11-13 两端连接处增加布下斜托板：默认各向桥内延伸12 cm、宽度与弹力带一致，
中央56 cm保持自由悬空。托板从各端参考布面切线向下额外降低1.5 cm，按布料粒子半径留3 mm法向间隙，
避免初始穿插；受载后通过原生接触限制端部下陷，不额外固定布料粒子。
`SceneConfig.elastic_bridge_ramp_length` 控制长度（0关闭），`elastic_bridge_ramp_drop` 控制额外降低量，单位米。
入口/出口平台高度仍为0.65/0.55 m，出生距离仍为-0.05 m；下述旧数值属于历史记录。
高度查询同时考虑变形布面和裸露托板。新进程生效，未重启训练。
29项定向检查含CUDA短测通过；既有air_time奖励断言与当前零权重不符，排除后运行。
静态预览见 `.validation/microduck_bridge_ramps_20261113/preview.png`；尚未测量策略过桥冲击的改善幅度。

2026-11-12 最新修正：出口蓝色横梁、支柱和布料出口固定端均跟随 `elastic_bridge_exit_height`，
与终点平台齐平；model_750诊断后当前值为0.575 m。入口仍为0.65 m，布面参考形状在两端高度间线性过渡，
叠加原有下垂。支撑面查询及preview/HOME出生高度同步。下述保留高横梁/出口台阶的描述已失效。

修改 `scenes.py` 中 `SceneConfig.elastic_bridge_exit_height` 后重启进程生效，单位米。
该参数现在同时控制终点平台、出口蓝色横梁、支柱和布料固定端；入口保持原高度。

当前新训练使用history v3，腿部10维为XL330准静态模拟电流绝对值（A），取控制周期最后子步并按1 mA量化。
网络尺寸与v2相同；旧v1/v2回放保留原负载信号。实机接口与模型限制见 [电流反馈说明](CURRENT_FEEDBACK.md)。
按用户要求保留相对布面防滑、非脚部接触、关节限位和动作平滑约束，继续关闭七项预设步态奖励。

2026-11-12：当前桥配方已关闭步态奖励，`air_time`、`foot_clearance`、`foot_swing_height` 权重均为0；
`gait_cadence`、`gait_overdue`、`foot_swing_progress`、`commanded_foot_placement` 继续为0，课程不会重新启用。
`cloth_stall` 也已关闭。下文保留步态奖励的描述属于历史配置；已有进程不会热更新。

2026-11-12 最新出生点：从入口向桥中央移动3 cm，根节点x=-0.37 m、y=0。
`elastic_bridge_spawn_distance=-0.03`；负值进入布料，正值退到入口平台。
训练/回放/R重置查询该点实时布面高度，preview/HOME按参考网格插值高度。重启进程生效。
下面零距离和入口前3 cm的设置均为历史。

2026-11-12 最新出生点改为布料入口边缘中点：`elastic_bridge_spawn_distance=0.0`，
当前根节点坐标x=-0.4 m、y=0。训练、回放、R重置和preview/HOME均使用此值；重新启动进程生效。
下述入口前3 cm的设置为历史。

2026-11-12 最新出生点：训练、普通回放、R重置和preview/HOME统一在入口平台中线、距桥入口3 cm。
距离由 `scenes.py` 的 `SceneConfig.elastic_bridge_spawn_distance=0.03` 控制，现已重新启用该字段。
根节点x为 `-elastic_bridge_length/2 - elastic_bridge_spawn_distance`，当前为-0.43 m，y=0。
高度使用刚性平台顶面加机器人站姿高度；3 cm指根节点水平位置到桥边的距离，不是脚尖净距。
下文中央出生描述为历史。已有进程需重新启动才生效。

## 三轴IMU加速度（history v2）

最新布料Velocity网络在当前帧和20帧历史中加入三轴加速度计比力。
Actor当前74维、Critic当前83维，历史20×61；原始输入为1294/1303维，历史仍压缩为64维。
加速度来自躯干IMU安装点的物理子步速度差，包含旋转偏心效应，减去重力后转换到IMU坐标。
控制周期平均、除9.81并限幅±16g。reset初始读数为零，首个完整控制周期后有效；尚未标定实机噪声和延迟。
上一版无加速度网络保留v1布局回放；init-from可零初始化新增列，迁移旧平地或v1布料策略。
作用范围、材料、奖励和课程均保持不变。以下无加速度网络说明为v1历史记录。

## 布料 Velocity 网络（2026-11-12）

新训练自动使用独立 `cloth_history`：Actor当前71维，Critic当前80维，加过去20帧×58维，
各自通过1160→128→64的MLP编码器；20帧不包含当前帧，50 Hz时约0.4秒。
双腿各5路反馈进入当前帧及历史，排除头颈。当前仿真信号是控制周期内平均饱和PD力矩/0.96，
属于无量纲负载代理，尚不是标定后的电流反馈。真实安培映射、噪声和延迟仍需硬件标定。
仅velocity的cloth/hammock/elastic_bridge新训练使用该网络；平地及其他任务网络不变。
旧checkpoint回放按保存的standard/history结构构建。

从旧平地策略迁移无需手动转换文件：

```bash
.venv/bin/python -m examples.microduck.elastic_bridge train \
  --init-from /home/ubuntu/models/velocity_24300/model_24300.pt \
  --num-envs 32 --iterations 10000 --viewer null
```

迁移保留旧主干/归一化/探索std，逐帧扩展历史编码器；新增反馈和更早帧的输入权重为零，
使初始动作保留旧策略函数，随后通过PPO学习新输入。优化器与课程默认重建。
该改动不调整奖励、课程或材料，也不代表已启动训练或通过过桥技能验收。


## 2026-11-12 独立奖励与训练配置

当前奖励调整（2026-11-13）：`head_pose_tracking=2.0`，头部指令及全部指令课程范围固定为四维零偏移，
跟踪正常HOME抬头姿态（neck_pitch/head_pitch均为0.3491 rad，head_yaw/head_roll为0），不随机改变目标。
`head_pose_bias=0` 且其全部课程阶段为0；`pose` 本就排除头颈关节，头颈动作平滑惩罚保留。
`upright` 从2.0降为1.0。路径的 `progress_weight`、`center_weight`、`success_bonus`、`failure_cost`
均为0；路径指令、到终点停下及成功/失败终止判定保留。修改仅对新加载配方生效。

弹力桥 Velocity 已与平地 Velocity 分开配置，当前入口仍为
`python -m examples.microduck.elastic_bridge train|play`，共享入口的
`--task velocity --terrain elastic_bridge` 自动选择相同的独立配方。
保留 `task=velocity` 的机器人/观测兼容性，另记录 `recipe_profile=elastic_bridge_velocity_v1`。

| 要修改的内容 | 唯一配置/实现位置 |
| --- | --- |
| 桥上全部奖励权重、参数、课程、指令、事件、终止与基础观测 | `configs/elastic_bridge_velocity.json` 的 `spec` |
| 路径速度、成功/失败奖励和中线代价 | 上述文件的 `spec.commands.twist.path_following` |
| 桥上 PPO rollout、保存频率、学习率、熵等 | `configs/elastic_bridge_training.json` 的 `rl` |
| 桥上默认训练轮数和运行名 | 上述文件的 `training` |
| 桥专用奖励公式 | `bridge_rewards.py`；通过 `REWARDS` 分派，允许覆盖已有函数或添加新函数 |
| 布料Velocity网络、20帧缓存与腿部反馈契约 | `cloth_policy.py` |
| 平地 Velocity 配方 | `configs/official_snapshot.json` 的 `tasks.velocity` |

桥配方是独立完整快照，不在运行时继承平地奖励或课程。五项 `cloth_*` 奖励已复制到桥专用模块，
日志名保持不变；通用直立/速度跟踪等数学原语仍复用，桥专用改动应写在 `bridge_rewards.py`，
不要为了调桥而修改共享 `reward_functions.py` / `recipe.py`。
两者继续复用物理、观测适配与 PPO 执行器；这些共享代码的修改仍可能影响两者。
桥几何/材料和物理仍在 `scenes.py` / `env.py`，本次未改变物理。

桥新训练默认使用 cloth_history、10000轮、24步/轮、每50轮保存、自适应学习率（初始1e-4）、熵0.005；
init-from将旧网络迁移为新结构；resume/play保留保存的网络类型。目标KL为0.01。
显式 CLI 覆盖默认 CLI 参数；自定义 `--training-config` 的 `rl` 再覆盖内置 `rl`，
其 `ppo` 最后覆盖 PPO 参数；严格 resume 使用 checkpoint 保存的 PPO。
桥内置训练文件声明了 scope，误用于平地时会报错；短入口也拒绝其他 task/terrain/solver。

```bash
.venv/bin/python -m examples.microduck.elastic_bridge train \
  --init-from /home/ubuntu/models/velocity_24300/model_24300.pt \
  --num-envs 32 --viewer null
```

本次仅拆分，保持当前奖励、路径速度0.15–0.30 m/s、vx下限0.1 m/s、中央出生与成功判定。
修改桥专用配置/公式不会改变平地配方指纹；修改平地快照也不会改变桥配方指纹。
新训练保存完整 `resolved_task_spec`，并按任务/场景保存配方指纹。
本次共享加载器及指纹算法发生变化，旧 checkpoint 需 `--init-from` 迁移训练；
旧权重诊断回放需 `--allow-recipe-mismatch`，不放宽旧配方严格 resume，不改历史 config.json。
正在运行的进程不热更新。本次未启动长期训练，也不代表稳定过桥能力已改善。

以下为历史设置和实验记录，当前独立配置入口以上述说明为准。

2026-11-12最新更新：取消按终点距离减速，行走时身体前向指令最低0.1 m/s；
基准速度采样0.15–0.30 m/s。进入目标区域后仍置零站稳，取消低速倒退修正。
以下关于终点渐进减速及倒退修正的说明为历史，正在运行的进程不热更新。

2026-11-12更新：路径前进基准速度改为每回合均匀采样0.15–0.30 m/s；
朝向误差限速和终点减速仍保留。已有训练进程需重新启动才生效。

2026-11-12更新：按用户要求，新训练与回放（包括R重置）统一在布料中央x=y=0出生，
高度取重置后的布面；play直立、零初速度。下文入口出生描述为历史。
已有进程不热更新，旧checkpoint回放新出生位置需添加`--allow-recipe-mismatch`。

## 2026-11-11 路径穿越训练（历史）

Velocity 弹性桥现在默认采用 `bridge_path.py` 的位姿反馈路径指令，参数保存在
`configs/official_snapshot.json` 的 `velocity.terrain_overrides.elastic_bridge.commands.twist.path_following`。
从入口前15 cm的平台出生，每回合采样0.05–0.10 m/s前进速度；每个控制步按中线位置和朝向纠偏，
横移世界速度限幅0.04 m/s、偏航指令限幅0.6 rad/s，转换成身体坐标输入既有history策略。
保持Actor/Critic观测契约，位置反馈位于策略外部；实机需要相应定位来源。

出口平台与入口相同，连同横梁长0.8 m、宽0.8 m、顶面高0.65 m；不侵入柔性跨段。
平台中心为目标，靠近时减速，允许小幅倒退修正越过目标的位置。
到目标5 cm内、朝向误差≤15°、双脚已进入出口平台且均接触、直立倾角≤30°、
速度≤0.05 m/s持续0.5秒才成功。成功与跌落是真终止，不进行timeout bootstrap；
40秒时限才是截断。左右侧边和两平台外端越界均失败。
路径奖励包括沿桥X轴有符号进度、中线偏差、成功奖励与失败惩罚；原布料步态奖励保留。
共享play默认使用同一控制器；加`--manual-command --command 0.1 0 0`可关闭路径反馈，
改用速度指令及I/K、J/L、U/O键控制。R仍为唯一重置方式。

本轮保留用户材料编辑：stretch=1000、面内阻尼0、弯曲阻尼硬编码0、密度0.10、摩擦1。
下文中央出生、随机速度重采样、无出口平台、出口timeout及5000刚度的描述均为历史。

```bash
.venv/bin/python -m examples.microduck.elastic_bridge train \
  --init-from /home/ubuntu/models/velocity_24300/model_24300.pt \
  --num-envs 32 --iterations 10000 --viewer null \
  --training-config .validation/microduck_bridge_path_20261111/train_config.json \
  --run-name elastic_bridge_path_24300
```

该配置每轮128步，每50轮保存；学习率固定1e-4，熵系数0.005。
加载源history网络、探索方差和两个归一化器；新优化器及课程计数，非原配方严格resume。
路径控制不保证已学会柔性桥，需用完整穿越成功率独立验收。

## 历史场景与验证记录

新增地形 `elastic_bridge`：沿 X 方向长 0.8 m、宽 0.30 m，两端高度 0.65 m，初始中心下垂 4 cm。
两条短边各整排固定，两条长边自由；两端有横梁和支柱，桥下只有 z=0 地面，没有隐藏的硬桥板。
入口新增长0.8 m、宽0.8 m、顶面z=0.65 m的刚性平地平台，与入口横梁和桥面连续衔接。
训练仍在弹力带中央（x=0、y=0）出生，按原始桥面高度放置。
`play`回放及R重置改为在布面入口前5 cm的刚性平台中线上直立出生、朝+X，按平台顶面高度放置。
回放采用默认站姿、零初速度，不采样恢复任务的倒地姿态。
入口平台保留；静态预览仍在中央出生，使用参考桥面高度。
使用 SynReal 原生弹性膜、完整三角形面接触；不是通过外部力托住鸭子。

尺寸和材料集中在 `scenes.py` 的 `SceneConfig.elastic_bridge_*`：40×10 网格单元、451 顶点、800 三角形，
面密度 0.10、拉伸刚度 5000、弯曲和剪切刚度均为0、面内阻尼1、弯曲阻尼0。它是弹性带的膜模型近似，尚未做真实橡胶标定。
当前几何的固定端间跨度为 0.8 m；4 cm 是初始下垂，不是负载后的下垂限制。
入口平台尺寸参数为 `elastic_bridge_approach_length`、`elastic_bridge_approach_width`；
`elastic_bridge_spawn_distance` 保留兼容旧配置，当前中央出生不使用该距离。

接触摩擦由 `SceneConfig.elastic_bridge_friction` 单独调节，当前默认1.0。
该值在桥场景初始化时写入 `model.soft_contact_mu`，作用于该场景所有布料软接触（不限鸭脚），
不修改刚性入口平台的摩擦。参数要求有限且非负，0表示关闭这部分接触摩擦；修改后需重启场景。

启动和重置直接使用原始网格，布料速度和残余外力归零；`elastic_bridge_sag` 决定参考网格下垂。
不再进行启动前预模拟、平衡缓存或收敛/释放检查。第一个物理子步起正常模拟，允许自然下沉和晃动。
没有休眠、接触唤醒或位置锁定。静态 `preview` 仍不运行求解器。

阻尼分为 `elastic_bridge_damping`（面内阻尼）和 `elastic_bridge_bend_damping`（弯曲阻尼）。
弯曲阻尼默认0，替代旧代码写死的 `edge_kd=0.08`：当前轻薄网格在5 ms步长下，后者会使空桥持续大幅振荡。
只将弯曲阻尼归零的8秒独立空桥对照中，约2秒自然稳定，第3秒窗口最大位移约1.3微米；
原配置每秒窗口位移约13–25 cm。完整机器人平台零指令回放10秒，末段仍有约2–6 mm残余移动，
独立空桥结果不代表完整场景完全静止。阻尼并非越大越稳定，更改质量、网格或步长后需重新检查。
该修改不取消弹性变形，也不保证旧策略能走过桥。修改源码后须重启进程。

从仓库根目录运行：

```bash
# 静态场景预览，无需登录或 checkpoint。
.venv/bin/python -m examples.microduck.elastic_bridge preview

# 新训：默认 history 网络，显式指定规模与训练轮数。
.venv/bin/python -m examples.microduck.elastic_bridge train \
  --num-envs 32 --iterations 30000 --viewer null

# 用已有 SynReal Velocity 行走策略微调；替换为实际权重路径。
.venv/bin/python -m examples.microduck.elastic_bridge train \
  --init-from <walking_run>/model_N.pt --num-envs 32 --iterations 30000 --viewer null

# 回放桥面训练的权重。R 手动重置，跌倒和越界不会自动重置。
.venv/bin/python -m examples.microduck.elastic_bridge play \
  --checkpoint <bridge_run>/model_N.pt --use-checkpoint-environment \
  --viewer gl --command 0.1 0 0

# 同配方续训，恢复优化器、归一化和课程；环境数显式指定。
.venv/bin/python -m examples.microduck.elastic_bridge train \
  --resume <bridge_run>/model_N.pt --use-checkpoint-environment \
  --num-envs 32 --iterations 1000 --viewer null
```

同样可用共享 `train.py` / `play.py` 入口传 `--solver synreal --task velocity --terrain elastic_bridge`。
物理默认 2.5 ms×8 子步、非线性/线性迭代 5/35、关节 primal=4，控制与 history 都保持 50 Hz。
短入口默认训练 32 环境，共享地形入口默认 1 环境；32 不是经过吞吐优化的批量推荐。

只对 `velocity + elastic_bridge` 应用 `official_snapshot.json` 中的 `terrain_overrides`：
出生在弹力带中央，朝 +X，yaw 随机范围 ±0.05 rad；前进指令 0.05–0.20 m/s，横移/yaw 指令为0，
站立比例沿用 Velocity 的 5%–25% 课程。仍保留既有噪声、推扰和头部指令课程。
从桥中央开始沿桥行走，当前没有出口平台或来回过桥任务。
初始帧和重置时桥面速度为0；随后立即正常模拟，无需鸭脚接触即可自然下沉和回弹。

复用布料的有效推进、速度跟踪、停滞、相对桥面脚滑和非脚部接触奖励；日志仍使用 `cloth_*` 名称。
足部净空与根节点高度在平台上取刚性顶面、在桥上取实时变形三角形，软接触排除刚体代理；
局部 reset 只恢复选中世界的机器人与桥面，平台接触复用刚体传感器。
侧边界按所在平台/桥面宽度内缩4 cm，越界按失败处理；平台后端和桥出口内缩4 cm到边或回合耗尽按timeout处理。
平台与桥的连接处没有内部越界线；新建平台不伸进弹性跨段。
Actor/Critic 当前帧仍为 61/70 维，支持现有 standard/history checkpoint 初始化；不增加外部平衡辅助。
本场景的命令覆盖不改变平地、软垫、原四角布料或其他任务的配方。

以下为加入入口平台前的历史验证：2026-11-10 已完成静态预览、几何/重置/奖励/命令回归、真实双环境 3 秒物理短测、
双环境 2 轮 PPO、追加 1 轮严格续训与单环境 150 帧无自动重置回放。
固定端最大位移为0；HOME 固定动作下最大向下位移约33.6 cm，150帧中98帧检测到足部软接触，约1.36秒首次失败。
这证明接触、弹性变形和训练链路可运行，不代表学会行走或长期稳定；没有启动长期训练。
证据、日志、短训权重及截图位于 `.validation/microduck_elastic_bridge_20261110/`。

入口平台变更验证见 `.validation/microduck_bridge_approach_20261110/`：包括接缝几何、平台/变形面高度、
边界连续性、选择性出生/重置、静态截图和model_23400的真实回放。旧桥面checkpoint需显式
`--allow-recipe-mismatch`诊断回放或`--init-from`迁移训练，不能作为旧场景严格续训。

本次model_23400、0.15/0/0指令、6秒回放从平台正确出生，0自动重置，但偏航后从平台掉落，
未观察到桥面接触或成功上桥；场景接入不能视为旧平地策略已掌握上桥。

已移除功能的历史验证（不代表当前行为）：`.validation/microduck_bridge_settle_20261110/`。真实双环境中鸭子位于入口平台，
2秒桥面最大位移为0；选择性重置恢复平衡形状，外部速度扰动后桥面仍能运动。
