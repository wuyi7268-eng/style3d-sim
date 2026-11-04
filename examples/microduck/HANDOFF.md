# MicroDuck / SynReal 维护交接




Synreal账号密码：
wuyi_meta
YpCVTFAK

###AI不需要删除这部分内容，不担心泄露






更新：2026-10-30（Asia/Shanghai）。本文件是本目录唯一维护的交接正文。
保留当前契约、关键证据和未完成事项；旧逐次流水账及逐条奖励源码索引已移除，原实验产物未删除。
精简前内容见 [恢复备份与来源说明](archive/2026-10-30_before_simplification/README.md)，不作为当前维护指令。
文档与代码冲突时检查源码；解释历史实验时读取该运行的 `config.json`、checkpoint 和对应代码版本。

## 1. 接手原则与目标

- 当前主线：同一 MicroDuck 策略在 SynReal 平地和海绵垫上稳定站立、运动、启停，避免平地能力退化。
  另有独立 StandUp 专家训练；VelStand 已接入 history 双教师 PPO + BC，融合技能尚未完成收敛验收。
  长期方向包括布料/软垫、水中、吊床、沙地、台阶，并融合到一个策略；这些不代表已实现的技能。
- **项目任务清单（恢复自原维护文件）：**
  1. 布料、柔性垫子上行走：当前海绵软垫混合训练属于进行中的柔性支撑面工作；布料薄片是不同物理模型，未完成。
  2. 水里（泳池）游泳：原生粒子流体水池已接入 `scenes.py`、`--terrain pool`；游泳奖励与策略技能尚未实现。
  3. 吊床上行走：基础下垂布面场景已加入统一 `scenes.py`，可选择用于 SynReal 训练及 checkpoint 回放；尚未做物理短测或技能训练验收。
  4. 沙地上行走：原生颗粒床场景已加入统一 `scenes.py`，可选择用于 SynReal 训练及 checkpoint 回放；尚未验证颗粒接触动力学或行走技能。
  5. 台阶行走：实体阶梯场景已加入统一 `scenes.py`，可选择用于 SynReal 训练及 checkpoint 回放；尚未做物理短测或技能训练验收。
  6. 建立一个总 policy 融合以上地形/运动技能：未实现；当前按任务和场景配置训练，不能视作融合策略。
  顺序与内容恢复自本地编辑器历史；状态仅描述当前进度，不代表其余项目已排期或完成方案设计。
- **不要停止健康的用户训练。** 用户已授权：确认训练崩溃后排查、修复，并从合适的有效 checkpoint 恢复。
  进程消失不等于崩溃；先核对日志、退出原因和用户是否主动停止，避免重复启动。
- 先看 `git status --short`。工作树包含大量未提交修改、删除和未跟踪模块；禁止擅自 reset/clean。
  转交代码不能只保留已跟踪文件。不要覆盖用户正在编辑的 `env.py`。
- 根目录 `.venv` 用于本地训练；官方子项目独立 `.venv` 用于官方入口/导出。
  修改遵守根 `AGENTS.md`：行宽 128，必要代码注释用英文。进入官方目录先读其自己的 `AGENTS.md`。
- 环境/求解/材料默认值集中在 `env.py`；奖励、课程、事件和指令集中在 `configs/official_snapshot.json`。
  不把权重散放回 Python 常量，不再新增重复的环境配置文件。
- 凭据只通过正常隐藏输入或进程环境使用；不得写入文档、脚本、报告、Git diff 日志。
  `--force-login` 可能替换已有会话，不作为后台诊断的默认选项。
- 源码、参数、原生库不会热更新已运行进程。不要篡改历史 `config.json` 或指纹伪装同配方续训。
- GPU 评估会争用训练算力；先小批量、短时间测试。文档整理不启动动力学测试或额外训练。

## 2. 当前训练与场景快照

### 2026-11-04 新步频平地运行 4200 / 9000 轮行走对照

运行 `2026-11-02_23-31-50_041508_velocity_cadence_history_flat_1024` 的4200轮权重不迈步、9000轮会走，
已用同条件 SynReal 平地短测复现。恢复保存物理5 ms×4、非线性5/线性35、primal=1，
均值动作，关闭逐帧噪声/自动推扰/摩擦随机化，保留固定抬头指令及校准偏差/延迟。
16环境分为0/0.1/0.2/0.3 m/s指令各4个，8秒测试统计末7秒；两组0重置、统计帧均保持直立。
4200各非零指令的平均前进速度接近0，9000分别为0.087/0.173/0.251 m/s。
4200指令已影响网络动作输出，支持“站稳但未形成推进”的行为结论，不能归因于指令未送入网络。
训练4200/9000前200轮平均腾空奖励0.175/0.935、净空奖励0.214/0.727，抬脚指标在4800–6000轮上升；
步频奖励仍仅约1.6e-6/4.4e-6，未验收步频目标。未筛选中间权重或证明9000为最优/首次会走。
进一步固定0.2 m/s指令分析：4200均值/采样动作速度约0/0.0068，9000约0.176/0.174；
4200均值站立仍有7.869奖励率，采样增大运动成本却未得到腾空收益，9000均值靠腾空+净空约2.487补偿迈步成本。
步频门控要求落脚时另一脚也稳定接触；9000均值152个落脚候选均因另一脚离地被拒，步频奖励0、逾期罚饱和。
这是几何接触传感器判定，不等同于支撑力；当前门控对这组已会走的步态仍无正反馈。
已从运行git diff重建快照，velocity配方及保存的local配置与当前一致，课程最晚约2000轮结束；
奖励竞争与稀疏塑形支持早期站立局部策略的解释，未做训练消融，不能确认某一项为唯一根因。
当前源码指纹与历史运行不同，本次为允许差异的诊断回放；没有修改训练/环境/奖励或启动训练。
复现脚本、逐指令数据、训练指标、哈希和限制：`.validation/microduck_4200_vs_9000_20261104/REPORT.md`。

### 2026-11-03 清透水面显示

`SceneConfig` 增加 `pool_water_color=(0.30,0.72,0.82)`、`pool_water_opacity=0.22`、
`pool_water_roughness=0.08`。Newton GL 材质接口只接受RGB，场景内使用局部网格适配器，
在不透明物体之后执行常量alpha混合、关闭水面深度写入并恢复GL状态；水面不投实体阴影。
这是半透明显示近似，不含物理折射/深度吸收；USD分支写入displayOpacity，未做USD视觉验收。
CUDA OpenGL已核验池底和入水机器人躯干可透视，重复重建与关闭窗口无报错；显示回归7通过。
截图与脚本：`.validation/microduck_clear_water_20261103/`。没有改流体动力学，保留用户当前密度5000。

### 2026-11-03 原生流体游泳池场景

统一 `scenes.py` 新增 `pool`，`train`/`play --terrain pool` 可选，默认1环境。
`SceneConfig.pool_*` 配置池内长0.6 m、宽0.4 m、水深0.18 m、池壁高0.26 m/厚0.025 m，
水粒子间距0.02 m，共5400粒/世界；密度1000 kg/m³、体积模量200 kPa、动力黏度0.001 Pa·s。
采用原生 `configure_fluid` 压力/黏性与双向形状接触，不额外注入浮力或修改机器人质量。
Poly6离散静止体积用于粒子质量与固定边界样本；池底/四壁为碰撞盒，顶部开放，无水面碰撞盖板。
finalize 后按拓扑排除 affine proxies，逐世界校验水粒子数；只重置选定世界的水粒子位置/速度。
出生在水面上方，地形高度观测以池底为基准，不把水面当作硬支撑；沿用所选任务的现有奖励/终止逻辑。
CUDA GL/USD 使用随实际粒子重建的水面网格，CPU显示回归使用蓝色粒子层；不依赖全局Show Particles。
场景/显示/共享物理测试36通过、2跳过，含双世界选择性重置和单/三世界显示偏移、代理排除。
真实 `velstand_his/model_4550.pt`、恢复保存物理、单世界无窗口回放30帧（0.6秒）、0重置，
逐帧水位置/速度有限、接触未溢出；末帧水范围约x±0.291/y±0.192/z=0.006–0.165 m。
静态OpenGL水面截图与回放脚本/日志：`.validation/microduck_pool_20261103/`。
这是水池场景和短步进验证，不是长期流体标定或游泳技能验收；现有陆地权重不会因此自动学会游泳。

### 2026-11-03 沙粒尺寸缩至毫米级

沙床从 1.4×1.2×0.24 m、间距12 cm、直径约11.5 cm的大球，改为鸭子足下0.32×0.24×0.008 m小床；
规则间距2 mm、球半径0.96 mm（直径1.92 mm），4层共76,800粒/世界，属于毫米级粗砂近似。
沙粒之间走原生颗粒求解，但与托盘/机器人仍使用软接触记录，保留按粒子数乘8分配的接触容量。
多世界训练的颗粒数按每世界76,800线性增加；实际吞吐和颗粒接触动力学仍需目标GPU测量。
球形离散颗粒不是粒径分布/多面体砂粒，当前薄小床只供单世界物理和策略短测，不代表真实砂土标定。
沙地 profile 的训练默认环境数改为1（其他场景仍为1024），可由CLI显式覆盖；1024世界将复制约7860万沙粒，
不适合作为当前验证默认。
场景/显示回归17通过、2跳过；CUDA OpenGL截图与脚本在 `.validation/microduck_sand_size_20261103/`。
单世界固定HOME动作、2.5 ms×8/关节primal4短测10控制帧（0.2秒）完成，无非有限状态或接触容量溢出，
峰值约8.4万软接触。机器人落下后颗粒飞溅最高约0.248 m、速度分量约3.4 m/s，且部分穿入底板毫米级；
保留原材料参数的本次尺寸修改尚不能视作真实沙子动力学标定或稳定行走验收，需后续接触参数/积分验证。

### 2026-11-03 VelStand 回放物理配置不匹配

用户回放 `velstand_his/model_4550.pt`、平地30世界、vx=0.2、关闭跌倒重置时，
采用当前 `env.py` 的 5 ms×4、非线性5/线性50、关节 primal=1，覆盖了该权重的
2.5 ms×8、非线性5/线性35、primal=4。原诊断 `20261103_144420_796814_unstable.pt`
记录环境0线速度分量118.94 m/s，触发100 m/s阈值；前一子步已异常。
同权重30世界无窗口对照保留可视回放的指令持有语义、固定足摩擦、无跌倒重置：
当前配置在 control_step=109257/substep=2 复现失稳；恢复 checkpoint 配置完成100控制帧、0重置。
证据：`.validation/microduck_velstand_replay_20261103/`。这是2秒短测，不是长期稳定验收，
也未隔离步长与关节迭代的各自贡献。原生缓存未包含在失败诊断内，不能据此声称定位了原生缺陷。
此权重回放应加 `--use-checkpoint-environment --no-randomize-foot-friction` 并显式选择所需地形。
配置差异提示和启动日志现包含 `joint_primal`，并提示恢复保存配置；没有改当前默认物理或保护阈值。

### 2026-11-03 沙地默认显示修复

Newton 默认关闭 `Show Particles`，此前沙地只显示托盘；现在 `SceneTerrain` 独立绘制已注册沙粒，
不依赖全局粒子开关，也不包含机器人 affine proxies。单世界沙地回放默认拉近相机。
共享 render 现为可选场景调用 `display_state`，粒子/布面使用与刚体一致的世界显示偏移，不修改物理状态。
场景与渲染定向回归 17 通过、2 个 GPU 物理/预览测试跳过；单/三世界覆盖隐藏全局粒子时的显示、
代理排除、逐帧位置更新、世界偏移和物理状态不变。真实 CUDA/OpenGL 静态截图确认 240 沙粒可见：
`.validation/microduck_sand_render_20261103/sand.png`（同目录保留脚本和日志）。
本次仅验证显示；当前半径 0.0576 m 的粗颗粒外观仍为球形，不代表细沙或长期行走能力已验收。

### 2026-11-03 吊床布面漏检修复

吊床 15×15 顶点对应约 12.9×6.4 cm 网格，原来仅启用粒子—刚体接触，缺少边/三角形内部接触。
固定 HOME 动作、同一材料/步长的真实 SynReal 对照复现：0.22 秒躯干 z=0.853 m，
实际布面 z=0.927 m，机器人已经穿下布面。启用完整面接触后，同一时刻躯干 z=0.977 m、
布面 z=0.851 m；80 帧/1.6 秒采样内未再整只穿下布面，但后段发生侧倾，并非稳定站立验收。
`add_scene(hammock)` 现预构建机器人碰撞网格 SDF；共享环境为吊床启用
`enable_rigid_soft_full_surface_contact`，沿用既有软表面接触容量配置，不增加硬盖板或改材料。
场景定向测试（含 CUDA 面接触/静态预览）14 通过；共享物理/软垫接触回归20通过。
`model_18000.pt` 单世界、vx=0.2 m/s 无窗口回放40帧、0重置；尚未验收长期策略表现或GL显示。
证据及对照脚本：`.validation/microduck_hammock_contacts_20261103/`。

### 2026-11-03 沙地颗粒注册修复

`SceneTerrain.finalize()` 原先按正粒子半径选择沙粒，误包含 SynReal finalize 新增的刚体 affine proxies，
导致原生初始化报 `granular particles cannot also belong to elastic or affine topology`。
现在按 finalize 后的 affine/弹性拓扑排除非沙粒，并逐世界校验实际沙粒数等于场景生成数量。
真实 MicroDuck 单世界为 240 沙粒＋60 代理粒子；注册仅保留 240 沙粒，三世界也通过归属与位置核对。
此前只有空沙床的注册测试没有覆盖机器人代理，本次已补上真实机器人单/多世界回归。
场景测试 12 通过、1 个独立预览 GPU 测试跳过；同 `model_18000.pt`、单世界、vx=0.2 m/s
完成 SynReal 无窗口回放 10 帧、0 重置。日志：`.validation/microduck_sand_indices_20261103/replay.log`。
这仅验证初始化和短步进链路，未验证 GL 显示或长期沙地行走稳定性；没有改沙床尺寸、材质或机器人动力学。

### 2026-11-02 23:31 新步频奖励平地 Velocity history 训练

用户要求新奖励改完后开启 SynReal 平地 Velocity，1024 环境、30000 轮、历史编码器。
已从头启动，无 resume/init-from/教师 BC；任务 velocity、official profile、terrain=default、seed=42。
使用 `configs/rtx4090_synreal_official_adapted.json`，显式 `--policy-variant history --num-envs 1024 --iterations 30000`。
物理沿用当前平地默认：0.005×4、非线性5/线性35、primal=1；PPO rollout=24、save_interval=300。
运行目录 `logs/rsl_rl/microduck_synreal/2026-11-02_23-31-50_041508_velocity_cadence_history_flat_1024/`。
启动 PID=3885670；启动记录与标准输出在 `.validation/microduck_cadence_history_flat_20261102_233108/`。
首次后台启动因缺少认证环境在训练前退出；后续通过进程环境提供凭据正常启动，未强制替换会话。
已核对保存配置为 HistoryActorCritic、1024/30000，新奖励 gait_cadence=0.5、gait_overdue=-0.1；
首10轮完成、model_0.pt已保存，初始约2.35秒/轮。此为启动验证，非收敛验收或持续监测承诺。
任何后续操作先重查 PID 与日志，不能按此快照推断实时状态。

### 2026-11-02 当前软垫材料默认值

用户要求网格数量随垫子尺寸自动变化。`env.py` 的 `PAD_CONFIG` 现在用 `cells_per_meter`（每米格数）配置，
可传单个数或 X/Y/Z 三个值。当前保留用户最新尺寸 1.2×1.2×0.12 m，密度为 (37.5,37.5,50)，
解析得到 45×45×6、72900 个四面体，与改动前拓扑一致。只改 `size` 就会重新计算网格数。
规则为每轴 max(1, ceil(size×cells_per_meter))，浮点整数边界按数值容差处理，单元尺寸约为密度的倒数。
`cells=None` 表示自动模式，`resolved_cells` 获取实际网格数；显式 `cells` 优先，兼容旧 checkpoint 和诊断覆盖。
新 checkpoint 同时记录密度配置和 `terrain.pad_cells` 实际计数；旧 checkpoint 仍使用原固定数量。
独立场景支持 `--cells-per-meter D` 或 `--cells-per-meter DX DY DZ`，与 `--cells` 互斥；
`--size` 在自动模式下会重算数量。启动日志打印实际 cells 和每垫四面体数。
网格密度相关定向回归 100 通过、1 跳过；已核对当前自动网格与原 45×45×6 顶点/四面体完全一致，
且真实旧 `model_400.pt` 旁配置仍恢复 24×24×3 固定网格。本次未新增动力学测试。
用户最新材料为 shear_modulus=60 Pa、lame_lambda=7000 Pa、damping=35，本次网格接口改动保留这些值。

此前用户要求明显下陷后几秒内缓慢恢复、少弹跳，近似记忆海绵。对应历史验证几何为
1×1×0.4 m、60×60×6，水平约 1.67 cm、竖直约 6.67 cm，共 129600 个四面体。
该历史验证材料为 shear_modulus=100 Pa、lame_lambda=7000 Pa、damping=200，
density=80、contact_stiffness=5000；使用现有原生弹性＋材料阻尼近似慢回弹，不是新增塑性或真实泡沫本构。
阻尼同时减慢压入和恢复。较高阻尼的实际 checkpoint 单世界、直立起点回放验证了渐进下陷，
但旧策略出现失衡，不能声称已经适应新材料；最终配置的实测范围以报告为准。
独立零重力、无接触的 3 cm 初始局部凹陷释放对照用于测量材料恢复；脚本、数据与限制见
`.validation/microduck_memory_foam_20261102/REPORT.md`。定向测试 49 通过、1 跳过。
该历史几何、mu=100 的同条件测试中，damping=5 约 0.2 秒恢复 90%，damping=200 约 5.6 秒，
后者无采样反弹超调；damping=500 在 8 秒内未恢复 90%，当时选择 200。
该历史配置实际回放 3 秒无保护故障/重置，但局部凹陷最高约 5.5 cm 且旧策略失衡；
慢回弹材料验证不等于此前 2–3 cm 凹陷目标或策略稳定性验收。
此前 6×6×0.6 m 垫子因回放卡顿曾从 60×60×6 降为 30×30×3；不再是当前用户选择的几何。
批量场景软接触容量按 max(4096×环境数, 2×软表面三角形数) 分配；原固定容量在细网格初始地面接触时即溢出。
实际 SynReal 同 checkpoint 单垫、直立初始姿态对照测量脚下相对周围 0.3 m 采样点的局部凹陷，
避免将整块垫子的沉降当作脚印；证据与验证范围见 `.validation/microduck_footprint_20261102/REPORT.md`。
此前 60×60×6 配置单垫实测 6 秒，第 3–6 秒局部凹陷 26.95–30.70 mm，平均 28.69 mm；无重置或保护故障。
初始落地有瞬态波动，10 cm 网格形成双脚附近局部凹坑，不等于精细分离的脚掌印；定向回归 69 通过、1 跳过。
历史 2/2 Pa、6 cm 厚配置曾记录局部体积比 0.239/0.201，触发旧 0.25 阈值重置。
用户明确要求关闭过度压缩保护：`SoftPadGuard.settings.min_volume_ratio` 现为 0.0，
允许任意正体积压缩；零体积、翻转、非有限状态的回滚/重置和故障频率限制仍保留。
本次仅短时回放和局部凹陷验证，未完成长时间行走或训练稳定性验收。
仅对新建且采用当前默认值的场景生效；`--use-checkpoint-environment` 仍恢复 checkpoint 中的旧材料。
海绵垫 profile 的默认物理设置为 physics_dt=0.0025、decimation=8、joint_constraint_primal_iterations=4，
控制周期仍为 0.02 s；普通训练/回放无需再显式传入 `--physics-substeps 8 --joint-constraint-primal-iterations 4`。
显式 CLI 可覆盖，`--use-checkpoint-environment` 仍恢复保存的物理；纯平地和 HOME 使用各自原有默认设置。
下述训练快照保留各次运行的历史参数，不代表当前默认值。

### 2026-11-01 VelStand history 软垫微调

用户指定从 `2026-11-01_16-15-53_velstand_history_512_joint4/` 迁移至软垫，256 环境、比例 0.4，
并明确选择 **PPO 微调、保留 history、暂停双教师 BC**。固定源权重为 `model_4450.pt`，不随原目录更新。
配置：`configs/rtx4090_synreal_velstand_soft_pad_history.json`；显式 CLI `--num-envs 256 --pad-probability 0.4`。
用户在停止训练后要求课程重新开始：当前 JSON 已设 `inherit_training_stage=false`。
下一次从原平地权重 `--init-from` 时，保留 history 网络、动作 std 和双归一化，轮次/课程计数从 0 开始，
优化器重新初始化，仍为纯 PPO、256 环境、软垫比例 0.4。启动命令必须去掉 `--inherit-training-stage`，
也不要用 `--resume`（它会恢复保存的课程）。本次仅修改配置，没有重新启动训练。
下述从 4451 开始的记录属于已停止的历史实验；其验证目录 `run_training.py` 专用于当时的继承课程实验，
不用于新的课程重启。新实验使用正常 `examples.microduck.train` 入口。
154 个平地＋102 个软垫，使用当前 `env.py` 的 6×6×0.06 m、24×24×3 网格、base_z=0 软垫；
注意这不是早期 base_z=-0.06 的齐平凹槽垫，软垫未变形顶面为 0.06 m。材料与几何本次未改。
此前已停止的实验继承学生 Actor/Critic、两套 history 编码器、动作 std、归一化和成熟课程，从 4451 继续编号；
优化器重置，固定 LR=1e-4、entropy=0.001、rollout=24、save=50，计划追加 60000 轮。
物理保留源配置：2.5 ms×8、非线性 5 / 线性 35、关节 primal=4，动作/历史仍为 50 Hz。
这属于显式地形迁移，不是旧平地环境的严格 `--resume`；新 checkpoint 后续可按自身环境严格续训。
原平地 PID=3171780 启动本实验时仍健康运行；用户随后明确要求关闭原平地训练，已发送 SIGINT 并确认退出。
最后持久化平地 checkpoint 为 `model_4550.pt`；停止记录见原 recurrence 验证目录的 `user_stop.json`，
状态标为 `stopped_by_user`，不得当作崩溃自动恢复。软垫 PID=3314870 保持运行，关闭时已到第 4467 轮。
软垫启动 PID=3314870；实时路径 `.validation/microduck_velstand_soft_pad_20261101/training_status.json`，
输出日志同目录 `training.log`，启动记录 `production_process.json`。任何操作前重查 PID 与状态。
用户随后也明确要求关闭软垫训练：已向 PID=3314870 发送 SIGINT 并确认退出，状态为 `stopped_by_user`。
停止前日志到第 4482 轮，尚未到下一保存点 4500；目录中只有迁移初始化的 `model_4450.pt`，
本次正式软垫训练的后续更新未保存。停止记录在同目录 `user_stop.json`，不得当作崩溃自动恢复。
迁移/软垫/BC 定向回归 73 通过、1 跳过；补充 BC 不继承断言后迁移套件 16 通过。
真实 SynReal 8 环境×3 轮完成，逐项确认网络/双归一化迁移一致、无软垫保护故障；不等于长期技能验收。
正式运行目录 `logs/rsl_rl/microduck_synreal/2026-11-01_22-07-40_939388_velstand_history_soft_pad_256_p04/`；
启动后已核对 4451、4452 两轮损失有限，软垫变形/翻转故障计数为 0，共享 GPU 时每轮约 60 秒。

### 2026-10-31 新平地 history 训练（优先核查此记录）

用户明确要求从头训练、1024 环境、持续监测，确认崩溃后查因并续训；随后确认线性迭代使用当前值 35。
10:23（Asia/Shanghai）已启动 velocity_history_flat_supervised，任务 velocity、纯平地、history 独立双编码器，
seed=42，目标 30000 轮；物理 **5/35**、dt=0.005、4 子步，其余沿用当前 env.py 和既有训练 JSON。
运行目录：logs/rsl_rl/microduck_synreal/2026-10-31_10-23-25_454469_velocity_history_flat_supervised/。
启动训练 PID=2420345、监测 PID=2420344；任何操作前重新确认进程身份与状态。
实时入口为 .validation/microduck_supervised_20261031/status.json，详细操作见同目录 README.md。
监测每 10 秒更新；额外保留最后 64 个物理子步前的状态/动作，崩溃时写 failure_trace.pt；不改奖励或物理保护。
用户后续要求：详细检查及汇报每 20 分钟一次；崩溃仍及时处理。
初始 model_0 已加载核对网络、优化器、双归一化参数均有限；这不代表收敛或长期数值稳定。
监测器只报告崩溃/停滞，持续 agent 任务负责查因、验证修复及 checkpoint 恢复，不能无条件循环重启。
下列 10-30 记录为历史快照，不应据其 PID/目录判断当前训练。

2026-10-30 14:55 只读核查：PID `1958708` 仍在运行，14:46:29 启动，命令如下（记录用途，勿重复启动）：

```bash
.venv/bin/python -u -m examples.microduck.train \
  --training-config examples/microduck/configs/rtx4090_mixed_soft_pad.json \
  --init-from logs/rsl_rl/microduck_synreal/2026-10-28_10-04-52_798619_velocity_PCG/model_13500.pt \
  --num-envs 128 --pad-probability 0.3 --iterations 30000 \
  --run-name velocity_inset_pad_finetune --viewer null
```

运行根目录为 `logs/rsl_rl/microduck_synreal/`，当前运行子目录：
`2026-10-30_14-46-36_838789_velocity_inset_pad_finetune/`。
TensorBoard 事件文件中的 PID 与进程一致，14:55 仍有更新；当时最新保存为 `model_100.pt`。
另有 `2026-10-30_14-48-33_254992_velocity_inset_pad_finetune/`，事件 PID 为 `1959886`，
最后事件修改时间 14:49，仅见 `model_0.pt`；**不能按目录最新时间把它误认成当前训练**，退出原因本次未判定。
PID、轮次和存活状态仅为时间快照，任何后续操作必须重查。

已核对当前运行保存配置与 `env.py` 的主要场景参数一致：

| 项目 | 当前值 / 含义 |
| --- | --- |
| 后端 / 任务 | SynReal / velocity，`mixed_soft_pad` |
| 环境 | 128：前 90 个纯平地，末 38 个软垫；固定分组 |
| 软垫 | 6×6×0.06 m，`base_z=-0.06`，未变形顶面 z=0，固定底面 |
| 网格 | 24×24×3；水平单元 **25 cm**、竖直 2 cm；每垫 10368 个四面体 |
| 材料 | density=80、shear_modulus=2000、lame_lambda=10000、damping=5 |
| 接触 | contact_stiffness=5000、particle_radius=0.0015 |
| 地板 | 软垫组为四周地板＋凹槽底板；垫面上方没有 z=0 硬盖板 |
| 出生 | `fixed_groups_center`；软垫组每次 reset 回到 XY=(0,0)，其他姿态随机化保留 |
| 训练边界 | 软垫组根节点任一轴绝对值 >2.9 m 时 timeout；纯平地组半径 >2.5 m |
| 物理 | dt=0.005，4 子步，控制周期 0.02 s；非线性 5 / 线性 15 |
| 其他求解 | primal=1、block preconditioner=False、air_density=1.225、implicit Euler |
| 正常训练 | Actor 噪声、动作探索、PPO 更新开启；摩擦随机化开启 |
| 摩擦 / 推扰 | 每环境双脚共享摩擦 [0.7,1.3]；每 3–6 秒 X/Y 速度增量各 ±0.3 m/s |
| PPO | rollout=24、save=100、固定 LR=1e-4、entropy=0.001、计划 30000 轮 |
| 初始化 | 加载原平地 13500 的网络、动作 std 和归一化；优化器/新课程计数重置 |

**尺寸改变同时改变了离散精度。** `env.py` 中“保留 5 cm 水平单元”的注释已不符合实际，本次仅记录，未改代码。
旧 1.2 m / 24 网格是 5 cm；当前 6 m / 24 是 25 cm，不能当作材料动力学完全相同的加长垫。
若保持 5 cm，需要 120×120×3，四面体数为当前的 25 倍，不能直接假定现有训练开销可接受。

旧 1.2 m 垫的训练阈值为 ±0.5 m，从中心按实际 0.3 m/s 理想匀速仅约 1.7 秒就复位。
因此“短 episode 限制长距离稳定性学习”是合理假设，但尚未做保持网格间距等条件一致的长短垫训练对照。
当前加大尺寸并不能单独证明这一假设或解决头颈失稳。

监测入口注意：旧 `.validation/microduck_monitor_20261029/watch.py` 绑定旧运行，不会自动跟随新训练。
14:55 的本机 `6006` TensorBoard 仍指向 **10-29 旧运行**，不是上述 6 m 新运行。
本次未建立新的持续监控任务，不能据旧脚本/旧面板声称新训练已持续评估。

## 3. 代码导航与配置契约

| 文件 | 职责 |
| --- | --- |
| [env.py](env.py) | 唯一共享环境；PhysicsConfig、PAD_CONFIG、EnvironmentConfig、PD、步进与重置 |
| [common.py](common.py) / [train.py](train.py) | CLI、配置与登录；PPO 新训、严格续训、迁移初始化 |
| [checkpoint.py](checkpoint.py) / [ppo.py](ppo.py) | 权重/归一化迁移与哈希；任务 PPO、首帧归一化适配 |
| [policy.py](policy.py) | 可选历史 Actor–Critic；独立历史编码器与复用当前帧统计量的归一化 |
| [play.py](play.py) | 本地 checkpoint 回放、交互指令、回放边界；不传 checkpoint 则是官方 walk 演示 |
| [scenes.py](scenes.py) | 统一场景：软垫/凹槽、吊床/沙地/台阶、Rough/Slope；接触、分组、重置、保护与单世界回放 |
| [policy_spec.py](policy_spec.py) | 资产、规范关节顺序、HOME、PD 增益、碰撞过滤 |
| [recipe.py](recipe.py) / [velocity_commands.py](velocity_commands.py) | 配方/课程/指纹；六方向低速与连续启停采样 |
| [reward_functions.py](reward_functions.py) | 上游奖励公式与本地步态、垫面脚滑、推进奖励实现 |
| [reward_functions.py](reward_functions.py) / [velocity_rewards.py](velocity_rewards.py) | 提取的官方奖励函数 |
| [observations.py](observations.py) / [events.py](events.py) | 观测噪声/延迟；摩擦、重置、推扰与课程 |
| [adapter.py](adapter.py) / [sensors.py](sensors.py) / [reset_functions.py](reset_functions.py) | 状态约定、几何接触和重置公式 |
| [stability.py](stability.py) | 物理异常快照 |
| [task_registry.py](task_registry.py) / [任务快照](configs/official_snapshot.json) | 任务映射；奖励/事件/指令/课程与任务 PPO |
| [evaluate_standup.py](evaluate_standup.py) | 四类起点站起评估 |
| [ui.py](ui.py) / [ui_backend.py](ui_backend.py) / [ui_assets](ui_assets/) | 本机中文训练与评估工作台 |
| [mjlab.py](mjlab.py) / [export_recipe.py](export_recipe.py) | 官方独立训练/ONNX 路径；配方与地形导出 |

`train`、`play`、`scenes` 都委托 `MicroDuckEnv.advance_physics()`，不是三套动力学实现。

2026-11-03 场景模块合并：`soft_pad.py`、`mixed_terrain.py`、`pad_contacts.py`、`terrain.py`、
`soft_pad_policy.py`、`soft_pad_scene.py` 的实现统一移入 `scenes.py`，旧文件已删除。
训练/普通回放入口及 `--terrain` 参数不变；独立软垫入口改为
`python -m examples.microduck.scenes --preview` 或 `python -m examples.microduck.scenes --checkpoint <model.pt>`，
其余参数保持原约定。场景导入统一使用 `examples.microduck.scenes`；归档命令保持历史原文。
本次为文件归并，保留已有场景物理；同时修复静态预览对空求解器调用 `prepare_contacts()` 的入口错误。
CPU 预览仍受当前纹理 SDF 只能在 CUDA 构建的限制。配方指纹改为覆盖统一文件，
旧 checkpoint 诊断回放仍需显式 `--allow-recipe-mismatch`，不能伪装为旧源码的严格续训。
合并后全目录回归 369 通过、17 跳过；新增独立入口 CUDA 单帧静态预览测试另行通过，
清除认证环境后确认无需登录和求解器。没有启动训练或新增动力学验证。

HOME 保持有明确不同配置：dt=1/480、8 子步、5/30、primal=3、block=True、air_density=0。
HOME 测试不能当作策略 50 Hz 配置的步态验证。

配置顺序：显式 CLI 覆盖训练 JSON 的 `training` 默认值；环境默认来自 `env.py`。
PPO 顺序为 `ppo.py` 默认 → 任务 PPO → JSON `ppo`；严格 resume 使用保存的 PPO。
默认配方为 `official_adapted_v1`，观测契约为 `official_asymmetric_pd_v2`；旧简化配方已删除。
`--run-name` 只是标签。`--init-from` 与普通回放默认使用当前环境；checkpoint 只帮助选择地形 profile。
`--use-checkpoint-environment` 才显式恢复已保存的物理/地形设置，未保存字段不能当作已恢复的历史事实。

| 加载方式 | 契约 |
| --- | --- |
| `--resume` | 同任务/后端/配方及环境校验；恢复优化器、归一化和课程；iterations 表示本次追加量 |
| `--init-from` | SynReal 同任务 Velocity/VelStand、无自碰撞；加载网络/std/归一化，重置优化器和训练计数，不继承教师 BC |
| `--init-from ... --inherit-training-stage` | 加载网络/std/归一化并继承轮数与课程阶段；优化器仍重置，使用当前奖励配置 |
| `play --checkpoint` | 均值动作推理；保留 checkpoint 旁的 config.json |
| `--allow-recipe-mismatch` | 仅用于明确接受差异的诊断回放；不修改历史元数据，不放宽严格续训 |

resume 不恢复完整仿真/RNG 状态；迁移也不保证保留原技能。
历史策略的运行时缓存随新环境 reset 后重建，不随 PPO checkpoint 恢复旧 episode。
两种网络均由 `create_runner()` 统一归一化 learn() 的首帧，包含纯平地续训；不要在 train 中重复包装。
配方指纹含环境、地形及适配源码，相关文件的注释改动也可能改变指纹。
文档不再维护易过时的“当前指纹”常量，应实时读取源码与运行元数据。
`model_0.pt` 已做过一次 PPO 更新；`model_N` 的编号不是完成更新次数。
`--inherit-training-stage` 从 checkpoint 内保存的 `iter + 1` 开始，首次 reset 前恢复课程控制步计数；
`--iterations` 仍表示追加轮数。要求新旧 `num_steps_per_env` 一致，避免课程步数在后续续训中漂移。
课程按当前快照在该步数求值，不复制旧权重或仿真状态；普通 `--init-from` 默认行为不变。

### 可选历史网络

训练使用 `--policy-variant standard|history`，默认 `standard` 保留原网络。
`history` 在当前观测之外缓存过去 10 个控制步的 Actor 前 48 维；不含当前步，按最近到最远排列。
每环境独立缓存，控制步只推进一次；重复读取不推进，局部 reset 仅清理选中环境并用首帧填满。
Actor 和 Critic 使用同一份历史数据，但各有独立的 `480 → 128 → ELU → 64` 编码器，参数与梯度不共享。
Velocity 的 Actor 为 `61+64 → 512 → 256 → 128 → 14`，Critic 为 `70+64 → 512 → 256 → 128 → 1`。
历史每帧复用当前 Actor 前 48 维的归一化统计量，不重复计入统计；Critic 当前观测保留自身统计量。
PPO 存储传输维度为 Actor 541 / Critic 550，编码器在策略内将历史压缩为 64 维。
`config.json` 保存网络选项和历史契约；resume、普通回放及独立软垫回放自动识别。
两种网络不相互转换；新历史网络从头训练，同类型 checkpoint 才能续训或 `--init-from`。
网络实现集中在 `policy.py`，缓存集中在 `observations.py`；未增加奖励、特权特征或循环网络。

## 4. 物理、观测、边界与数值保护

- 软垫是独立场景选择：`--terrain mixed_soft_pad` 可与注册任务组合，仍要求 SynReal。
  它替换任务默认的平地/Rough/Slope 地形，不将这几种几何叠加。CLI、工作台和独立软垫回放不再限定 Velocity。
  任务的碰撞模型、Actor/Critic 维度、出生姿态、指令、奖励和终止课程继续由任务定义；
  旧软垫躯干高度 <0.065 m 的附加终止仅保留在 Velocity 家族，不能截断起身/坐姿/翻滚任务。
  分地形速度误差只用于 Velocity/VelStand；BallKick 的球随机器人 XY 平移，并按实际垫面放置。
  `scenes --checkpoint` 自动选择任务和自碰撞模型，按保存的 Critic 维度加载 standard/history 策略；
  根据 checkpoint 轮次恢复出生课程，`R` 重采样任务姿态并重置垫子和历史，保持无 episode 自动复位。
  独立回放仅为 Velocity/VelStand 使用速度键，其他任务保留自身指令和阶段推进；HOME 展示仍使用初始姿态。
  StandUp 专项评估入口也接受软垫，站立高度相对变形垫面测量。
  2026-11-01 验证：13 个基础任务和 5 个 Rough 变体完成真实 SynReal 单世界软垫两步检查；
  15 个 Backlash 变体在求解器构造时遇到既有 `D6 (0 linear, 2 angular)` 不支持错误，不能宣称可运行。
  这是底层关节能力限制，未通过删关节或变更机器人模型绕过。短测不代表技能收敛或长期接触稳定性。
  证据与 `model_4750.pt` 普通/独立软垫回放检查见 `.validation/microduck_soft_pad_tasks_20261101/`。
- 14 个 HOME 关节偏移动作；规范顺序为左腿 0–4、头颈 5–8、右腿 9–13。
  每物理子步饱和 PD：KP=2、KD=0.053、力矩上限 0.96。轮式/backlash 被动 DOF 另行映射。
- Actor 为 61 维：角速度 3、重力 3、关节位置 14、速度 14、上次动作 14，
  twist 在 48:51，head 在 51:55，body 在 55:61。任务未用指令必须 zero padding。
  Critic 按任务变化：Velocity/VelStand 70、StandUp/GroundPick 68、BallKick 74；省略官方双脚力通道。
- Newton 四元数主要 xyzw、官方 facade 为 wxyz；COM 速度与 link-origin 速度要做角速度补偿。
  reset/push 的 FK、affine proxy 和驱动历史只能同步被编辑世界，不能恢复成全局同步。
- 碰撞代理为凸包，自碰撞默认关闭；有限刚度关节限位不是逐步硬裁剪，目标角也不等于实际关节角。
  自碰撞奖励此时无有效信号。几何接触/接触法向不能冒充求解后接触力。
- 固定软垫数为 `floor(N*p+0.5)`，仅末尾对应世界有软体；比例不是每次 episode 的抽样概率。
  变形保护分母使用实际软垫世界数。渲染用独立粒子偏移缓冲，不改物理粒子。
- **训练与回放边界不同：** `play.main()` 将 `pad_edge_timeout_margin=None`，允许跨垫边，
  但全部世界仍有中心半径 2.5 m 限制。当前 6 m 垫上，这个回放边界位于垫内；不是取消了所有位置复位。
  训练按垫边内缩 8 cm 提前截断 episode，让鸭子在接近垫边时复位；回放即使恢复 checkpoint 环境也会执行上述边界覆盖。
  两者保留倒地、episode 时限、变形保护；Velocity 回合通常 20 秒。
  独立 `scenes` 不做 episode 自动复位，`R` 手动重置机器人、垫子及策略历史。
- 训练启用探索、噪声、推扰与学习。回放使用均值动作，关闭逐帧噪声和自动推扰，保留偏差/延迟，
  默认关闭摩擦随机化；恢复历史环境或显式开关可能改变摩擦设置。头部固定指令也可能与训练采样不同。
  混合回放使用完整环境步并传递鼠标力；纯平地可走快速路径。相同代码不保证跨进程轨迹逐位一致。

`SoftPadGuard` 每控制步独立保存粒子、刚体、关节和驱动历史；不能将 solver 的另一缓冲当作未改动的上一帧。
每子步检查单四面体体积比 J≤0 或非有限状态；故障世界回滚并隔离，步末失败重置。
失败帧屏蔽普通奖励，`soft_pad_failure=-50` 乘 dt 后为 -1，done=true、time_out=false。
其中 `soft_pad_failure` 是 Velocity 配方的显式惩罚；其他任务保留各自奖励定义，仍屏蔽失败帧普通奖励并终止。
旧 J≤0.25 正体积压缩保护已按用户要求关闭；J 是局部剩余体积比，不是整垫厚度变化比例。

单步失败数 >max(4, 软垫数×5%)，或 100 控制步内至少 8 次且转移失败率 >0.1%，将停止训练。
`diagnostics/guard_stop.json` 记录停止原因/分子/分母，另有各类首次故障快照。
其他物理爆炸、非有限状态及容量溢出仍 fail-fast；不得放宽保护掩盖问题。
接触刚度从 50000 降到 5000、lambda 从 2000 提到 10000 曾改善短测；改变了接触/可压缩性，
不是闭源 FEM 根因修复或真实海绵标定，亦不保证长训永不触发保护。

## 5. Velocity 奖励与判读

权重以快照对应任务的 `rewards` 为准。本地步态扩展现用于 `velocity`、`velstand` 及其 Rough/Backlash 变体。

2026-11-02 新增指令步频塑形，实现在现有 `reward_functions.py` 的本地保留区，无独立奖励模块。
Velocity/VelStand 及 Rough/Backlash 共八任务启用 `gait_cadence=+0.5`、`gait_overdue=-0.1`。
步频单位为左右脚合计步数/秒：f=clamp(3+5×平移指令模长+|yaw 指令|,3,6)，
例如 0.2/0.4 m/s 分别为 4/5 步/秒；这是未经过长期训练验证的初始参数，不是已测得的最佳步频。
所有参数位于各奖励的 `params.cadence`，六个关联项必须使用一致参数；配置回归检查这一约定。
有效步要求至少 0.06 s 腾空、8 mm 净空、0.03 s 稳定接触；另一脚须在最近 0.08 s 内有过稳定支撑，
由 `support_window_s` 控制，不再要求确认落地的同一帧另一脚仍接触；允许无双支撑的换脚。
左右交替才更新间隔，同脚重复点地不推迟超时，同时落地不计有效步。
节奏正奖励为有界的每控制步奖励率，间隔误差 15% 内满分，超出后高斯衰减；
乘实际有符号推进与速度跟踪门控，不能仅靠原地踏步或反向运动获得步频奖励。
超过目标交替间隔 1.5 倍后，正奖励逐渐失效并产生有界超时成本，2.5 倍时达到最大成本。
平移/转向指令阈值分别为 0.01，零指令关闭；躯干相对地形高度低于 0.09 m 或倾角大于 35° 时关闭并清理状态。
指令任一分量相对上一步变化超过 0.01 时清理计时；先免罚/免奖 0.3 s，再用 0.3 s 渐入。
两个奖励共享一次控制步更新，局部 reset 仅清理对应环境；不新增 actor 观测或固定相位时钟。

八任务 `air_time` 改用同脚周期 2/f 的 25%～60% 时间窗，保留原权重 +3；
4 步/秒时仍为 0.125～0.3 s，5 步/秒时为 0.1～0.24 s，并增加上述直立门控。
落点时间尺度改为 clamp(0.72/f,0.12,0.24) s，4 步/秒时保持原 0.18 s；
腾空、净空、摆动峰值和落点使用一致的平移/转向激活条件，纯转向也能启用这些项。
净空/峰值的高度公式不变。其余站立约束、软垫相对脚滑和推进不足项仍仅在 Velocity 配置中启用，
不能据共用步频项推断 VelStand 已具备这些额外约束。
新增日志为 `Metrics/cadence_target_steps_per_s`、`cadence_measured_steps_per_s`、
`cadence_valid_fraction`、`cadence_overdue`；测得步频只统计有效且尚未完全过期的交替样本，应结合有效比例判读。
奖励/步态/配方/回放指令/任务变体定向测试 78 通过、1 GPU 测试跳过，包含 CPU MuJoCo 环境接线与 reset。
未启动训练或 SynReal 动力学评估，尚无高速步态改善和软垫适应性的行为验收。
新配方改变指纹，旧策略需显式迁移微调；运行中的训练不会热更新。

2026-11-04 修复交替落脚判定并新增逐脚前摆反馈：上述八任务共享支撑历史修复；
仅 `velocity` 新增 `foot_swing_progress=+0.5`，沿用该任务现有平地/软垫混合训练范围。
新增项的 `params.cadence` 也须与其余六项一致。其余任务不自动新增前摆项。
每脚从首次离地到首次触地记录位移，经上述有效交替落脚确认后才采样；触地后的滑动不计入，
短于稳定接触时间的接触抖动不重置离地起点。方向在离地时由平移指令和躯干 yaw 固定到世界系，
分别计算世界系脚位移和相对躯干位移在该方向的投影，取较小者并截去负值；
因此单纯垫子/全身平移、身体后退或原地垂直抬脚不能代替有效前摆。后退/横移按各自指令方向计算。
目标为 `clamp(advance_time_fraction × |v_xy_cmd| / f, min_advance, max_advance)`，
默认 `advance_time_fraction=0.6`、`min_advance=0.008 m`、`max_advance=0.05 m`；
0.2/0.3/0.4 m/s 直行分别约 3/4/4.8 cm。这是初始塑形尺度，未经长训调优。
每脚得分为位移/目标截到 [0,1]，两脚取较低得分作为有界奖励率，不强制左右精确等幅；
每份样本只保留 `freshness_cycles × 2/f` 秒，默认 1.5 个同脚周期，不迈步或重复同脚不能持续领取。
乘有符号实际平移推进比例 [0,1]，并沿用直立、指令变化清理及 0.3 s 免奖/0.3 s 渐入；
零平移及纯转向关闭新增项，纯转向仍由原有步频项覆盖。局部 reset 只清理选中环境。
日志 `Metrics/foot_swing_advance_left/right` 为新鲜已确认样本的有效位移均值（米），
`foot_swing_score_left/right` 为逐脚分数，`foot_swing_pair_valid_fraction` 为两脚样本均新鲜的比例；
分数是完成度，不代表已乘实际推进和渐入门控，需结合 `Task/foot_swing_progress` 判断最终反馈。
定向回归 112 通过、13 跳过，涵盖无双支撑换脚、过期支撑、单脚替代、垂直抬脚、被动平移、
反向位移、拖脚、触地后滑动、接触抖动、后退/横移/旋转坐标系、计时缓存、部分重置和真实 CPU 环境接线。
旧软垫接触序列 CPU 重放：保存物理的 0.3/0.4 m/s 样本均识别左/右 21/22 个落脚，
旧同帧门控在统计窗口内均为 0/0；重放缺少真实 root 高度，固定为合格值，不能视为完整动力学回放。
证据见 `.validation/microduck_gait_feedback_20261104/REPORT.md`。
未启动训练或新的 SynReal 动力学评估，不能声称单脚短摆或软垫速度已改善；
旧权重需用 `--init-from` 显式迁移到当前配方微调，不能用 `--resume` 伪装为同配方续训。

| 本地项 | 权重 | 要点 |
| --- | ---: | --- |
| vertical_body_velocity | -1.0 | 世界系躯干 link-origin Z 速度平方 |
| standing_foot_motion / standing_foot_contact | -1.5 / -0.5 | 零 twist 门控，约束脚速与双脚几何接触 |
| hip_roll_neutral / action_over_limit | -1.0 / -0.5 | 髋侧摆偏离 HOME；目标超限 0.3 rad 余量 |
| straight_foot_heading | -1.5 | 相对身体脚尖方向，5° 容差，转向/横移/大倾角门控 |
| foot_slip | -0.1 | 软垫使用脚接触点相对表面切向速度；硬地保留原定义 |
| commanded_foot_placement | 0.15 | 触地时以另一只支撑脚和速度指令计算期望落脚点；误差越小正奖励越大；无 gait clock |
| foot_swing_progress | 0.5 | 仅 Velocity；两脚分别完成有效前摆，较弱侧限制奖励，样本会过期 |
| foot_clearance / foot_swing_height | 2.0 / 0.25 | Velocity 净空目标从 2.5 cm 随指令速度提高，最高 5 cm；误差用有界正奖励表示 |
| soft_pad_failure | -50.0 | 数值保护故障的失败惩罚 |
| command_progress | -0.25 | 仅混合场景；沿指令方向的推进不足，两出生组均适用 |
| neck_action_rate | -0.05 | Velocity/VelStand 及 Rough/Backlash：额外约束四个头颈策略输出的相邻步变化 |
| head_angular_velocity | -0.02 | 上述八任务：头部 jaw_soft 刚体世界系 XY 角速度平方和，允许世界 Z 轴转向 |

2026-11-02 头部稳定性配方调整：上述八个任务的 head_pose_tracking 保留 weight=2.0、std=0.5，
增加 fine_std=0.1、fine_weight=0.3，以 70% 宽尺度和 30% 精细尺度混合跟踪头颈关节目标。
跟踪项和新增 neck_action_rate 均标记 local，导出上游配置时保留。复用现有奖励函数，不改变 PD 或观测。
相关配方、回放指令、history 测试 31 项通过；尚无新配方训练后的头部摆幅对照，不能认定晃动已经改善。
修改奖励只影响后续训练，旧 checkpoint 回放不会自动改变；配方指纹变化，需显式迁移初始化微调。

头部实际角速度惩罚定义在 reward_functions.py，读取 body_link_ang_vel_w，包含躯干传递的旋转，
不以头颈关节速度替代。躯干相对地形高度从 0.09 到 0.11 m 渐入，倾角从 20° 到 40° 渐出，
倒地/低姿态时关闭；head_pose 指令改变或 reset 后先免罚 0.3 s，再用 0.2 s 渐入。
局部 reset 只清理对应环境的等待计时；该项标记 local 以保留导出配置。
Metrics/head_angular_velocity_xy_rms 记录未门控的头部 XY 角速度 RMS（rad/s），
Metrics/head_stability_gate 记录平均门控系数，Task/head_angular_velocity 为非正加权项。
仅验证公式、实际模型头部绑定和 reset/奖励链路；未完成新配方训练后的行为改善验收。

推进项对零速/纯转向关闭；速度阈值 0.03 m/s，容差 0.02，倾角≤45°；换指令后免罚 0.3 s，再渐入 0.3 s。
脚滑按有效接触平方速度平均，不让相反方向抵消。新公式存在不等于步态已改善。
落脚点项只在 touchdown 且另一只脚仍支撑时计费；它使用当前支撑几何、速度误差和 yaw 指令，保持 61D actor 观测契约。
2026-11-04 按用户软垫拔脚需求，仅 `velocity` 的净空和摆动峰值目标同步提高5 mm：
`0.025 + 0.035 × command_speed`，封顶0.05 m；直行0.2 m/s时目标由2.7 cm提高到3.2 cm。
其他任务保留 `0.02 + 0.035 × command_speed`、上限0.045 m；Velocity混合训练的平地/软垫组共用提高后的目标。
脚高由 `MixedTerrain.heights` 相对变形垫面测量，负净空截为0；该奖励不直接限制支撑脚压陷或碰撞穿透。
本次仅调整四个高度数值，未进行微调训练或软垫行走行为验收；旧权重回放不会自动提高抬脚。
官方腾空/净空/脚滑有指令门控，零 twist 踏步不能直接归因于“刷腾空奖励”。

指令范围 vx±0.4、vy±0.3 m/s、yaw±1 rad/s，普通重采样 3–8 秒。
六方向均有低/中/高幅值采样；每次 reset 约 20% 进入连续启停组，站 2–3 秒、运动 3–4 秒、停止后同速再启动。
站立课程参数为 5/10/15/20/25/25%，控制步节点 0/12000/18000/24000/36000/48000。
课程控制步每 PPO 轮增加 24，不乘环境数；条件覆盖及启停组使该参数不等于总静止帧比例。
head_pose_bias 约 601 轮起生效并逐阶段增加；action_rate 也会加重。课程变化时不能直接比较总奖励高低。

评估至少分别记录平地/软垫：实际速度与误差、倒地次数、存活时间、倾角、起停响应、数值保护率。
`Task/*` 是未乘 dt 的加权奖励；episode reward 还受时长影响。
`Physics/fall_fraction` 是采样状态比例；`reset_count` 含 timeout，日志均值可能为小数，都不是 episode 摔倒率。
`recovery_success` 的奖励均值也不是恢复成功率。分地形统计按出生组，离垫后不能冒充软接触采样。

## 6. 最有价值的实验结论与未解问题

以下均是历史实验，具体条件不能套用于当前 6 m / 25 cm 网格训练。
证据在仓库根 `.validation/`，通常不跟踪 Git；转交机器时需另保留这些产物。

### 检查点筛选（2026-10-30）

旧混合运行：`logs/rsl_rl/microduck_synreal/2026-10-29_17-46-16_491657_velocity_inset_pad_finetune/`。
训练记录到 13603，最后权重 13600；1.2 m / 24×24×3 齐平垫、5/15。
奖励/速度误差约 3000 轮后进入波动平台，**没有通过软垫运动收敛验收**。

评估 100/300/600/1200/6000/9300/13600；100/300/13600 各两 seed，共 10 次 16 环境、12 秒测试。
固定保存的 1.2 m 环境、统一成熟课程，关闭噪声/推扰/摩擦随机化，保留训练边缘复位。
前进稳定性及平地保留优先时暂推荐旧目录 `model_300.pt`：
0.3 m/s 指令下，300 的平地/软垫 vx 均值约 0.177/0.116，13600 约 0.062/0.063。
300 两 seed 前进档位无摔倒，但转向共 2 次失败；100 更快但前进/停步仍摔倒。
没有穷举全部 137 份权重，不代表跨边缘或当前大垫最优，也不代表已学会软垫运动。
报告及曲线：`.validation/microduck_selection_20261030/REPORT.md`。

### 边缘、低头与后仰

- 1 m 齐平凹槽场景的 9000 轮回放中，记录到脚接触硬边角、向内法向及随后减速/后退。
  保持网格间距扩大到 2 m 后，没有周边地板接触仍会在垫内摔倒；两组各 5 软垫环境/13 秒均 3 次失败。
  所以真实边缘接触和策略内部失稳都存在，不能说“场景肯定没问题”或“只有空气墙”。
  证据：`.validation/microduck_edge_scene_20261030/REPORT.md`。
- 更早 1.2 m / model1000 测试中，头颈异常可发生在远离边缘、没有硬地板接触时。
  中性头部指令仍失败，不能只归因于回放抬头指令。
  证据：`.validation/microduck_edge_pitch_20261029/REPORT.md`。
- 3 m / 24 网格、base_z=0 的凸起垫上，model9000、0.3 m/s：原策略 13 秒内软垫 3 次失败、平地 0。
  仅将头颈动作替换为指令姿态后变成软垫 13、平地 2；**该干预未合入**。
  有身体先倾倒，也有头颈先偏离；不能将固定头部当作修复。
  证据：`.validation/microduck_head_control_20261030/REPORT.md`。
- 鼠标拖拽曾出现 hinge 轴偏离/锚点分离与小幅越限；没有证明单关节实际连续转满 360°。
  简单减小鼠标力未消除，下一步需检查真实角度、轴约束和连接误差。
  证据：`logs/microduck_soft_pad/diagnostics/joint_*.json`、`joint_audit.py`。

### 已修复但不能回退的基础问题

- 2026-11-02 落脚奖励批量广播：`commanded_foot_placement_reward` 的 yaw 指令必须为 `[N,1,1]`，
  才能与双脚位置 `[N,2,2]` 相乘。旧 `[N,1]` 在 N=5/256 时崩溃，在 N=2 时会错误混用环境的 yaw。
  回归覆盖 N=1/2/5/256、左右脚落地、不同 yaw 和零指令；奖励/配方/回放定向测试共 37 项通过。
  本次 `model_400.pt`、5 环境、100 帧无窗口回放尝试在 SynReal 交互登录处 EOF，未进入物理验证。
- 2026-11-02 软垫起身穿透：`model_400.pt` 在当前 6 m / 24×24×3 凸起垫上，
  俯卧/仰卧起身短测出现约 5–19 mm 的碰撞网格顶点穿透，垫面只下沉亚毫米量级，且保护计数为零。
  仅看四面体体积保护不能排除机器人穿透；原 SDF 接触可能只覆盖薄部件较浅的交叠位置。
  顶面接触补丁（现位于 `scenes.py` 的 `PadVertexContacts`），在每次完整碰撞生成后补充网格部件最低顶点与本世界变形顶面的重心坐标接触，
  使用原生软接触刚度/阻尼求解，不移动状态或添加硬盖板；平地组、垫外和底面以下不补充顶面支撑。
  普通 SDF、侧面/底面和解析形状接触仍走原路径。相同碰撞 generation 重查不会重复添加记录。
  `MixedTerrain.check_contacts()` 同时覆盖训练、普通回放与独立软垫场景；元数据记录
  `vertex_face_contacts=lowest_mesh_vertex_top_surface_v1`，配方指纹包含新模块。
  旧 checkpoint 可显式 `--allow-recipe-mismatch` 诊断回放；接触行为已改变，不能冒充旧配方严格续训。
  证据、固定动作对照、测试与限制见 `.validation/microduck_penetration_20261102/REPORT.md`。
- 局部 reset/push 曾改写其他世界；现已做选中世界 FK/proxy/drive-history 隔离。
  `tests/test_state_isolation.py` 有两后端 GPU 用例；保留此契约。
- 旧安装原生扩展曾让软垫世界影响平地世界运动。更换仓库二进制的历史对照中，纯平地/半垫场景的
  平地组 vx 从 0.1335/0.0801 变为 0.1837/0.1837 m/s，消除了所测严重跨世界减速。
  证据：`.validation/microduck_flat_mixed_comparison/`，含哈希、备份、固定动作与回放对照。
  不代表全部环境数敏感性或软垫步态已解决；复发时先核对实际加载库，别盲目加迭代次数。
- 数值发散历史中 implicit Euler 优于旧 Midpoint-Cayley 故障回放；保留 `stability.py` 保护。
  证据：`.validation/microduck-stability/REPORT.md`。
- `train` 无探索诊断与 `play` 首帧动作一致，之后小物理差异会放大；重复同一 play 也出现类似差异。
  尚未定位原生重复运行差异，不能据此单独指控训练入口。
  证据：`.validation/microduck_train_play_diagnostic/`。

后续优先：按当前运行实际配置检查曲线与每数百轮 checkpoint，固定指令分开评估平地和软垫。
若做长距离假设验证，应明确控制网格间距、材料、求解、出生、指令和 episode 边界；记录实际连续运动时间。
不要同时改奖励、材料、求解和头部控制后再声称找到单一根因。

## 7. StandUp 与官方适配边界

StandUp 已复用官方奖励和出生课程。独立官方 Python 构造配置后与本地快照逐项核对，20 个奖励均有实现；
Actor 61 / Critic 68。仍使用共享 PD，缺双脚力、部分随机化，自碰撞默认无有效信号。

`configs/rtx4090_synreal_standup.json` 选择 SynReal / standup / official 平地、15000 轮。
保留任务 PPO：rollout24、save250、adaptive LR 初值0.001、entropy0.01；不从 Velocity 传 resume/init-from。
回合 6 秒，倒地不立即终止；站立/坐姿/俯卧/仰卧出生概率依次为：
0 轮 40/40/20/0%，600 轮 25/30/35/10%，1500 轮 20/25/30/25%，2500 轮 15/20/30/35%。
仰卧保留 ±90° 扰动，不是任意姿态恢复保证。

历史验收：50 项相关回归；StandUp GPU 套件 13 项；64 环境×5 轮真实 PPO 完成。
该短训权重在坐姿/俯卧/仰卧各 4 环境评估均 0/4 起身，不能称为技能成功。
证据 `.validation/microduck_standup_port/`；入口验收未启动 15000 轮正式训练。

评估强制起点类别，关闭逐帧噪声/自动推扰，保留姿态随机化及传感器偏差/延迟。
判据为躯干高度≥0.10 m、倾角≤20°、双脚几何接触连续 0.5 秒；分别报告到达率、结束保持率和首次保持时间。
超时后的站姿复位不能计成功；几何判据不等于严格承重验证。

官方 VelStand 路线为 Velocity + StandUp 专家 → Velocity 热启动学生 → PPO + 双教师 BC，最终是一个学生策略。
本地已新增 `distillation.py`：SynReal 平地、无自碰撞、history 网络的 Velocity → VelStand 热启动与双教师 BC。
原来的不带教师参数的 VelStand 仍是随机初始化普通 PPO，不能视作完整融合流程。

### SynReal history 双教师入口（2026-11-01）

```bash
# First run a short GPU smoke test; this command starts actual training.
.venv/bin/python -m examples.microduck.train \
  --training-config examples/microduck/configs/rtx4090_synreal_velstand_history.json \
  --num-envs 64 --iterations 5
# After validating the teachers and smoke run, omit those overrides for the configured training budget.
```

配置中的 `training.velocity_teacher`、`training.standup_teacher` 可自由更换为兼容的 checkpoint 路径；
也可用 CLI `--velocity-teacher` / `--standup-teacher` 覆盖。目录输入在新运行启动时解析为编号最大的文件。
默认配置暂固定用户提供的 Velocity `model_25750.pt` 与 StandUp `model_5250.pt`，不是最佳权重推荐。
StandUp 继续训练后，只需替换 `standup_teacher`，再启动新的 VelStand 实验；无需改源码。
参数与 SHA256 会写入新运行 `config.json`。运行中不热换教师，不读取后续新生成的 checkpoint。

学生继承 Velocity 的 Actor/Critic、两套历史编码器、动作 std 与双归一化。
默认 `ppo.teacher_bc.warm_start_optimizer="inherit"`，与官方一样继承 PPO Adam 动量、步数及参数组设置，
包括 checkpoint 当时的学习率；同步 PPO 自适应调度器的 LR 标量，避免下一次更新覆盖恢复值。
有效初始学习率记入 `velstand_warm_start.effective_ppo_learning_rate`；新任务仍使用配置中的 PPO 调度策略。
设为 `"reset"` 可清空 PPO Adam，并使用新任务 `ppo.algorithm.learning_rate`。BC Adam、轮次及课程始终从零开始。
继承前检查参数布局、动量形状及有限性；不能与 `--resume` / `--init-from` 混用。
要求两位教师同为 SynReal history、当前观测契约、14 维 HOME 偏移 PD、同物理配置；支持不同 Critic 维度。
学生允许在保持控制周期不变时整数倍细分教师的物理步长，以及增加关节约束 primal 求解次数；
不允许粗化时间步、减少关节求解次数或修改其他求解参数。数值细化仍会改变仿真轨迹，不代表动力学完全相同。
`training.physics_substeps` / `--physics-substeps` 只细分每个控制步，history 仍按 50 Hz 更新。
当前 VelStand JSON 采用 8 子步（dt=0.0025），其他任务的 env.py 默认 4 子步保持原值；保存间隔改为 50 轮。
原因是 2026-11-01 的 1024 环境双教师训练在原 5 ms 设置下出现仿射代理变形、关节约束偏离与速度发散。
同批次固定动作纯物理重放可重现异常，2.5 ms 细分明显改善；完整训练验证状态见
`.validation/microduck_velstand_failure_20261101/REPORT.md`，不能用短测代替长期稳定性与技能验收。
该修复已通过 299 项回归（17 跳过）、实际双教师 64 环境×5 轮新训、2 轮严格续训和 100 帧回放。
从此前失败的第 100 轮诊断权重显式迁移至细分物理后，1024 环境连续完成第 101–160 轮，跨过半蹲出生课程，
未触发保护且仿射代理行列式保持正值。正式新运行的 PID、日志位置见该验证目录的 `repaired_training.json`；
它使用原定两位教师重新热启动，保留用户当前 60000 轮预算，不改写旧运行或教师元数据。
用户随后明确要求 512 环境持续训练：已等上述 1024 环境运行保存 `model_50.pt` 后切换为 512 环境严格续训，
追加 59949 轮以完成总计 60000 轮，保留优化器、双教师和课程进度；继续使用 2.5 ms×8 子步。
上述 512 环境运行后来在第 825 轮附近再次失稳；`repaired_training_512.log` 是已退出运行的日志。
最新进程与运行目录仍以 `repaired_training.json` 为准，新的恢复记录见下文。
512 是此次显式 CLI 设置，不修改其他任务共享的 env.py 环境数默认值；不得为检查而中断健康训练。
预训练技能和 VelStand 碰撞模型下的迁移效果仍需分别评估，参数兼容不代表教师已学会技能。

### 2026-11-01 15:43 VelStand 再次失稳与关节求解细化

2.5 ms×8 的此前修复只缓解了问题，不能称为已彻底解决。失败运行
`2026-11-01_14-31-11_574205_velstand_history_dual_teacher_512` 在 control_step=19809、substep=6，
环境 57 的角速度从 864.595 增至 1125.647 rad/s，触发原有 1000 rad/s 保护。
前一帧峰值在左踝，失败帧在左髋；本次不能沿用旧头颈失稳结论。PPO/BC 与最后 `model_800.pt` 均有限。
原设置恢复诊断在落地时重现严重角速度增长，且发生在同一个策略动作持有期间；
证据指向冲击下约束求解/积分失稳，尚未定位到具体原生 C++/CUDA 缺陷。

本次只把 `joint_constraint_primal_iterations` 从 1 增到 4；2.5 ms×8、控制/动作/历史 50 Hz、
非线性 5 / 线性 35、PD、奖励及双教师保持原配置。VelStand JSON 已加入此参数，CLI 可显式覆盖。
固定动作对照显示角速度峰值下降，但未恢复全部原生求解缓存，不能声称精确重放了原异常。
正式恢复使用原运行 `model_800.pt`，保留 PPO/BC 优化器、学生归一化、冻结教师与课程，从 801 继续；
新目录记录 `solver_transfer_from`、源哈希与前后物理参数，禁止改写旧配置伪装严格同配方续训。
512 环境、总计 60000 轮；测试中人为跳到最终课程的权重不能用于正式恢复。
复现、压力测试、恢复脚本与状态位于 `.validation/microduck_velstand_recurrence_20261101/`，
详细验收以该目录 `REPORT.md` 为准。已通过 301 项回归（17 跳过）、512 环境原课程 80 轮、
最终课程 40 轮、2 轮迁移短训及 1 轮标准严格续训；原课程测试最高角速度分量 216.087 rad/s，
最小仿射代理行列式 0.587801，均未触发保护。正式恢复启动 PID=3171780，任何后续操作须重查状态。
后台脚本保留异常前 128 个物理子步轨迹，保护阈值不放宽，
不静默复位失稳环境，也不无条件循环重启。新日志使用该目录 `training.log`。
有限轮次测试不等于长期数值稳定或技能收敛验收；查看实时 `training_status.json` 后再报告进度。

每个控制步在学生归一化之前捕获原始当前帧与 10×48 历史，计算冻结教师标签。
两个教师各用自己的 Actor、历史编码器和冻结归一化；历史包含学生实际执行动作，不改写为教师动作。
倾角 <25° 由行走教师监督，>35° 由起身教师监督，中间区间交给 PPO。
起身教师当前 twist 槽清零；历史不含指令槽。站立且带身体指令的帧退出行走监督，
其中纯姿态倾斜交给起身教师，带高度指令的直立帧只走 PPO。
未照搬官方教师专用 yaw remap 或 >75° 关节屏蔽。

每轮 PPO 后进行 BC，混合并打乱两类有效样本，逐样本动作 MSE 乘教师系数后取批均值，与官方聚合方式对齐。
仅更新学生 Actor 和 Actor 历史编码器。默认 BC LR=3e-4、epochs=5、mini_batches=4、min_mini_batch=512，
批大小为 max(512, 有效样本数整除4)，余数单独成批；只对起身组设至少64样本门槛，行走组不设门槛。
两位教师系数默认均1；这些设置已显式写入训练 JSON 的 `ppo.teacher_bc`。
BC 缓存原始当前帧和历史，使用 PPO 更新结束时的最新学生统计量归一化，BC 期间不更新统计量。
日志保留 `Loss/bc_walk_fraction`、`Loss/bc_stand_fraction`、`Loss/teacher_bc`、`Loss/bc_updates`，
并增加每位教师的 `bc_{walk,stand}_{samples,selected,mse,updates}`：门控样本数、实际选中数、未加权 MSE、参与批数。
MSE 为本轮各次 BC 更新前误差的样本加权均值；没有有效样本时记0，必须结合 selected 看，不能当成完美拟合。
初期未采到倒地样本时起身 BC 可以为零；不能将 BC 下降当成恢复成功率。

BC 契约已升至 v2；旧 v1 学生仍可独立回放，但不允许静默按新训练语义续训。
学生 checkpoint 内嵌冻结教师状态和 BC 优化器；严格续训从该 checkpoint 恢复原教师，
不要求原教师文件仍存在，也不接受通过改 JSON 偷换教师。更换教师继续训练已有学生尚无单独入口。
回放只构造学生，不加载教师；细分物理步长的新权重回放需加 `--use-checkpoint-environment`。
严格续训用 `train --solver synreal --resume <VelStand/model_N.pt> --use-checkpoint-environment`，
不要再传带教师初始化参数的训练 JSON。物理细分是显式环境变化，不能伪造旧运行元数据按原配方续训。

全目录 CPU/非 opt-in 回归 245 通过、16 跳过；真实教师完成单世界 MuJoCo CPU 两轮诊断，
冻结教师不变、BC 更新与学生独立回放通过。验证范围及复现脚本见 `.validation/microduck_velstand_bc_20261101/`。
本轮为代码接入与 CPU 验证，未启动正式融合训练；SynReal GPU 双教师短训与行为收敛需另行验收。
2026-11-01 官方对照复核：快照八个配置部分与仓库内官方构造结果一致，20 项定向测试通过；
此后 v2 优化已对齐 BC 聚合、默认 warm-start 优化器继承及 BC 学生归一化时机；物理/奖励缺项仍存在。
初次对照见 `.validation/microduck_velstand_bc_20261101/OFFICIAL_AUDIT.md`，
v2 全目录回归 249 通过、16 跳过；真实教师两轮 CPU 诊断通过，Adam 继承后的初始 LR=7.59375e-5。
v2 验证见 `.validation/microduck_velstand_bc_v2_20261101/REPORT.md`。

本地注册 33 任务仅表示代码入口覆盖，不表示 33 个技能均收敛：
基础 13 项 velocity、velstand、standup、sitstand、groundpick、ballkick、rollers、swizzle、
rollercrouch、rollerslope、rollerstandup、spin、roulade；另有 5 Rough、10 Backlash、5 Rough+Backlash，
精确名称见 `task_registry.py`，不虚构组合。

官方复现仍缺 BAM、部分动力学随机化和力/冲击奖励；本地可选双教师 BC 的输入映射与优化契约见上文，
不包含全部官方教师专用修正。地形为 seed42 固定样本。
以运行元数据 `official_alignment` 的 supported/omitted/approximate 字段判断，不以快照中有名称判断已有效执行。
`deployment_compatible=False`；本地 PD checkpoint 不能宣称直接真机部署。

## 8. 常用命令与验证

命令从仓库根执行；启动前核对已有进程与 GPU 负载。模型旁保留 `config.json`。

```bash
# Replay a specific checkpoint with its recorded environment.
.venv/bin/python -m examples.microduck.play --solver synreal \
  --checkpoint logs/rsl_rl/microduck_synreal/2026-10-29_17-46-16_491657_velocity_inset_pad_finetune/model_300.pt \
  --use-checkpoint-environment --allow-recipe-mismatch \
  --num-envs 10 --pad-probability 0.5 --viewer gl --command 0.3 0 0 --num-frames 30000

# Train the separate StandUp expert; do not transfer a Velocity checkpoint.
.venv/bin/python -u -m examples.microduck.train \
  --training-config examples/microduck/configs/rtx4090_synreal_standup.json \
  --num-envs 1024 --iterations 15000

# Replace the placeholder with an actual StandUp checkpoint.
.venv/bin/python -m examples.microduck.evaluate_standup --solver synreal \
  --checkpoint "<standup_run>/model_14999.pt" --num-envs 8 --viewer null \
  --episodes 3 --output logs/standup_evaluation.json

# Optional local workbench; credentials are entered privately in the terminal.
.venv/bin/python -m examples.microduck.ui --login

# Use a free port to inspect the new run without replacing the existing old-run dashboard.
.venv/bin/tensorboard \
  --logdir logs/rsl_rl/microduck_synreal/2026-10-30_14-46-36_838789_velocity_inset_pad_finetune \
  --host 127.0.0.1 --port 6007
```

回放 I/K 前后、J/L 横移、U/O 转向、X 零速；用 `--pad-probability 0|1` 分开看两种地形。
上面的回放诊断允许配方差异，不是旧训练代码的严格复现；恢复环境后仍执行回放专用边界。

训练入口对照可显式使用 `--init-from ... --diagnostic-no-exploration --num-frames ... --command 0.3 0 0`：
均值动作、冻结权重和归一化、不运行 PPO、不保存模型，关闭观测噪声/interval events，仍直接调用训练 env.step。
摩擦仍取训练设置；对齐默认 play 需另加 `--no-randomize-foot-friction`。
普通训练不要带此开关，`--command` 也不控制普通 PPO 指令采样。
带 viewer 的训练命令更新路径与无窗口训练仍有差异，不能笼统说开窗口仅影响渲染。

工作台 `http://127.0.0.1:8765` 支持独立多实验/预览，关闭预览不停止训练。
关闭网页不停止子进程，但控制台终端 Ctrl+C 会停止其拥有的子进程；不接管外部终端训练。
预览加载最新完整保存权重，不自动热更新。多实验共享 GPU；不自动排队或融合策略。

按代码影响选择回归，不为纯文档修改跑全套 GPU：

```bash
WARP_CACHE_PATH=/tmp/microduck-warp-cache .venv/bin/python -m pytest -q examples/microduck/tests
# Opt-in GPU tests require available CUDA and normal SynReal authentication.
MICRODUCK_GPU_TESTS=1 WARP_CACHE_PATH=/tmp/microduck-warp-cache \
  .venv/bin/python -m pytest -q examples/microduck/tests/test_state_isolation.py
```

关键测试：`test_checkpoint_transfer`、`test_shared_physics`、`test_fixed_pad_groups`、`test_inset_pad`、
`test_mixed_terrain`、`test_soft_pad_guard`、`test_locomotion_rewards`、`test_training_diagnostic`、`test_standup`。
回放边界改动历史定向 22 项通过；StandUp 验收范围见上文，旧完整测试计数不作为当前全套验证结果。
报告须区分静态/CPU 测试、真实 SynReal 物理短测、PPO 链路与长期行为收敛。

历史网络严格复核（2026-10-30）：开启两项 GPU 测试开关后，全目录 **247 项通过，无失败或跳过**。
新旧网络各完成真实 SynReal 4 环境混合场景新训 5 轮、续训 2 轮、同类型初始化 1 轮，
以及纯平地新训 2 轮、续训 1 轮；另验证旧 `model_13500.pt` 初始化与回放。
16 次训练/回放入口检查累计 2208 条训练转移，1311 个 TensorBoard 标量均有限；
网络、梯度、Adam 状态、归一化、课程计数、22 个实际计算的奖励项及奖励加权汇总均通过检查。
新旧独立软垫回放各 40 帧并手动 reset，历史按首帧重建，权重/归一化保持冻结。
CPU 同状态同动作对照在三个课程阶段、自动 reset 和推扰下，当前观测、逐项奖励、指令及终止结果完全相同。

复核修复两处运行问题：标准网络纯平地续训首帧漏归一化；软垫自检在 `base_z=0` 为整数时浮点比较报错。
原 9 项几何测试失败来自依赖可编辑的全局实验几何；埋垫测试现显式构造 1.2 m 齐平垫，
场景材料测试分别覆盖埋垫/凸起垫。实际环境尺寸、材料、奖励、PD、课程和采样均未调整。
本次真实混合短测使用当前 6 m、24×24×3、`base_z=0` 凸起垫，不是第 2 节历史运行的埋垫。
证据与复现脚本：`.validation/microduck_history_audit_20261030/REPORT.md`。
以上证明训练/恢复/回放链路可运行，不代表短训网络已经收敛或掌握软垫运动。

## 9. 来源与文档维护

- [官方训练项目](microduck_rl-develop/README.md)、[官方目录规则](microduck_rl-develop/AGENTS.md)。
- [VelStand 流程](microduck_rl-develop/docs/velstand_policy.md)、
  [双教师实现参考](microduck_rl-develop/src/mjlab_microduck/tasks/distill.py)。
- [机器人运行时](microduck-main/README.md)、[部署契约](microduck-main/docs/policy-manifest.md)。
- 具体奖励查 `official_snapshot.json`、本地实现及
  `microduck_rl-develop/src/mjlab_microduck/tasks/`，不再复制数百条行号索引。

官方副本不代表最新上游，勿擅自更新。优先修改本地适配层，不为让来源测试通过而同步改写官方源。
`export_recipe.py` 会修改生成源码/快照和指纹；保留已有 `local:true` 奖励/课程/commands，
不保证保留所有其他手工修改，纯排查/文档整理不要顺手导出。

后续直接更新对应章节，不再追加互相覆盖的“最新摘要”。进程状态注明时间，实验结论保留配置、证据路径和限制。
此前文档精简仅更新交接正文与维护入口；历史网络改动及验证范围见第 3、8 节。
