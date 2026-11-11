# MicroDuck 弹性带桥

新增地形 `elastic_bridge`：沿 X 方向长 1.2 m、宽 0.29 m，两端高度 0.65 m，初始中心下垂 4 cm。
两条短边各整排固定，两条长边自由；两端有横梁和支柱，桥下只有 z=0 地面，没有隐藏的硬桥板。
入口新增长0.8 m、宽0.8 m、顶面z=0.65 m的刚性平地平台，与入口横梁和桥面连续衔接。
鸭子在弹力带中央（x=0、y=0）朝+X出生；训练、回放和R重置按预静置后的实际桥面高度放置。
入口平台保留；静态预览也在中央出生，但使用未松弛的参考桥面高度。
使用 SynReal 原生弹性膜、完整三角形面接触；不是通过外部力托住鸭子。

尺寸和材料集中在 `scenes.py` 的 `SceneConfig.elastic_bridge_*`：40×10 网格单元、451 顶点、800 三角形，
面密度 0.6、拉伸刚度 1600、弯曲刚度 0.0002、材料阻尼 0.1。它是弹性带的膜模型近似，尚未做真实橡胶标定。
当前几何的固定端间跨度为 1.2 m；4 cm 是初始下垂，不是负载后的下垂限制。
入口平台尺寸参数为 `elastic_bridge_approach_length`、`elastic_bridge_approach_width`；
`elastic_bridge_spawn_distance` 保留兼容旧配置，当前中央出生不使用该距离。

启动和重置现在使用空桥在重力下预静置后的形状。`elastic_bridge_sag` 是材料参考网格的初始下垂，
不是仿真开始时的平衡下垂；实际平衡下垂随尺寸和材料参数变化，启动时重新求解。材料参考形状不会被覆盖。
启动前只对独立空桥做数值松弛并检查释放后的漂移，同一进程同配置复用结果；这会增加一次启动等待。
每个世界的空桥随后休眠，位置保持在平衡形状、速度为0；机器人进入桥面接触范围（含接触生成余量）
或桥面受到手动外力/速度扰动时立即唤醒。唤醒后保持正常弹性动力学，直到下一次重置。
运行时不增加材料阻尼、不改刚度、不施加托举力。静态 `preview` 不运行求解器，仍显示参考网格。

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
初始帧和重置时桥面速度为0；鸭脚进入接触范围后立即恢复正常受力下沉和回弹，不锁死承重桥面。

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

初始静止验证：`.validation/microduck_bridge_settle_20261110/`。真实双环境中鸭子位于入口平台，
2秒桥面最大位移为0；选择性重置恢复平衡形状，外部速度扰动后桥面仍能运动。
