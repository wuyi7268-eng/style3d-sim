# MicroDuck 平地跑步

当前按用户要求，`configs/running.json` 的 `spec.running_randomization_enabled=false`：
质量/惯量、COM、armature 随机化保持关闭；`spec.running_disturbances_enabled=true` 单独开启推扰与初始倾斜。
推扰与倾斜仍按下文课程在第11749轮开始逐步增加，普通回放关闭。
原有脚摩擦、编码器及 IMU 随机化不变。此设置仅对新启动进程生效。

独立任务 `running`，来自 [HannesVonEssen/microduck-running](https://huggingface.co/HannesVonEssen/microduck-running)
对应的 [microduck-playground@828d950](https://github.com/Vottivott/microduck-playground/tree/828d950134e29a8d04cbb51720a22c8729047fb7)。
源码文件 SHA256 和模型 revision 保存在 [configs/running.json](configs/running.json)。

复用共享环境、robot_walk、PD、观测和 PPO；新增公式集中在 [running.py](running.py)。
奖励、课程、指令、观测配置及 PPO 固化为独立快照，不运行时继承本地 Velocity 或弹力桥配方。
仅支持平地，既有任务配置不变。工作台、普通 train/play 和本地 checkpoint 续训都识别 `running`。

发布版 ONNX 已下载到 `logs/pretrained/microduck-running/policy.onnx`，SHA256 与发布 manifest 一致。
直接在 SynReal 回放发布权重：

```bash
.venv/bin/python -m examples.microduck.play_running_onnx --command 1 0 0
# 可用 --command 2.2 0 0 对照发布版指令；--viewer null --num-frames 150 用于无窗口短测。
```

该独立入口默认 SynReal、GL、单环境、50 Hz；I/K 调前进，J/L 调横向，U/O 调转向，X 零指令，R 重置。
`--onnx <path>` 可指定同契约模型。直接使用ONNX内置归一化，头/身体指令为零，不二次归一化，动作不裁剪。
动力学仍是本地PD而非BAM，所以这里只验证跨仿真器推理执行，不能保证复现发布版跑步表现。
2026-11-12 实测发布权重、vx=1 m/s、150帧：推理完成且0自动重置，但发生翻倒
（minimum_upright_cos=-0.94，平均机身前向速度约-0.015 m/s），尚未成功在SynReal复现跑步。
记录见 `.validation/microduck_running_20261112/official_onnx_report.json`。

```bash
# 小规模链路检查
.venv/bin/python -m examples.microduck.train --solver synreal --task running \
  --num-envs 4 --iterations 5 --run-name running_smoke --viewer null

# 从头训练；速度学习后自动执行 100 + 150 + 200 轮鲁棒性微调
.venv/bin/python -m examples.microduck.train --solver synreal --task running \
  --num-envs 4096 --iterations 12199 --run-name running --viewer null

# 本地训练结果回放，R 手动重置；零指令请求站立
.venv/bin/python -m examples.microduck.play --solver synreal \
  --checkpoint <run>/model_N.pt --command 1 0 0 --num-frames 1500

# 同配方续训，iterations 为追加轮数
.venv/bin/python -m examples.microduck.train --solver synreal \
  --resume <run>/model_N.pt --use-checkpoint-environment --iterations 1000 --run-name running_resume
```

控制频率 50 Hz，14 个未裁剪 HOME 偏移动作，Actor 61 / Critic 70 维，默认普通 MLP。
回合 12 秒；3% 的环境采样精确零 twist，其余为正向速度，小范围侧向 ±0.02 m/s、yaw ±0.05 rad/s。
保留上游继承的 20% 纯前向桶（vx 至少 0.3 m/s）。训练时头/身体指令保留微小非零采样，回放置零。
前进速度课程从 0.20–0.45 m/s 逐步到 1.65–2.20 m/s；边界按控制步计，默认每轮 24 步。
Running 强制每轮 24 步，以保持速度和鲁棒性课程的轮数语义。
4096 环境对应上游每轮样本预算；减少环境数不会自动延长课程，不能视为相同训练量。

鲁棒性阶段由 [running_randomization.py](running_randomization.py) 自动切换，沿用同一策略、优化器和归一化统计，
在阶段边界重置回合。普通 checkpoint 续训按已完成轮数接续，不重新从速度探索开始。
这里轮号从 0 开始，区间左闭右开；配置在 `spec.robustness`：

| 阶段 | 轮号区间 | 轮数 | 水平速度推扰（各轴） | 躯干 / 头部 COM（各轴） | 初始 roll/pitch |
| --- | --- | --- | --- | --- | --- |
| 速度学习 | [0, 11749) | 11749 | 0 | ±3 / ±3 mm | 0° |
| 鲁棒性 1 | [11749, 11849) | 100 | ±0.03 m/s | ±3 / ±3 mm | ±1° |
| 鲁棒性 2 | [11849, 11999) | 150 | ±0.06 m/s | ±5 / ±5 mm | ±1.5° |
| 鲁棒性 3 | [11999, 12199) | 200 | ±0.10 m/s | ±8 / ±6 mm | ±2° |

推扰每 3–6 秒采样一次，给世界坐标水平速度加增量，不是单位为 N 的持续外力。
12199 以后继续训练会保持最终难度。官方记录父 checkpoint 为 11748、发布 checkpoint 为 12195，
与文字描述的 450 轮并不完全相加一致；本地明确从下一轮 11749 开始执行完整 450 次更新，
不宣称复刻官方历史 checkpoint 编号。倾斜重置按上游函数将 yaw 置零，不叠加随机 yaw。

另外已接入上游随机化：

- 启动时按环境随机化 `trunk_base` 的质量和惯量，共用比例 `exp(2α)`，
  `α ~ U(0.5 ln(0.95), 0.5 ln(1.05))`，范围为标称值的 95%–105%，不是直接均匀采样比例。
- 每次回合重置重新采样躯干/头部重心和 14 个舵机关节的 armature（转子等效惯量，±10%），
  始终基于标称值，避免逐次漂移；只修改本次重置的环境。
- 保留现有脚底摩擦系数 [0.7, 1.3]、编码器偏置、IMU 噪声及观测延迟。

头部 COM 沿用上游的 `neck`、`neck_pitch`、`yaw_roll_motion`、`(bottom_head_shell|jaw_soft)`、
`bearing_roll` 选择器；其中 `bearing_roll` 实际是髋部链接，这是上游命名选择的既有情况。
普通回放关闭新增随机化；训练的 `config.json` 会记录 `running_randomization`。
SynReal 的 COM 更新需要重建求解器缓存，频繁回合重置可能降低吞吐量；没有修改核心求解器。
已验证局部重置不会改动其他环境的公开状态数组，尚未证明缓存重建后其他环境的后续轨迹逐位一致。

主要奖励为 `5 × clamp(vx_body, 0, 2.4) / 2.4`，另有速度跟踪、出生朝向保持和侧滑/yaw 误差代价。
保留原版较弱的姿态/角动量/动作变化约束、15 mm 足部净空目标；air_time 权重为零。
动作变化代价从 -0.02 延迟增至 -0.10。未引入本地行走的步频、软垫或布料奖励，也不规定必须有腾空相。
与原版一致，前进奖励不按零指令关闭，因而零指令桶也不保证停稳；速度指令不是速度上限。

这是**训练任务移植**，不是发布权重的直接兼容层：HF checkpoint 来自另一版 RSL-RL 和 BAM，
不能直接交给本地 `--resume` / `--init-from`。本地新训练的 checkpoint 可正常保存、续训和回放。
本配方采用原版速度探索阶段，并使用发布版的 2.2 m/s 课程、2.4 m/s 奖励上限、750 轮高速阶段间隔；
自动执行上述最后 450 轮鲁棒性微调，但尚未长期训练验收。
共享适配仍使用 PD 而非 BAM，没有 BAM 电压、摩擦和执行器延迟随机化，Critic 省略脚力通道。
JSON 保留上游事件以说明来源，实际适配差异写入运行的 `official_alignment`，不可视为全部事件已实现。
原发布模型约 1.65 m/s 的速度不是本项目的验收结果，也不代表实机验证。

**脚力观测**是左右脚各自的世界坐标净接触力 `(Fx, Fy, Fz)`，共 6 维，
包含水平摩擦力和垂直支撑力；上游用 `sign(F) × log(1 + |F|)` 压缩数值范围。
它只提供给训练时的 Critic，帮助估计状态价值，Actor 部署时不需要这些传感器。
触地布尔值只能表示是否接触，不能表示承重大小或摩擦力；关节 PD 力矩也不能冒充脚力。
当前 SynReal 适配尚未取得经验证的真实脚部接触力输出，因此保留 Actor 61 / Critic 70 维，
而非上游的 Critic 76 维，没有用虚构数值补齐。

新增任务接线会改变共享源码指纹，既有 checkpoint 的严格校验仍会拒绝源码差异；
需要严格续训时使用其原源码，诊断回放可显式 `--allow-recipe-mismatch`。不改写旧 checkpoint 元数据。
以后仅修改 `running.py`、`running_randomization.py` 或 `configs/running.json` 不会改变其他任务的配方指纹。

2026-11-12 鲁棒性补齐验证：71 项定向回归通过、12 项跳过，覆盖阶段边界、随机化范围/不累积、
局部重置、逐环境 COM 速度换算、PPO 优化器保持和新进程式 checkpoint 恢复。
SynReal 2 环境压缩为速度阶段 + 三个鲁棒性阶段各 1 轮，完成 192 条转移，8 个 Actor 参数张量更新，
参数有限，求解器惯性缓存与模型一致。33 个既有任务及桥有效配方逐项未变。
证据与限制见 `.validation/microduck_running_robustness_20261112/REPORT.md`。

此前任务移植验证：97项相关回归经修复复测通过，覆盖CPU PPO更新/保存/恢复、课程边界、局部重置和任务隔离；
SynReal 2 环境 × 2 轮完成 96 条转移，8 个 Actor 参数张量更新，参数有限，19 项奖励全部适配；
独立 play 加载该 checkpoint 完成 30 帧，0 次自动重置。
证据位于 `.validation/microduck_running_20261112/`。尚未长期训练或验收跑步收敛；未停止现有用户训练。
