# 秋千任务：SynReal / MuJoCo

来源：[microduck-swing 模型卡](https://huggingface.co/HannesVonEssen/microduck-swing)，
[完整源码 c5fcc50](https://github.com/Vottivott/microduck-playground/tree/c5fcc50219fef01ac9931d0079c583ccbb29b689)。

统一入口 `swing.py`，通过 `--solver synreal|mujoco` 选择训练和回放后端，默认 SynReal、seed42。

用上游 MuJoCo/BAM 环境回放现成策略（显式 seed27，原生 MuJoCo 窗口）：

```bash
.venv/bin/python -m examples.microduck.swing play --pretrained --solver mujoco --seed 27
```

自动缓存固定 commit 源码到 `logs/pretrained/microduck-swing/`，复用官方子项目已有虚拟环境；
不覆盖官方源码或安装依赖。使用上游双绳 tendon、BAM、观测及归一化。MuJoCo 训练默认 CUDA，回放默认 CPU，均可显式 `--device`。
两个后端均不使用绳子自碰撞。
`R` 手动重置，空格暂停；回放清除自动终止条件。`--viewer null --num-frames 1800` 可做无窗口短测。

先看现成策略（首次运行自动下载固定版本，随后离线使用）：

```bash
.venv/bin/python -m examples.microduck.swing play --pretrained
```

加载上游 `alpha050.pt` 的 Actor 和冻结归一化，SHA256 为
`49fb3599be519cb604ed65e1ba495fb9f5746e5bee1d4cb95107dcd165f47f9c`。
缓存位于 `assets/swing/alpha050.pt`，权重不纳入 Git；下载源为模型卡指定的 GitHub commit。
这是上游策略在 SynReal 中的诊断回放，不是上游 MuJoCo/BAM 环境的复现。
`R` 重置全部环境，空格暂停；鼠标右键可施力。超时或几何越界不自动重置。
默认回放 1800 控制帧（36 秒仿真时间）；可用 `--num-frames` 延长，关闭窗口退出。

本地训练与权重回放：

```bash
# 无策略预览：仅保持坐姿 PD，不给秋千额外推动或平衡辅助。
.venv/bin/python -m examples.microduck.swing preview

# 选择后端训练；正式训练自行增加环境数和轮次。
.venv/bin/python -m examples.microduck.swing train --solver synreal --num-envs 8 --iterations 1000
.venv/bin/python -m examples.microduck.swing train --solver mujoco --num-envs 8 --iterations 1000

# 替换为实际路径；config.json 必须与 checkpoint 保存在同一目录。
.venv/bin/python -m examples.microduck.swing play --solver synreal --checkpoint <synreal_run>/model_N.pt
.venv/bin/python -m examples.microduck.swing play --solver mujoco --checkpoint <mujoco_run>/model_N.pt
# 两个后端均可续训，solver 必须与 checkpoint 一致。
.venv/bin/python -m examples.microduck.swing train --solver mujoco --checkpoint <mujoco_run>/model_N.pt --iterations 1000
```

训练日志默认位于 `logs/rsl_rl/microduck_swing/<时间>_<后端>/`。两个后端都支持 `preview`。
MuJoCo 训练使用 `--viewer null`；SynReal 训练可选 GL。回放两个后端都支持 GL/null 和 `--report path.json`，
共同记录摆角（rad）、双绳长度和横向偏移（m）；SynReal 另有对齐误差/失效标记，MuJoCo 另有角速度/松绳量/绳长差。
`--config` 只用于 SynReal 新运行，默认 `configs/swing.json`；checkpoint 自动恢复保存的配置。
MuJoCo 使用固定上游任务配置，训练保存 `config.json`、`params/env.yaml`、`params/agent.yaml`，
并记录、恢复上游 `MICRODUCK_SWING_*` 环境变量。
续训恢复保存配置、优化器、归一化与课程步数，但不恢复仿真/RNG 状态；源码变化会拒绝严格续训。
仅诊断回放可用 `--allow-source-mismatch`。上游 `--pretrained` 目前只供回放，不能直接续训。

实现与边界：

- `swing.py` 是唯一训练/回放入口，先选择后端再导入对应依赖；MuJoCo 自动进入官方虚拟环境，复用上游环境/PPO。
  SynReal 继承共享环境的碰撞、饱和 PD、步进与分世界状态同步，复用本地 PPO。
- `swing_scene.py` 负责上游座椅资产、惯量及双绳拉力；总质量约 0.895 kg，座椅刚性固定到躯干。
  两条绳分别作用在左右连接点，松弛时不推，绷紧后按 2000 N/m 拉伸。
  395 mm 处增加柔顺限长惩罚，具体参数见 `configs/swing.json`；不是刚性摆杆或运动学动画。
- `swing_task.py` 保留上游 17 项奖励、双向历史峰值、几何失效锁存和权重课程。
  24 秒训练回合，无初始推动；正奖励在持续几何违规后关闭，但回合继续，以防靠提前重置逃避惩罚。
- Actor 61 / Critic 68，50 Hz 控制；Actor 的 48:51 是 `[0, body_y_world.x, body_y_world.z]`，
  其余 10 个指令槽为零。动作裁剪到 ±1，再乘 0.7 rad。固定源码实际采用 `SWING_SEATED_FRAME`
  作为关节默认姿态/动作参考；没有改动行走任务的 HOME。
- SynReal 使用本地 PD，未移植 BAM、电池/摩擦随机化、IMU 安装误差/传感器延迟，以及上游额外 PPO 修正。
  绳力在每个 1 ms 子步施加；限长惩罚不同于 MuJoCo 的 tendon soft limit。
  座椅及支架按上游默认仅显示、不碰撞；不是带真实织物绑带接触的模型。

2026-11-09 验证：静止悬挂 5 秒，双绳 382.08–382.29 mm，无几何失效；
上游 alpha050 在 SynReal 的 6 秒诊断摆幅约 26.5°，出现松绳及几何失效（171/300 帧已锁存），
没有数值异常或自动重置。这不是稳定荡秋千验收，不能引用上游 173.20° 作为本地效果。
真实 PPO 短训、严格续训、回放与回归记录见 `.validation/microduck_swing_20261109/`。
本次没有启动长期训练。

统一入口合并后复核：两个后端各完成 2 环境×2 轮 PPO、追加 1 轮严格续训，
以及续训权重/alpha050/无策略预览各 30 帧无窗口回放；权重与训练标量有限，回放无自动重置。
定向回归 56 项通过、4 项跳过，代码检查通过。证据：`.validation/microduck_swing_unified_20261109/`。
本次未重新验收 GL 交互；合并入口不代表已解决 SynReal 转圈或跨后端动力学差异。
