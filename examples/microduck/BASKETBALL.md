# MicroDuck 篮球平衡任务

鸭子从篮球顶端出生，保持平衡并响应前进、横移、转向指令；不包含爬上篮球或倒地恢复。
来源：[Hugging Face 模型](https://huggingface.co/HannesVonEssen/microduck-basketball)、
[作者源码](https://github.com/Vottivott/microduck-playground/tree/052a2494dd6372289221c7a4f24a245b443dd75f/experiments/basketball)。
移植的任务公式固定到 `052a2494dd6372289221c7a4f24a245b443dd75f`。

## 选择后端

统一入口 `python -m examples.microduck.basketball train|play|preview --solver synreal|mujoco`。
两个后端均支持本地 LSTM PPO 新训、严格续训、本地权重回放和上游 ONNX 策略诊断回放。
默认 SynReal、CUDA；MuJoCo 支持 CUDA 批量环境，CPU 仅支持 `--device cpu --num-envs 1`。
命令从项目根目录执行，使用根目录 `.venv`，SynReal 沿用项目已有登录方式。

```bash
# 无策略动态预览：自由球＋HOME关节PD，不加平衡辅助，可能掉下篮球。
.venv/bin/python -m examples.microduck.basketball preview --solver synreal
.venv/bin/python -m examples.microduck.basketball preview --solver mujoco

# 本地训练；建议先短测，再增大环境数和轮次。
.venv/bin/python -m examples.microduck.basketball train --solver synreal --num-envs 64 --iterations 5
.venv/bin/python -m examples.microduck.basketball train --solver mujoco --num-envs 64 --iterations 5

# SynReal 长训示例：从头训练，无窗口；30000轮是训练预算，不保证收敛。
.venv/bin/python -m examples.microduck.basketball train \
  --solver synreal --num-envs 64 --iterations 30000 --viewer null

# checkpoint旁必须保留config.json。替换 <run>/model_N.pt 为实际文件。
.venv/bin/python -m examples.microduck.basketball play --solver synreal --checkpoint <run>/model_N.pt
.venv/bin/python -m examples.microduck.basketball play --solver mujoco --checkpoint <run>/model_N.pt
.venv/bin/python -m examples.microduck.basketball train --solver synreal --checkpoint <run>/model_N.pt --iterations 1000
```

新训练默认8环境，日志在 `logs/rsl_rl/microduck_basketball/`。续训默认恢复保存的环境数，
要求相同求解器、环境数、源码；恢复网络、Adam、归一化、控制步计数和每个世界的辅助课程等级。
仿真状态、随机数状态、LSTM回合记忆不作为完整现场恢复，新回合重新开始。
本地 checkpoint 可跨求解器回放，用于物理迁移诊断；不保证行为一致。
源码改变后的诊断回放可显式加 `--allow-source-mismatch`，训练续训不允许。

默认训练同时学习原地平衡与小速度指令。若只训练原地平衡，可复制 `configs/basketball.json`，
将 `commands.standing_probability` 设为 `1.0`，新训时通过 `--config <自定义配置.json>` 指定。
`--command 0 0 0` 只控制回放，不覆盖训练时的随机指令；续训始终使用 checkpoint 保存的配置。

## 上游预训练策略

```bash
.venv/bin/python -m examples.microduck.basketball play --solver synreal --pretrained
.venv/bin/python -m examples.microduck.basketball play --solver mujoco --pretrained

# 网络不能访问HF时，先下载模型仓库的policy.onnx，再指定本地文件。
.venv/bin/python -m examples.microduck.basketball play --solver mujoco --pretrained --policy-file /path/to/policy.onnx
```

首次自动下载到 `logs/pretrained/microduck-basketball/policy.onnx`，后续可以离线使用。
下载和用户指定文件均验证发布版 b11 的 SHA256：
`e105148b160b3a86170621648215be38dd5018e955930ce182394f4513c7569b`。
ONNX内置归一化；显式维护每个环境独立的 `h/c [1,1,256]`，每个控制步更新一次。
发布图固定batch=1，多环境推理逐环境执行，不将不同世界的记忆混用。
`--pretrained` 仅回放，不是把上游 RSL-RL 5/BAM checkpoint 当成本地 RSL-RL 2 checkpoint续训。

### 在 SynReal 微调上游 Actor

`train --init-from-onnx logs/pretrained/microduck-basketball/policy.onnx` 可以从校验过的 b11 ONNX
还原可训练的 LSTM/MLP Actor。ONNX 与 PyTorch 的 LSTM 门顺序不同，导入器显式转换，并测试连续推理及局部记忆重置一致性。
Actor 归一化保持冻结；本地 Critic、优化器、探索标准差和辅助课程重新初始化。这是跨动力学微调，不是上游完整训练状态续训。
训练后生成正常本地 `.pt`，后续用 `--checkpoint` 严格续训，继续冻结 Actor 归一化并保存初始化来源。

```bash
.venv/bin/python -m examples.microduck.basketball train --solver synreal \
  --init-from-onnx logs/pretrained/microduck-basketball/policy.onnx \
  --config .validation/microduck_basketball_finetune_20261111/config.json \
  --num-envs 64 --iterations 5 --viewer null
```

该运行配置使用固定学习率 `2e-5`、初始探索标准差 `0.1`，每100轮保存，其他任务配置沿用默认值。
辅助课程从最强等级开始逐步撤除；有辅助时站稳不能视为自由篮球平衡技能通过验收。

## 控制与观测

默认窗口回放1500帧（30秒）；`--viewer null --num-frames 150 --report output.json` 可无窗口记录。
`--command 0.05 0 0` 设置初始前进/横移/yaw指令，默认 `[0,0,0]` 为原地平衡。
I/K 前后、J/L 横移、U/O 转向、X 零速；键盘限制为训练范围 ±0.15/±0.10 m/s、±0.50 rad/s。
改变指令保留LSTM记忆。`R` 同时重置鸭子、球和LSTM，空格暂停；鼠标右键可拖拽施力。
跌倒、超时均不会自动重置或清空记忆，便于连续观察；报告记录失败标记。

Actor严格为61维：角速度3、重力3、相对HOME关节位置14、上一控制周期的关节速度14、上次动作14、速度指令3、零填充10。
Actor没有球位置/速度。Critic为64维，增加躯干速度和球的特权状态，不含上游足部力通道。
本地Actor与Critic分别使用LSTM256，再接512/256/128 ELU网络。
控制50 Hz，动作裁剪±1、相对HOME偏移1 rad，无动作低通滤波。

## 场景、课程与移植边界

配置统一在 [configs/basketball.json](configs/basketball.json)，不修改行走任务快照。
`basketball_scene.py` 使用完整机器人碰撞模型和自由球关节；半径0.12 m、质量0.62 kg，
球缝只用于显示，不增加碰撞或惯量。机器人根部初始高度为 `2R+0.125+0.003 m`。

训练采用上游辅助课程 `[1,0.5,0.25,0.1,0.03,0]`：球的水平弹簧/阻尼及鸭子稳姿/回中力随等级减弱，
存活至少6秒升级，少于1.5秒降级，各世界独立。辅助等级和自由球比例写入TensorBoard，
有辅助时的奖励不能视作自由篮球平衡能力。回放、预览始终将辅助关闭。

保留上游篮球的居中、球面双脚位置、相对高度、直立、速度跟踪与球速度代价，以及本地动作/力矩/限位正则。
失败标准为相对球心高度低于 `R+0.05 m`、水平偏离超过0.16 m或倾角超过55°。

这是两个本地物理后端上的任务移植，不是原版MuJoCo/BAM完全复现：

- 两个后端均复用本地饱和PD（KP2、KD0.053、力矩上限0.96），没有移植BAM电机和电气随机化。
- 球使用实体刚性球，不模拟气压/球壳变形；摩擦优先级和4维接触传给MuJoCo，
  SynReal仍使用自己的接触/摩擦实现，两者不能认为动力学相同。
- 本地包含出生扰动、观测噪声、速度指令和一控制步关节速度延迟；
  未包含完整上游IMU偏差/延迟、推扰、电机随机化及全部共享行走正则项。
- 辅助力按物理子步更新；上游按控制步计算。Critic使用本地独立LSTM，结构与上游Critic不同。

上游97.01%的60秒存活率不代表本项目的验证结果。本地短训只检验链路，不代表学会自由球平衡。
软件来源许可和声明见 [assets/basketball/LICENSE](assets/basketball/LICENSE)、
[NOTICE](assets/basketball/NOTICE)；本地修改包括物理适配、配置、训练和回放入口。

## 验证记录

2026-11-10：当前工作区已存在完整篮球任务，本次复用并复核，没有新增重复任务或改变任务动力学。
31项篮球/共享物理测试通过（包含SynReal与MuJoCo的CUDA局部重置），Ruff按128列通过。
当前源码真实SynReal完成2环境×2轮PPO及1轮严格续训，共144条转移；checkpoint与69个标量样本有限，
续训控制步计数正确且策略权重更新。零指令、无辅助回放150帧，0自动重置，117帧触发失败判据。
这是训练/续训/回放链路验证，短训尚未学会平衡；本次未启动长期篮球训练，也未停止已有蹦床训练。
命令与结果见 `.validation/microduck_basketball_recheck_20261110/REPORT.md`。

2026-11-09：篮球与共享物理共31项检查通过，包含CPU观测、奖励、终止、课程、ONNX记忆管理、
回放重置契约及两个后端的CUDA分世界重置。Ruff和Python编译检查通过，GL场景截图已核验。
真实SynReal和MuJoCo各完成2环境×2轮PPO、追加1轮严格续训及150帧回放；
累计288条训练转移、138个有限标量，回放均0自动重置。自由球回放分别116/126帧处于失败状态，
明确未获得平衡技能。完整训练/续训/回放记录在
`.validation/microduck_basketball_20261109/`，最终结果见其中 `pipeline_summary.json`。
未启动长期训练，未做自由球技能收敛验收。

同日按用户要求回放上游b11：通过机器现有HTTP代理下载到2500571字节的ONNX，SHA256与发布值一致，
缓存位于上述 `logs/pretrained/` 路径。真实ONNX在本地MuJoCo/PD、seed42、零速度指令下完成300帧/6秒，
约0.76秒首次触发失败，263/300帧失败、0自动重置；不能认为策略已成功迁移到本地PD。
另启动的GL交互回放在138帧后正常结束，无异常堆栈；再次启动后可按R重置。数据与启动记录见
`.validation/microduck_basketball_pretrained_20261109/`。未做上游BAM环境复现或SynReal预训练策略回放验收。
