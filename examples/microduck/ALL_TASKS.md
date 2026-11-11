# MicroDuck 项目任务

项目场景及统一策略目标见 [维护交接中的项目任务清单](HANDOFF.md#1-接手原则与目标)。
本地已注册训练任务的精确名称以 [task_registry.py](task_registry.py) 为准；注册不表示技能已经收敛。

官方子项目文档未移动或合并。

独立 [篮球平衡任务](BASKETBALL.md) 使用 `basketball train|play|preview --solver synreal|mujoco`，
两个后端均支持本地盲观测LSTM训练、续训和回放。入口 `python -m examples.microduck.basketball`；
已下载并校验上游b11 ONNX，真实MuJoCo/本地PD回放会掉球，尚未通过技能迁移验收。不是官方33项注册任务。

独立 [SwingPump 秋千任务](SWING.md) 统一通过 `swing train|play|preview --solver synreal|mujoco`
支持两个后端训练、续训、本地权重和上游预训练策略回放；
入口 `python -m examples.microduck.swing`，不属于官方 33 项注册任务，尚未通过本地技能收敛验收。
