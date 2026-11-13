# MicroDuck 模拟电流与实机输入

当前新启动的 cloth/hammock/elastic_bridge Velocity 使用 `cloth_history` v3。
Actor 74维、Critic 83维、20×61历史及主干结构不变；腿部10维从有符号负载代理改为电流绝对值，单位A。
平地和其他任务的观测不变。

## 官方实现依据

- 官方机器人使用 Dynamixel XL330；`microduck-main/duck-control/src/bus.rs` 使用 `Xl330Controller`。
  从124开始读取的连续寄存器块包含126地址的 `Present Current`，按有符号16位数读取后取绝对值，
  存入 `Sensors.currents_ma`。单位1 mA，不是部分其他Dynamixel型号的2.69 mA。
- `microduck-main/duck-control/src/pickup.rs::features` 把该读数除1000，得到绝对电流A。
  注意这是pickup特征接口；官方普通行走策略的61维观测本身不含电流。
- 官方训练 `microduck_rl-develop/src/mjlab_microduck/robot/microduck_constants.py` 选择 BAM 的xl330/m6。
  `microduck_rl-develop/scripts/pickup_datagen.py` 使用 `actuator_force / kt` 生成电流，
  `pickup/features.py::CurrentSensorDR` 模拟增益、偏置、噪声、取绝对值和1 mA量化。
- 本地可复现参数来源为
  [BAM xl330/m6.json](https://github.com/Rhoban/bam/blob/62bd8ce12154340be97e06f7f41a0ca8f116d967/bam/params/xl330/m6.json)，
  `kt=0.36601349688984386 N·m/A`。它已折算到输出关节，不再乘一次减速比。
  官方仓库明确到XL330系列；BAM README将其标为XL330-288-T，不据此推断用户电机铭牌的完整后缀。

## 当前仿真实现及边界

`cloth_current.py` 使用当前位置PD驱动器给出的饱和力矩，生成准静态等效电流：

```text
I_A = round(abs(tau_PD / kt) / 0.001) * 0.001
policy_input = I_A / 1.0 A
```

每个控制周期使用最后一个物理子步的读数，避免正负力矩平均抵消，匹配总线按控制周期采样的接口形式。
当前帧Actor 61:71、Critic 70:80、每帧历史48:58均为该电流；不进行第二次经验归一化。
reset清零反馈和本环境历史，首个完整控制周期后有效；历史不包含当前帧。

这是当前PD动力学下的电流近似，输入并非BAM内部的电磁力矩，也不是实测电流。
本次未更换执行器、未加入完整电压/反电动势/减速器摩擦模型；未假定实机电流上限。
现有0.96 N·m驱动力矩上限对应约2.623 A等效电流，不代表实机允许该电流。
未照搬pickup的随机参数作为已标定数据；当前不加电流噪声或额外滤波延迟。
因此本实现对齐了单位、关节顺序、绝对值和量化语义，尚未证明模拟读数分布与具体实机一致。
进一步提高精度需在相同动作下对照实机日志，标定力矩映射、摩擦、限流和传感器时序。

## 实机输入与checkpoint

`hardware_leg_current(currents_ma)` 接收官方15关节顺序的总线数组，提取索引
`[0,1,2,3,4,10,11,12,13,14]`，排除头颈和索引9的嘴部，取绝对值并除1000。
输出顺序是左腿hip_yaw/hip_roll/hip_pitch/knee/ankle，然后右腿对应五项。
实机只需每个50 Hz控制步写入该10维读数，并沿用同一20帧历史时序；不要再次除以0.96。
该Python转换函数提供明确的接线接口，本次未修改官方Rust部署器或导出ONNX。
官方61维策略运行时仍需扩展74维当前观测、20帧历史及加速度处理后，才能运行本地cloth_history策略。

v1/v2 checkpoint继续使用原有符号负载代理回放/续训；v3契约记录电机参数来源、单位和采样方式。
旧平地standard/history可正常迁移，新输入连接由既有迁移逻辑初始化。
旧布料v1/v2用init-from迁移时，将当前帧及历史中旧10维负载的输入连接清零，其余网络与归一化保留，
优化器重建。因为绝对值不能恢复力矩符号，不能声称这种迁移保持旧动作函数完全不变；需要重新微调。
v3到v3保留所有权重；新旧信号不能仅凭相同张量维数互换。

## 验证范围

CPU测试覆盖电流单位与正负号、1 mA量化、15到10关节映射、最新子步采样、历史时序和局部reset，
还覆盖v1/v2/v3回放、反馈对称性、旧负载连接迁移及PPO更新。CPU控制回路借用布料观测配置，
不验证SynReal柔性桥动力学或实机部署；未启动长期训练。
电流、布料网络、IMU、checkpoint迁移、旧历史网络及桥配置隔离定向回归共69项通过；定向Ruff通过。
