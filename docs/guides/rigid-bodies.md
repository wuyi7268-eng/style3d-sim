# 刚体、碰撞与关节

Style3D Sim 用仿射粒子代理求解刚体，再重建 Newton 的刚体姿态和速度。
用户创建场景时仍以 Newton body、shape 和 joint 为单位，无需自己构造代理矩阵。

## 创建一个落地箱子

用下面的模型创建部分替换 [第一个仿真](../getting-started/first-simulation.md) 中的 builder 段，
后续依然使用同样的 state、control、pipeline 和推进循环：

```python
import newton
import warp as wp
import style3d_sim as s3d

builder = s3d.ModelBuilder(up_axis=newton.Axis.Z)
builder.add_ground_plane()
body = builder.add_body(
    xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
    label="box",
)
builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
model = builder.finalize(device="cuda:0")
solver = s3d.NewtonSolver(model)
```

`hx/hy/hz` 是半尺寸，因此箱子的边长为 0.2 m。
shape 提供碰撞几何，通常也参与质量与惯量的生成；body 只提供刚体身份与姿态。
单独 `add_body()` 没有可碰撞形状，不会自动得到箱子接触。

对于自定义质量与惯量，参考现有示例的 `mass`、`inertia`、`com` 和 `lock_inertia` 用法。
不要同时随意覆盖 shape 密度与 body 惯量，再假设质量仍由原值决定。

## 接触检测与接触材料

```python
pipeline = newton.CollisionPipeline(model)
contacts = pipeline.contacts()
pipeline.collide(state, contacts)
solver.step(state, following, control, contacts, 1.0 / 240.0)
```

这个片段依赖已经创建的 state 与 control。每个子步更新接触可以降低快速运动下的漏检风险。

| 设置入口 | 含义 |
| --- | --- |
| `builder.default_shape_cfg.ke` | 后续创建 shape 的接触刚度 |
| `builder.default_shape_cfg.kd` | shape 接触阻尼 |
| `builder.default_shape_cfg.mu` | 摩擦系数 |
| `model.soft_contact_ke/kd/mu` | 粒子/布料与 shape 接触参数 |
| pipeline 的 contact capacity | 接触缓冲区最多能保存的数据量 |
| `solver.config.enable_self_collision` | 变形体自碰撞路径，不是 shape 碰撞的总开关 |

`s3d.ModelBuilder` 提供了与普通 Newton builder 不同的接触预设。
当接触不稳定时先减小 dt 和消除初始穿透，再调整刚度、阻尼和迭代预算。
仅提高刚度可能让线性系统更难收敛。

## 关节与驱动

支持路径包含 ball、revolute、prismatic、fixed、free，以及可映射的 D6 约束。
模型中的关节几何、轴、父子 body 和初始状态仍由 Newton 编写。
简单示例和完整 URDF 场景见：

```bash
uv run python examples/example_basic_joints.py --help
uv run python examples/example_robot_g1.py --help
uv run python examples/example_rrabd_joints.py --help
```

先检查各示例的 `--solver` 选项，有的脚本默认不是 Style3D Sim。
`example_rrabd_joints.py` 演示多个 joint 自由度的正弦/余弦目标，可直接查看它对目标和速度目标的写入。

驱动来自 Newton `control` 中的目标、目标速度和广义力；驱动类型、增益和力矩限制来自模型。
`joint_f` 不是关节位置；prismatic 位置单位为 m，角度目标使用 rad。
如果在 step 前调用 FK 将 body 摆到目标位置，会把物理驱动变成外部强制姿态设置。

标量角关节通过 `joint_limit_lower/upper`（rad）和 `joint_limit_ke/kd` 配置单边限位。
限位独立于电机输出限制；有限刚度允许受载后出现少量越界。
局部重置时，使用 `solver.sync_affine_body_state(state, body_mask=mask)`，其中 mask 是形状为
`(model.body_count,)` 的 int32 CUDA 数组。非零项选中 body 及其相连关节的角度历史。
前面的 FK 更新也应使用相同的选择范围，以保留其他环境的状态。

## 查看电机输出

```python
solver.enable_joint_drive_feedback()
solver.step(state, following, control, contacts, 1.0 / 240.0)
effort = solver.joint_drive_effort.numpy().copy()
```

这段依赖已有的驱动场景。数组按 Newton DOF 排列，线性驱动单位 N，角驱动单位 N·m。
记录值对应积分前的驱动力，不包含被动阻尼、接触力或关节约束反力，因此不是总关节反作用力传感器。
下一个 step 会覆盖数组；保留历史时复制。启用反馈需在 CUDA Graph 捕获之前进行。

## 额外 body 约束

下面片段在 `finalize()` **之前**调用，`body` 是已经创建的动态刚体：

```python
builder.add_body_spring(
    body, -1,
    point0=(0.0, 0.0, 0.1), point1=(0.0, 0.0, 1.5),
    stiffness=1.0e4,
)
```

| 方法 | 用途 |
| --- | --- |
| `add_body_spring()` | 两个 body 点或 body 点与世界锚点之间的弹簧 |
| `add_body_distance_constraint()` | 使用 AL 的硬距离约束 |
| `add_body_point_line()` | 将 body 局部点约束到另一 body 或世界线段 |

body ID 为 `-1` 时，对应点是世界坐标；否则是该 body 的局部坐标。
`rest_length=None` 使用初始姿态的点距；`rest_length=0` 则将两点拉到一起。
point-line 使用有限线段，投影越过端点会钳制到端点，不是无限导轨。
完整签名见 [Python API](../reference/python.md)。
