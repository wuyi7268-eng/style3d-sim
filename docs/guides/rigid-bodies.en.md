# Rigid Bodies, Contacts, and Joints

Style3D Sim solves rigid bodies through affine particle proxies and reconstructs Newton body poses and velocities.
Define scenes with Newton bodies, shapes, and joints; proxy matrices are handled by the adapter.

## Create a Falling Box

Replace the builder section of [First Simulation](../getting-started/first-simulation.en.md) with this model setup.
Keep the same state, control, pipeline, and stepping loop afterwards:

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

`hx/hy/hz` are half-extents, so this box has 0.2 m sides.
Shapes provide collision geometry and usually contribute to mass and inertia generation;
bodies provide identity and pose. `add_body()` alone does not create a collidable box.

For custom mass and inertia, consult the examples using `mass`, `inertia`, `com`, and `lock_inertia`.
Consider the interaction between shape density and body inertia when overriding them.

## Collision Detection and Contact Materials

```python
pipeline = newton.CollisionPipeline(model)
contacts = pipeline.contacts()
pipeline.collide(state, contacts)
solver.step(state, following, control, contacts, 1.0 / 240.0)
```

This fragment assumes states and control already exist.
Updating contacts each substep reduces the risk of missed collisions during fast motion.

| Setting | Meaning |
| --- | --- |
| `builder.default_shape_cfg.ke` | Contact stiffness for subsequently created shapes |
| `builder.default_shape_cfg.kd` | Shape contact damping |
| `builder.default_shape_cfg.mu` | Friction coefficient |
| `model.soft_contact_ke/kd/mu` | Contact parameters for particles/cloth against shapes |
| Pipeline contact capacity | Maximum amount of contact data the buffers can hold |
| `solver.config.enable_self_collision` | Deformable self-collision path; shape collisions have a separate path |

`s3d.ModelBuilder` uses different contact presets from the regular Newton builder.
For unstable contacts, first reduce `dt` and remove initial penetration, then adjust stiffness, damping, and iteration budgets.
Higher stiffness can make the linear system harder to converge.

## Joints and Drives

Supported paths include ball, revolute, prismatic, fixed, free, and compatible D6 constraints.
Newton defines the geometry, axes, parent/child bodies, and initial state.
Inspect simple and URDF examples:

```bash
uv run python examples/example_basic_joints.py --help
uv run python examples/example_robot_g1.py --help
uv run python examples/example_rrabd_joints.py --help
```

Check the `--solver` option: some scripts do not default to Style3D Sim.
`example_rrabd_joints.py` demonstrates sine/cosine targets across multiple DOFs
and shows how to write position/velocity targets.

Drives read targets, target velocities, and generalized forces from Newton `control`;
the model supplies drive types, gains, and torque limits.
`joint_f` is not a position. Prismatic positions use meters and angular targets use radians.
Calling FK before every step to place bodies at their targets externally forces poses instead of simulating drives.

Scalar angular joints use `joint_limit_lower/upper` (radians) and `joint_limit_ke/kd` for unilateral stops.
These stops act independently of motor effort limits; finite stiffness permits a small overshoot under load.
When resetting selected bodies, call `solver.sync_affine_body_state(state, body_mask=mask)` with an int32 CUDA
array of shape `(model.body_count,)`. Nonzero entries select bodies and their connected joint-angle histories.
Apply the same selection to the preceding FK update so other environments keep their state.

## Read Motor Output

```python
solver.enable_joint_drive_feedback()
solver.step(state, following, control, contacts, 1.0 / 240.0)
effort = solver.joint_drive_effort.numpy().copy()
```

This requires an existing driven scene. Values follow Newton DOF order: N for linear drives, N·m for angular drives.
They report pre-integration drive efforts and exclude passive damping, contact forces, and joint constraint reactions.
They therefore do not measure total joint reaction force.
The next step overwrites the array; copy it for history. Enable feedback before CUDA Graph capture.

## Additional Body Constraints

Call this **before** `finalize()`; `body` is an existing dynamic body:

```python
builder.add_body_spring(
    body, -1,
    point0=(0.0, 0.0, 0.1), point1=(0.0, 0.0, 1.5),
    stiffness=1.0e4,
)
```

| Method | Purpose |
| --- | --- |
| `add_body_spring()` | Spring between body points or a body point and a world anchor |
| `add_body_distance_constraint()` | Hard distance constraint using augmented Lagrangian (AL) |
| `add_body_point_line()` | Constrain a body-local point to another body's or the world's line segment |

For body ID `-1`, the associated point uses world coordinates; otherwise it uses body-local coordinates.
`rest_length=None` uses the initial point distance; `rest_length=0` pulls the points together.
Point-line constraints use finite segments and clamp projections at the endpoints.
See [Python API](../reference/python.en.md) for complete signatures.
