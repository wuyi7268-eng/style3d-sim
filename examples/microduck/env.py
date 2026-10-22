"""Simulator-neutral MicroDuck tasks (position PD, not the mjlab BAM recipe)."""

from __future__ import annotations

import newton
import torch
import warp as wp

from examples.locomotion.math_utils import projected_gravity, quat_rotate_inverse

from .policy_spec import EFFORT, HOME, KD, KP, build_robot, filter_authored_collider
from .stability import check_physics
from .tasks import TASKS, TaskLogic


class MicroDuckEnv:
    """RSL-RL VecEnv with identical controls, observations and rewards on both solvers."""

    num_actions = 14
    num_obs = num_privileged_obs = 61
    max_episode_length = 1000

    def __init__(
        self,
        solver="mujoco",
        num_envs=64,
        device="cuda:0",
        seed=42,
        viewer=None,
        task="velocity",
        recipe="official_adapted_v1",
        enable_self_collisions=False,
    ):
        if solver not in ("mujoco", "synreal") or num_envs < 1:
            raise ValueError("solver must be mujoco or synreal; num_envs must be positive")
        if task not in TASKS:
            raise ValueError(f"unknown MicroDuck task: {task}")
        if recipe != "official_adapted_v1":
            raise ValueError(f"unknown recipe: {recipe}")
        self.recipe = recipe
        self.physics_dt = 0.005
        self.decimation = 4
        self.step_dt = self.physics_dt * self.decimation
        self.official = None
        self.task = task
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.backend = solver
        self.cfg = {
            "solver": solver,
            "task": task,
            "task_recipe": recipe,
            "actuator": "position_pd",
            "kp": KP,
            "kd": KD,
            "effort_limit": EFFORT,
            "policy_dt": self.step_dt,
            "recipe": recipe,
            "self_collisions": enable_self_collisions,
            "physics_dt": self.physics_dt,
            "decimation": self.decimation,
        }
        self.viewer = viewer
        self.sim_time = 0.0
        torch.manual_seed(seed)
        wp.init()
        builder_type = newton.ModelBuilder
        if solver == "synreal":
            import style3d_sim as s3d

            builder_type = s3d.ModelBuilder
        # This is intentionally configurable for playback diagnostics. Training
        # keeps the existing collision-free articulation unless explicitly changed.
        robot, qi, qdi = build_robot(builder_type, enable_self_collisions=enable_self_collisions, task=task)
        self.task_logic = TaskLogic(task, robot)
        if task == "ballkick":
            filter_authored_collider(robot, len(robot.shape_body) - 1)
        builder = builder_type(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.replicate(robot, num_envs)
        ground = builder.add_ground_plane()
        filter_authored_collider(builder, ground)
        self.model = builder.finalize(device=device)
        from .official_randomization import randomize_foot_friction

        self.cfg["foot_friction_randomization"] = randomize_foot_friction(self.model, task, num_envs, self.device)
        self.model.set_gravity((0.0, 0.0, -9.81))
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)
        if solver == "mujoco":
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                integrator="implicitfast",
                solver="newton",
                iterations=20,
                update_data_interval=1,
                # Recovery poses exceed the standing broadphase estimate (~1900 mesh pairs).
                # Reserve candidate slots as well as actual constraint rows for MJWarp.
                njmax=256,
                nconmax=4096,
            )
        else:
            self.solver = s3d.NewtonSolver(self.model)
            self.solver.config.joint_constraint_mode = s3d.JointConstraintMode.AUGMENTED_LAGRANGIAN
            self.solver.config.affine_body_rigidity_model = s3d.AffineBodyRigidityModel.SO3_AUGMENTED_LAGRANGIAN
            # Fallen-pose contact corrections can approach a half-turn in one step.
            # Cayley's inverse then amplifies the recovered angular velocity and feeds
            # it back into the next solve. Use the paired, dissipative implicit path.
            self.solver.config.inertia_mode = s3d.InertiaMode.IMPLICIT_EULER
            self.solver.config.constraint_alpha = 200.0
            self.solver.config.nonlinear_iterations = 5
            self.solver.config.linear_iterations = 30
        self.cfg["inertia_integrator"] = "implicit_euler" if solver == "synreal" else "implicitfast"
        self.collision = newton.CollisionPipeline(self.model)
        self.contacts = self.collision.contacts()
        self.control = self.model.control()
        self.states = [self.model.state(), self.model.state()]
        self.qi = torch.tensor(qi, device=self.device)
        self.qdi = torch.tensor(qdi, device=self.device)
        self.home = torch.tensor(HOME, device=self.device)
        self.initial_q = wp.to_torch(self.model.joint_q).reshape(num_envs, -1).clone()
        self.actions = torch.zeros(num_envs, 14, device=self.device)
        self.commands = torch.zeros(num_envs, 13, device=self.device)
        self.commands[:, 0] = 0.1
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.total_resets = 0
        self.task_logic.initialize(self)
        from .official_recipe import OfficialRecipe

        self.official = OfficialRecipe(self, robot)
        self.cfg["task_recipe"] = recipe
        self.cfg["official_alignment"] = self.official.alignment
        self._reset(torch.arange(num_envs, device=self.device))
        self.graph = None
        self.cfg["cuda_graph"] = self.graph is not None
        if viewer:
            viewer.set_model(self.model)
            viewer.set_world_offsets((0.5, 0.5, 0.0))

    def _sync(self):
        # Torch and Warp may use different CUDA streams; synchronize at ownership boundaries.
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            wp.synchronize_device(str(self.device))

    def _views(self, state=None):
        state = self.states[0] if state is None else state
        return (
            wp.to_torch(state.joint_q).reshape(self.num_envs, -1),
            wp.to_torch(state.joint_qd).reshape(self.num_envs, -1),
            wp.to_torch(state.body_q).reshape(self.num_envs, -1, 7)[:, 0],
            wp.to_torch(state.body_qd).reshape(self.num_envs, -1, 6)[:, 0],
        )

    def _reset(self, ids):
        self._sync()
        from .official_events import prepare_reset

        prepare_reset(self.official, ids)
        for state in self.states:
            q, qd, _, _ = self._views(state)
            q[ids] = self.initial_q[ids]
            qd[ids] = 0.0
            self.task_logic.reset_state(self, ids, q)
            self._sync()
            newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)
            if self.backend == "synreal":
                self.solver.sync_affine_body_state(state)
        self.actions[ids] = 0.0
        self.episode_length_buf[ids] = 0
        self.official.reset(ids)
        self._sync()

    def reset(self):
        self._reset(torch.arange(self.num_envs, device=self.device))
        return self.get_observations()

    def _kinematics(self):
        self._sync()
        q, qd, pose, velocity = self._views()
        quat = torch.nn.functional.normalize(pose[:, 3:7], dim=-1)
        linear = velocity[:, :3]
        from .official_adapter import link_origin_velocity

        linear = link_origin_velocity(linear, velocity[:, 3:], quat, wp.to_torch(self.model.body_com)[0])
        return (
            q,
            qd,
            pose,
            quat_rotate_inverse(quat, linear),
            quat_rotate_inverse(quat, velocity[:, 3:]),
            projected_gravity(quat),
        )

    def get_observations(self):
        actor, critic = self.official.observations.compute()
        return actor, {"observations": {"critic": critic}}

    def step_playback(self, actions):
        """Replay through the same observation, reset and event pipeline as training."""
        return self.step(actions)[0]

    @torch.no_grad()
    def step(self, actions):
        if actions.shape != self.actions.shape or not torch.isfinite(actions).all():
            raise ValueError("Expected finite actions with shape (num_envs, 14)")
        self.official.observations.cache = None
        previous = self.actions.clone()
        converted = actions.to(self.device)
        self.actions.copy_(converted)
        desired = self.home + self.actions
        torque = torch.zeros_like(self.actions)
        targets = wp.to_torch(self.control.joint_target_q).reshape(self.num_envs, -1)
        for substep in range(self.decimation):
            self._sync()
            newton.eval_ik(self.model, self.states[0], self.states[0].joint_q, self.states[0].joint_qd)
            self._sync()
            q, qd, _, _ = self._views()
            # Encode the same clipped PD torque as a position target on both backends.
            torque = (KP * (desired - q[:, self.qi]) - KD * qd[:, self.qdi]).clamp(-EFFORT, EFFORT)
            targets[:, self.qdi] = q[:, self.qi] + (torque + KD * qd[:, self.qdi]) / KP
            self._sync()
            self.states[0].clear_forces()
            if self.backend == "synreal":
                self.collision.collide(self.states[0], self.contacts)
            self.solver.step(self.states[0], self.states[1], self.control, self.contacts, self.physics_dt)
            self.states.reverse()
            check_physics(self, substep)
            self.collision.collide(self.states[0], self.contacts)
            self.official.sensors.update()
        self._sync()
        newton.eval_ik(self.model, self.states[0], self.states[0].joint_q, self.states[0].joint_qd)
        q, qd, pose, linear, angular, gravity = self._kinematics()
        if not all(torch.isfinite(x).all() for x in (q, qd, pose, linear, angular)):
            raise FloatingPointError(f"{self.backend}: non-finite physics state")
        self.episode_length_buf += 1
        timeout = self.episode_length_buf >= self.max_episode_length
        fallen = (pose[:, 2] < 0.065) | (gravity[:, 2] > -0.3)
        reward, task_terms = self.official.reward(q, qd, pose, linear, angular, gravity, torque, previous)
        from .official_events import termination

        failure = termination(self.official, gravity)
        done = failure | timeout
        self.sim_time += self.step_dt
        # Record pre-reset geometry so simulation diagnostics cannot hide a fall behind auto-reset.
        physics = {"min_trunk_z": pose[:, 2].min().item(), "min_upright_cos": (-gravity[:, 2]).min().item()}
        ids = done.nonzero().flatten()
        if len(ids):
            self.total_resets += len(ids)
            self._reset(ids)
        if self.viewer is None or self.task == "groundpick":
            self.official.update_commands()
        self.official.pushes.apply()
        obs, extras = self.get_observations()
        extras["time_outs"] = timeout
        extras["physics"] = physics
        extras["log"] = {
            "Physics/reset_count": float(self.total_resets),
            "Physics/upright": (-gravity[:, 2]).mean(),
            "Tracking/forward_velocity": linear[:, 0].mean(),
            "Tracking/velocity_error": (linear[:, :2] - self.commands[:, :2]).norm(dim=-1).mean(),
            "Physics/fall_fraction": fallen.float().mean(),
        }
        if self.task not in ("velocity", "velstand"):
            # Phase commands are not velocities; avoid misleading velocity-tracking diagnostics.
            extras["log"] = {key: value for key, value in extras["log"].items() if not key.startswith("Tracking/")}
        extras["log"].update({f"Task/{key}": value.mean() for key, value in task_terms.items()})
        extras["log"].update(self.official.extras["log"])
        if self.viewer:
            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.states[0])
            self.viewer.end_frame()
        return obs, reward, done, extras

    def close(self):
        if self.viewer:
            self.viewer.close()
