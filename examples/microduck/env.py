"""Shared MicroDuck environment: training, policy replay, HOME holding and scene preview."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from types import SimpleNamespace

import newton
import torch
import warp as wp

from examples.locomotion.math_utils import projected_gravity, quat_rotate_inverse

from .policy_spec import (
    EFFORT,
    HOME,
    JOINT_NAMES,
    KD,
    KP,
    TaskLogic,
    build_robot,
    filter_authored_collider,
    joint_coordinates,
)
from .scenes import (
    SCENE_KINDS,
    MixedTerrain,
    OfficialTerrain,
    SceneTerrain,
    SoftPadConfig,
    add_inset_floor,
    add_scene,
    add_terrain_pad,
    pad_world_count,
    provision_pad_contacts,
)
from .stability import check_physics
from .task_registry import TASKS, base_task


@dataclass(frozen=True)
class PhysicsConfig:
    """One policy dynamics profile for training and replay; HOME has an explicit override."""

    physics_dt: float = 0.005
    decimation: int = 4
    nonlinear_iterations: int = 5
    linear_iterations: int = 50
    joint_constraint_primal_iterations: int = 1
    affine_body_block_preconditioner: bool = False
    air_density: float = 1.225

    def __post_init__(self):
        if not math.isfinite(self.physics_dt) or self.physics_dt <= 0:
            raise ValueError("physics_dt must be finite and positive")
        for name in ("decimation", "nonlinear_iterations", "linear_iterations", "joint_constraint_primal_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.affine_body_block_preconditioner, bool):
            raise TypeError("affine_body_block_preconditioner must be a boolean")
        if not math.isfinite(self.air_density) or self.air_density < 0:
            raise ValueError("air_density must be finite and nonnegative")

    @classmethod
    def home(cls):
        return cls(physics_dt=1.0 / 480.0, decimation=8, linear_iterations=30,
                   joint_constraint_primal_iterations=3, affine_body_block_preconditioner=True, air_density=0.0)

    @classmethod
    def from_checkpoint(cls, metadata):
        if "physics" in metadata:
            return cls(**metadata["physics"])
        defaults = cls()
        dt = metadata.get("physics_dt", defaults.physics_dt)
        period = metadata.get("policy_dt", defaults.physics_dt * defaults.decimation)
        if not math.isfinite(dt) or dt <= 0 or not math.isfinite(period) or period <= 0:
            raise ValueError("Checkpoint timing must be finite and positive")
        count = period / dt
        if not math.isclose(count, round(count), rel_tol=1.0e-6):
            raise ValueError("Checkpoint physics_dt must divide policy_dt")
        # Legacy checkpoints did not record the complete solve. Keep the validated
        # 5/15 policy defaults for missing fields; honor recorded 5/50 runs explicitly.
        return replace(defaults, physics_dt=dt, decimation=round(count), **metadata.get("solver_settings", {}))

    def with_substeps(self, count):
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("substeps must be a positive integer")
        return replace(self, physics_dt=self.physics_dt * self.decimation / count, decimation=count)

    def configure_synreal(self, solver):
        import style3d_sim as s3d

        cfg = solver.config
        cfg.joint_constraint_mode = s3d.JointConstraintMode.AUGMENTED_LAGRANGIAN
        cfg.affine_body_rigidity_model = s3d.AffineBodyRigidityModel.SO3_AUGMENTED_LAGRANGIAN
        # Keep the dissipative integrator paired with the trained contact solve.
        cfg.inertia_mode = s3d.InertiaMode.IMPLICIT_EULER
        cfg.nonlinear_solver = s3d.NonlinearSolver.NEWTON
        cfg.constraint_alpha = 200.0
        cfg.enable_collision_hessian_non_diagonal = True
        for name, value in asdict(self).items():
            if name not in ("physics_dt", "decimation"):
                setattr(cfg, name, value)


# Geometry construction lives in scenes.py; edit experiment materials here.
PAD_CONFIG = SoftPadConfig(
    # Cells per meter keep mesh spacing fixed when size changes (about 2.67 cm XY, 2 cm Z).
    size=(1.2, 1.2, 0.12), cells_per_meter=(35, 35, 4.0), base_z=0,
    # Native material damping approximates slow-recovery foam without permanent deformation.
    # Shear stiffness supports foot loads; damping slows both indentation and elastic recovery.
    density=80.0, shear_modulus=60.0, lame_lambda=7000.0,
    damping=35.0, particle_radius=0.0015, fix_bottom=True,
    # At 5/50, the previous 50 kN/m contact penalty inverted surface tetrahedra
    # under foot impact. Lower contact stiffness without hardening the pad itself.
    contact_stiffness=5000.0
)


@dataclass(frozen=True)
class EnvironmentConfig:
    """Central runtime profile shared by training, replay, and terrain selection."""

    name: str
    num_envs: int = 1024
    playback_num_envs: int = 1
    terrain: str = "default"
    pad_probability: float = 0.5
    pad_sampling: str = "fixed_groups_center"
    randomize_foot_friction: bool = True
    playback_randomize_foot_friction: bool = False
    spawn: str = "pad"
    clearance: float = 0.003
    physics: PhysicsConfig = PhysicsConfig()
    pad: SoftPadConfig = PAD_CONFIG

    def __post_init__(self):
        for count in (self.num_envs, self.playback_num_envs):
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError("environment counts must be positive integers")
        if self.terrain not in ("default", "mixed_soft_pad", *SCENE_KINDS):
            raise ValueError(f"unknown environment terrain: {self.terrain}")
        if not math.isfinite(self.pad_probability) or not 0.0 <= self.pad_probability <= 1.0:
            raise ValueError("pad_probability must be in [0, 1]")
        if self.pad_sampling not in ("fixed_groups_center", "independent_per_episode"):
            raise ValueError("unknown pad sampling mode")
        if self.spawn not in ("pad", "ground") or not math.isfinite(self.clearance) or self.clearance < 0:
            raise ValueError("spawn must be pad or ground; clearance must be finite and nonnegative")

    def cli_defaults(self, mode):
        """Explicit CLI options can temporarily override these experiment defaults."""
        return {
            "num_envs": self.num_envs if mode == "train" else self.playback_num_envs,
            "terrain": self.terrain,
            "pad_probability": self.pad_probability,
            "randomize_foot_friction": (self.randomize_foot_friction if mode == "train"
                                        else self.playback_randomize_foot_friction),
        }


# Common defaults above apply to both profiles. Override a field here only when
# flat and mixed experiments intentionally need different environment settings.
ENVIRONMENT_CONFIGS = {
    "official": EnvironmentConfig(name="official", terrain="default"),
    "mixed_soft_pad": EnvironmentConfig(
        name="mixed_soft_pad", terrain="mixed_soft_pad",
        # Keep policy updates at 50 Hz while refining soft-contact and joint solves.
        physics=PhysicsConfig(physics_dt=0.0025, decimation=8, joint_constraint_primal_iterations=4),
    ),
}
for _scene_kind in SCENE_KINDS:
    _scene_num_envs = 1 if _scene_kind in ("sand", "pool") else 1024
    ENVIRONMENT_CONFIGS[_scene_kind] = EnvironmentConfig(
        name=_scene_kind, terrain=_scene_kind, num_envs=_scene_num_envs
    )


def environment_config(name):
    try:
        return ENVIRONMENT_CONFIGS[name]
    except KeyError as exc:
        raise ValueError(f"unknown environment profile: {name}") from exc


def physics_matches(left, right):
    """Native metadata can round float settings to float32 before serialization."""
    saved = asdict(right)
    return all(math.isclose(value, saved[name], rel_tol=1.0e-6, abs_tol=1.0e-9)
               for name, value in asdict(left).items())


def resolve_physics(profile, metadata=None, *, use_checkpoint=False, home=False):
    current = PhysicsConfig.home() if home else profile.physics
    if metadata is None:
        if use_checkpoint:
            raise ValueError("--use-checkpoint-environment requires a checkpoint")
        return current
    saved = PhysicsConfig.from_checkpoint(metadata)
    if use_checkpoint:
        return saved
    if not physics_matches(saved, current):
        print(f"Using env.py physics: nonlinear={current.nonlinear_iterations}, linear={current.linear_iterations}, "
              f"dt={current.physics_dt:g}, substeps={current.decimation}, "
              f"joint_primal={current.joint_constraint_primal_iterations}; "
              f"checkpoint recorded nonlinear={saved.nonlinear_iterations}, linear={saved.linear_iterations}, "
              f"dt={saved.physics_dt:g}, substeps={saved.decimation}, "
              f"joint_primal={saved.joint_constraint_primal_iterations}. "
              "Use --use-checkpoint-environment to restore saved physics.", flush=True)
    return current


class MicroDuckEnv:
    """RSL-RL VecEnv with identical controls, observations and rewards on both solvers."""

    num_actions = 14
    num_obs = num_privileged_obs = 61
    max_episode_length = 1000

    def __init__(
        self,
        solver="mujoco",
        num_envs=None,
        device="cuda:0",
        seed=42,
        viewer=None,
        task="velocity",
        recipe="official_adapted_v1",
        enable_self_collisions=False,
        randomize_foot_friction=None,
        terrain="default",
        pad_probability=None,
        pad_config=None,
        physics=None,
        interactive=False,
        controller="policy",
        preview=False,
        spawn=None,
        clearance=None,
        pad_sampling=None,
        reset_on_fall=True,
    ):
        profile = environment_config("mixed_soft_pad" if terrain == "mixed_soft_pad" else "official")
        if randomize_foot_friction is None:
            randomize_foot_friction = (profile.playback_randomize_foot_friction if interactive
                                       else profile.randomize_foot_friction)
        if num_envs is None:
            num_envs = profile.playback_num_envs if interactive else profile.num_envs
        pad_probability = profile.pad_probability if pad_probability is None else pad_probability
        pad_sampling = profile.pad_sampling if pad_sampling is None else pad_sampling
        spawn = profile.spawn if spawn is None else spawn
        clearance = profile.clearance if clearance is None else clearance
        if solver not in ("mujoco", "synreal") or num_envs < 1:
            raise ValueError("solver must be mujoco or synreal; num_envs must be positive")
        if task not in TASKS:
            raise ValueError(f"unknown MicroDuck task: {task}")
        if recipe != "official_adapted_v1":
            raise ValueError(f"unknown recipe: {recipe}")
        if terrain not in ("default", "mixed_soft_pad", *SCENE_KINDS):
            raise ValueError(f"unknown terrain: {terrain}")
        if terrain != "default" and solver != "synreal":
            raise ValueError("Optional terrain scenes require SynReal")
        if controller not in ("policy", "home"):
            raise ValueError("controller must be policy or home")
        if (preview or controller == "home") and not interactive:
            raise ValueError("Preview and HOME holding require an interactive environment")
        if interactive and (num_envs != 1 or terrain == "default"):
            raise ValueError("Interactive scenes require one non-default terrain world")
        if spawn not in ("pad", "ground") or not math.isfinite(clearance) or clearance < 0:
            raise ValueError("spawn must be pad or ground and clearance must be finite and nonnegative")
        self.interactive, self.preview, self.controller = interactive, preview, controller
        self.reset_on_fall = bool(reset_on_fall)
        self.spawn, self.clearance = spawn, clearance
        self.mixed_terrain = terrain == "mixed_soft_pad"
        self.scene_kind = terrain
        self.scene_terrain = terrain != "default"
        self.recipe = recipe
        default_physics = PhysicsConfig.home() if controller == "home" else profile.physics
        self.physics = PhysicsConfig(**physics) if isinstance(physics, dict) else physics or default_physics
        self.physics_dt = self.physics.physics_dt
        self.decimation = self.physics.decimation
        self.step_dt = self.physics_dt * self.decimation
        self.official = None
        self.task = task
        self.family = base_task(task)
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
            "randomize_foot_friction": randomize_foot_friction,
            "physics_dt": self.physics_dt,
            "decimation": self.decimation,
            "physics": asdict(self.physics),
            "interactive": interactive,
            "controller": controller,
            "reset_on_fall": self.reset_on_fall,
        }
        self.viewer = viewer
        self.sim_time = 0.0
        self.steps = 0
        if seed is not None:
            torch.manual_seed(seed)
        wp.init()
        builder_type = newton.ModelBuilder
        if solver == "synreal":
            import style3d_sim as s3d

            builder_type = s3d.ModelBuilder
        # This is intentionally configurable for playback diagnostics. Training
        # keeps the existing collision-free articulation unless explicitly changed.
        robot, qi, qdi = build_robot(builder_type, enable_self_collisions=enable_self_collisions, task=task)
        self.robot_mass = float(sum(robot.body_mass))
        self.root = next(i for i, name in enumerate(robot.body_label) if name.rsplit("/", 1)[-1] == "trunk_base")
        self.task_logic = TaskLogic(task, robot)
        if self.family == "ballkick":
            filter_authored_collider(robot, len(robot.shape_body) - 1)
        if self.mixed_terrain:
            pad_config = SoftPadConfig(**pad_config) if isinstance(pad_config, dict) else pad_config or profile.pad
            provision_pad_contacts(robot)
            if interactive:
                robot.joint_q[2] += clearance + (pad_config.top_z if spawn == "pad" else 0.0)
                if spawn == "ground":
                    robot.joint_q[0] = -pad_config.size[0] / 2.0 - 0.3
            if interactive or pad_sampling == "independent_per_episode":
                add_terrain_pad(robot, pad_config)
                if interactive and pad_config.embedded:
                    add_inset_floor(robot, pad_config, has_pad=True)
        elif terrain != "default":
            scene_metadata = add_scene(robot, terrain)
        # Preserve the standalone topology for interactive diagnostics. Batched RL
        # replicates the same template, including a single-world training run.
        builder = robot if interactive else builder_type(up_axis=newton.Axis.Z)
        if not interactive:
            newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
            if terrain == "hammock":
                newton.solvers.SolverStyle3D.register_custom_attributes(builder)
            if self.mixed_terrain and pad_sampling == "fixed_groups_center":
                # Keep robot/body/joint ordering identical while omitting all soft
                # geometry from the flat cohort. Both cohorts share one policy.
                count = pad_world_count(num_envs, pad_probability)
                flat_template = robot
                if pad_config.embedded:
                    flat_template = builder_type(up_axis=newton.Axis.Z)
                    flat_template.add_builder(robot)
                    add_inset_floor(flat_template, pad_config, has_pad=False)
                builder.replicate(flat_template, num_envs - count)
                if count:
                    pad_template = builder_type(up_axis=newton.Axis.Z)
                    pad_template.add_builder(robot)
                    add_terrain_pad(pad_template, pad_config)
                    if pad_config.embedded:
                        add_inset_floor(pad_template, pad_config, has_pad=True)
                    builder.replicate(pad_template, count)
            else:
                if self.mixed_terrain and pad_config.embedded:
                    add_inset_floor(robot, pad_config, has_pad=True)
                builder.replicate(robot, num_envs)
        if self.mixed_terrain:
            self.terrain = MixedTerrain(self, builder, pad_config, pad_probability, sampling=pad_sampling)
        elif terrain != "default":
            self.terrain = SceneTerrain(self, terrain, scene_metadata)
        else:
            self.terrain = OfficialTerrain(self, builder)
        self.model = builder.finalize(device=device)
        if self.mixed_terrain or self.scene_terrain:
            self.terrain.finalize()
        from .events import randomize_foot_friction as apply_foot_friction

        self.cfg["foot_friction_randomization"] = apply_foot_friction(
            self.model,
            task,
            num_envs,
            self.device,
            enabled=randomize_foot_friction,
        )
        self.model.set_gravity((0.0, 0.0, -9.81))
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)
        if preview:
            self.solver = None
        elif solver == "mujoco":
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                integrator="implicitfast",
                use_mujoco_cpu=self.device.type == "cpu",
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
            self.physics.configure_synreal(self.solver)
            self.cfg["solver_settings"] = {
                name: getattr(self.solver.config, name)
                for name in ("nonlinear_iterations", "linear_iterations", "joint_constraint_primal_iterations",
                             "affine_body_block_preconditioner", "air_density")
            }
        self.cfg["inertia_integrator"] = "implicit_euler" if solver == "synreal" else "implicitfast"
        collision_options = {}
        if self.mixed_terrain or terrain == "hammock":
            # Keep the viewer's arbitrary-size pad budget and the replicated
            # training budget; capacity is storage, not a different contact law.
            # Refined pads produce ground contacts across their full surface; a fixed
            # per-world budget overflows even before the robot touches the pad.
            soft_capacity = (max(32768, 8 * self.model.particle_count) if interactive
                             else max(4096 * num_envs, 2 * self.model.tri_count))
            # Full-body recovery/ball/roller assets can exceed the two-foot model's 128 rigid contacts.
            rigid_per_world = max(128, 8 * sum(bool(flags & newton.ShapeFlags.COLLIDE_SHAPES) for flags in robot.shape_flags))
            collision_options = {
                "enable_rigid_soft_full_surface_contact": True,
                "soft_contact_margin": 0.01,
                "soft_contact_max": soft_capacity,
                "rigid_contact_max": 10240 if interactive else rigid_per_world * num_envs,
            }
        elif terrain != "default":
            # Grain-to-shape contacts still use this pipeline, including the tray.
            collision_options = {"soft_contact_max": max(4096 * num_envs, 8 * self.model.particle_count)}
        self.cfg["collision_options"] = collision_options.copy()
        self.collision = newton.CollisionPipeline(self.model, **collision_options)
        self.contacts = self.collision.contacts()
        if solver == "synreal" and self.solver is not None:
            # Reserve the complete collision-pipeline capacity before any solver step.
            # Static previews deliberately have no solver to provision.
            self.solver.prepare_contacts(self.contacts)
        self.control = self.model.control()
        self.states = [self.model.state(), self.model.state()]
        self.qi = torch.tensor(qi, device=self.device)
        self.qdi = torch.tensor(qdi, device=self.device)
        self.home = torch.tensor(HOME, device=self.device)
        # Present a stable servo-first view while preserving all passive DOFs in physics.
        coordinates = joint_coordinates(robot)
        passive = [name for name in coordinates if name.startswith("passive_")]
        self.joint_names = list(JOINT_NAMES) + passive
        self.all_qi = torch.tensor(qi + [coordinates[name][0] for name in passive], device=self.device)
        self.all_qdi = torch.tensor(qdi + [coordinates[name][1] for name in passive], device=self.device)
        self.backlash_qi = self.backlash_qdi = None
        if task.endswith("_backlash"):
            indices = [self.joint_names.index(f"passive_{name}_backlash") for name in self.joint_names[:14]]
            self.backlash_qi, self.backlash_qdi = self.all_qi[indices], self.all_qdi[indices]
        self.initial_q = wp.to_torch(self.model.joint_q).reshape(num_envs, -1).clone()
        self.actions = torch.zeros(num_envs, 14, device=self.device)
        self.commands = torch.zeros(num_envs, 13, device=self.device)
        self.commands[:, 0] = 0.1
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.total_resets = 0
        self.task_logic.initialize(self)
        if interactive:
            for state in self.states:
                newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, state)
                if self.solver is not None:
                    self.solver.sync_affine_body_state(state)
            self.initial_state = self.model.state()
            self.initial_state.assign(self.states[0])
            self.initial_targets = wp.clone(self.control.joint_target_q)
            if self.mixed_terrain:
                self.terrain.spawn_on_pad[:] = spawn == "pad"
                self.terrain.origins[:, 2] = pad_config.top_z if spawn == "pad" else 0.0
        if not interactive or controller == "policy":
            from .recipe import OfficialRecipe

            self.official = OfficialRecipe(self, robot)
            if interactive:
                self.official.observations.spec["actor"]["enable_corruption"] = False
            self.cfg["task_recipe"] = recipe
            self.cfg["official_alignment"] = self.official.alignment
            self._reset(torch.arange(num_envs, device=self.device))
        self.graph = None
        self.cfg["cuda_graph"] = self.graph is not None
        if viewer and not interactive:
            viewer.set_model(self.model)
            # Separate full pads and leave room for the adjacent flat-ground spawn.
            if self.mixed_terrain:
                spacing = (pad_config.size[0] + 2.0, pad_config.size[1] + 1.0, 0.0)
            elif terrain == "hammock":
                spacing = (2.2, 1.3, 0.0)
            elif terrain == "sand":
                spacing = (1.8, 1.6, 0.0)
            elif terrain == "pool":
                config = self.terrain.cfg["config"]
                spacing = (config["pool_length"] + 0.3, config["pool_width"] + 0.3, 0.0)
            elif terrain == "steps":
                spacing = (2.0, 1.8, 0.0)
            else:
                spacing = (0.5, 0.5, 0.0)
            if self.mixed_terrain and pad_config.embedded:
                # Finite floor tiles must not visually cover a neighbor's cavity.
                spacing = (*[2 * extent + 0.2 for extent in pad_config.floor_half_extents], 0.0)
            viewer.set_world_offsets(spacing)
            if self.scene_terrain and not self.mixed_terrain:
                self.terrain.configure_viewer(viewer)

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

    def sync_selected_state(self, state, ids):
        """Rebuild externally edited worlds without projecting surviving worlds or resetting their drive histories."""
        if not len(ids):
            return
        selected = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        selected[ids] = True
        articulation_world = wp.to_torch(self.model.articulation_world).long()
        if getattr(self, "interactive", False):
            # The standalone viewer labels its sole articulation world -1.
            articulation_world = articulation_world.clamp_min(0)
        articulation_mask = ((articulation_world >= 0) & selected[articulation_world.clamp_min(0)]).contiguous()
        self._sync()
        # Mask FK as well as proxy synchronization: joint reconstruction discards
        # constraint residuals still present in the integrated body-space state.
        newton.eval_fk(self.model, state.joint_q, state.joint_qd, state, mask=wp.from_torch(articulation_mask))
        if self.backend == "synreal":
            body_world = wp.to_torch(self.model.body_world).long()
            if getattr(self, "interactive", False):
                body_world = body_world.clamp_min(0)
            body_mask = ((body_world >= 0) & selected[body_world.clamp_min(0)]).to(torch.int32).contiguous()
            self._sync()
            self.solver.sync_affine_body_state(state, body_mask=wp.from_torch(body_mask))
        self._sync()

    def _reset(self, ids):
        self._sync()
        from .events import prepare_reset

        self.terrain.reset(ids)
        prepare_reset(self.official, ids)
        if self.scene_terrain:
            self.terrain.adjust_spawn(ids)
        for state in self.states:
            q, qd, _, _ = self._views(state)
            q[ids] = self.initial_q[ids]
            qd[ids] = self.task_logic.spawn_velocity[ids]
            self.task_logic.reset_state(self, ids, q)
            self.sync_selected_state(state, ids)
        if self.mixed_terrain:
            targets = wp.to_torch(self.control.joint_target_q).reshape(self.num_envs, -1)
            target_indices = self.qi if newton.use_coord_layout_targets else self.qdi
            targets[ids[:, None], target_indices] = self.task_logic.spawn[ids[:, None], self.qi]
        self.actions[ids] = 0.0
        self.episode_length_buf[ids] = 0
        self.official.reset(ids)
        self._sync()

    def reset(self):
        if self.interactive:
            if self.controller == "policy":
                # Recovery tasks resample their authored poses, including on a raised or embedded pad.
                self._reset(torch.arange(self.num_envs, device=self.device))
                self.sim_time, self.steps = 0.0, 0
                return None
            for state in self.states:
                state.assign(self.initial_state)
                state.clear_forces()
                if self.solver is not None:
                    self.solver.sync_affine_body_state(state)
            wp.copy(self.control.joint_target_q, self.initial_targets)
            self.actions.zero_()
            self.episode_length_buf.zero_()
            self.terrain.minimum_volume.fill_(1.0)
            self.sim_time, self.steps = 0.0, 0
            if self.official is not None:
                self.official.observations.reset(torch.arange(self.num_envs, device=self.device))
            self._sync()
            return None
        self._reset(torch.arange(self.num_envs, device=self.device))
        return self.get_observations()

    def _kinematics(self):
        self._sync()
        if self.interactive:
            newton.eval_ik(self.model, self.states[0], self.states[0].joint_q, self.states[0].joint_qd)
            self._sync()
        q, qd, pose, velocity = self._views()
        quat = torch.nn.functional.normalize(pose[:, 3:7], dim=-1)
        linear = velocity[:, :3]
        from .adapter import link_origin_velocity

        linear = link_origin_velocity(linear, velocity[:, 3:], quat, wp.to_torch(self.model.body_com)[0])
        return (
            q,
            qd,
            pose,
            quat_rotate_inverse(quat, linear),
            quat_rotate_inverse(quat, velocity[:, 3:]),
            projected_gravity(quat),
        )

    def encoder_state(self, q, qd):
        position, velocity = q[:, self.qi], qd[:, self.qdi]
        if self.backlash_qi is not None:
            position = position + q[:, self.backlash_qi]
            velocity = velocity + qd[:, self.backlash_qdi]
        return position, velocity

    def get_observations(self):
        if self.interactive:
            from .play import PolicyPlayback

            return PolicyPlayback(self).observe(), {}
        actor, critic = self.official.observations.compute()
        return actor, {"observations": {"critic": critic}}

    @torch.no_grad()
    def step_playback(self, actions):
        """Use shared physics while preserving the fast rigid-playback episode contract."""
        if self.interactive:
            self.step(actions)
            return self.get_observations()[0]
        if self.task not in ("velocity", "velstand", "groundpick", "ballkick") or self.mixed_terrain or self.scene_terrain:
            # Mixed terrain needs sensors and deformation guards, but visual
            # evaluation must not silently inherit training-only interval pushes.
            # Manual viewer forces still need to reach each physics substep.
            return self.step(actions, apply_interval_events=False, apply_viewer_forces=True)[0]
        self.advance_physics(actions, apply_viewer_forces=True)
        q, qd, pose, linear, angular, gravity = self._kinematics()
        if not all(torch.isfinite(x).all() for x in (q, qd, pose, linear, angular)):
            raise FloatingPointError(f"{self.backend}: non-finite playback state")
        self.episode_length_buf += 1
        self.official.common_step_counter += 1
        self.official.robot.data = SimpleNamespace(root_link_pos_w=pose[:, :3], root_link_quat_w=pose[:, [6, 3, 4, 5]])
        from .events import termination
        from .play import PolicyPlayback

        failure = termination(self.official, gravity) if self.reset_on_fall else torch.zeros_like(
            self.episode_length_buf, dtype=torch.bool
        )
        done = failure | (self.episode_length_buf >= self.max_episode_length)
        self.sim_time += self.step_dt
        ids = done.nonzero().flatten()
        if len(ids):
            self.total_resets += len(ids)
            self._reset(ids)
        if self.viewer is None or self.family not in ("velocity", "velstand"):
            self.official.update_commands()
        obs = PolicyPlayback(self).observe()
        if self.viewer:
            self.render()
        return obs

    @torch.no_grad()
    def apply_policy_targets(self):
        """Encode the same saturated position-PD torque for every policy entry point."""
        self._sync()
        newton.eval_ik(self.model, self.states[0], self.states[0].joint_q, self.states[0].joint_qd)
        self._sync()
        q, qd, _, _ = self._views()
        position, velocity = self.encoder_state(q, qd)
        desired = self.home + self.actions
        torque = (KP * (desired - position) - KD * velocity).clamp(-EFFORT, EFFORT)
        targets = wp.to_torch(self.control.joint_target_q).reshape(self.num_envs, -1)
        target_indices = self.qi if newton.use_coord_layout_targets else self.qdi
        targets[:, target_indices] = q[:, self.qi] + (torque + KD * qd[:, self.qdi]) / KP
        self._sync()
        return torque

    @torch.no_grad()
    def advance_physics(self, actions, *, training=False, apply_viewer_forces=False):
        """The only control/substep loop used by training and all .pt playback."""
        if actions is not None:
            if actions.shape != self.actions.shape or not torch.isfinite(actions).all():
                raise ValueError("Expected finite actions with shape (num_envs, 14)")
            self.actions.copy_(actions.to(self.device))
        if self.official is not None:
            self.official.observations.advance()
        torque = torch.zeros_like(self.actions)
        guarded = training and self.mixed_terrain
        if guarded:
            self.terrain.guard.begin_step()
        for substep in range(self.decimation):
            if actions is not None:
                torque = self.apply_policy_targets()
            self.states[0].clear_forces()
            if apply_viewer_forces and self.viewer is not None:
                self.viewer.apply_forces(self.states[0])
            if self.backend == "synreal":
                self.collision.collide(self.states[0], self.contacts)
                if self.mixed_terrain:
                    self.terrain.check_contacts()
            self.solver.step(self.states[0], self.states[1], self.control, self.contacts, self.physics_dt)
            self.states.reverse()
            if guarded:
                self.terrain.guard.after_substep(substep)
            if not self.interactive:
                check_physics(self, substep)
            if training:
                self.collision.collide(self.states[0], self.contacts)
                if self.mixed_terrain:
                    self.terrain.check_contacts()
                self.official.sensors.update()
        if guarded:
            self.terrain.guard.finish_step()
        self._sync()
        if not self.interactive:
            newton.eval_ik(self.model, self.states[0], self.states[0].joint_q, self.states[0].joint_qd)
            self._sync()
        return torque

    @torch.no_grad()
    def step(self, actions=None, *, apply_interval_events=True, apply_viewer_forces=False):
        if self.preview:
            return None
        if self.interactive:
            if self.controller == "policy" and actions is None:
                raise ValueError("Policy playback requires actions")
            self.advance_physics(actions, apply_viewer_forces=True)
            self.sim_time += self.step_dt
            self.steps += 1
            if self.official is not None:
                self.episode_length_buf += 1
                self.official.common_step_counter += 1
                self.official.update_commands()
            # No automatic episode reset: inspection must expose falls and pad failures.
            return None
        if actions is None:
            raise ValueError("Training requires actions")
        previous = self.actions.clone()
        torque = self.advance_physics(actions, training=True, apply_viewer_forces=apply_viewer_forces)
        q, qd, pose, linear, angular, gravity = self._kinematics()
        if not all(torch.isfinite(x).all() for x in (q, qd, pose, linear, angular)):
            raise FloatingPointError(f"{self.backend}: non-finite physics state")
        self.episode_length_buf += 1
        timeout = (self.episode_length_buf >= self.max_episode_length) | self.terrain.out_of_bounds(pose)
        reward, task_terms = self.official.reward(q, qd, pose, linear, angular, gravity, torque, previous)
        support_height = self.terrain.origins[:, 2] if self.scene_terrain else 0.0
        fallen = (pose[:, 2] - support_height < 0.065) | (gravity[:, 2] > -0.3)
        from .events import termination

        failure = termination(self.official, gravity) if self.reset_on_fall else torch.zeros_like(
            self.episode_length_buf, dtype=torch.bool
        )
        if self.mixed_terrain:
            # Preserve the existing Velocity guard without terminating valid seated/fallen task states.
            if self.family == "velocity" and self.reset_on_fall:
                failure |= pose[:, 2] - support_height < 0.065
            failure, timeout = self.terrain.guard.terminations(failure, timeout)
        done = failure | timeout
        terrain_log = self.terrain.log(reward, failure, timeout, linear, gravity, self.commands) if self.mixed_terrain else {}
        self.sim_time += self.step_dt
        # Record pre-reset geometry so simulation diagnostics cannot hide a fall behind auto-reset.
        physics = {"min_trunk_z": pose[:, 2].min().item(), "min_upright_cos": (-gravity[:, 2]).min().item()}
        ids = done.nonzero().flatten()
        if len(ids):
            self.total_resets += len(ids)
            self._reset(ids)
        if self.viewer is None or self.family not in ("velocity", "velstand"):
            self.official.update_commands()
        if apply_interval_events:
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
        extras["log"].update(terrain_log)
        if self.viewer:
            self.render()
        return obs, reward, done, extras

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        state = self.states[0]
        scene_terrain = getattr(self, "scene_terrain", False)
        if self.mixed_terrain or scene_terrain:
            state = self.terrain.display_state(self.viewer, state)
        self.viewer.log_state(state)
        if scene_terrain and not self.mixed_terrain:
            self.terrain.render_particles(self.viewer, state)
        self.viewer.end_frame()

    def close(self):
        if self.viewer:
            self.viewer.close()

    def check_scene_state(self):
        """Strict inspection checks; unlike training, never hide a fall behind a reset."""
        self._sync()
        for name in ("particle_q", "particle_qd", "body_q", "body_qd"):
            if not torch.isfinite(wp.to_torch(getattr(self.states[0], name))).all():
                raise FloatingPointError(f"The sponge scene produced non-finite {name}")
        self.terrain.check_state()
        indices = self.terrain.pad_indices.flatten()
        indices = indices[indices >= 0]
        rest = self.terrain.rest
        fixed = indices[torch.isclose(rest[indices, 2], rest.new_tensor(self.terrain.config.base_z))]
        if not torch.allclose(wp.to_torch(self.states[0].particle_q)[fixed], rest[fixed], atol=1.0e-6, rtol=0.0):
            raise AssertionError("The bonded bottom face moved")
