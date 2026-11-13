"""Running-only inertial randomization and the released robustness curriculum."""

import math
import re

import newton
import torch
import warp as wp


def robustness_stage(spec, control_steps):
    config = spec["robustness"]
    iteration = control_steps // spec["rl"]["num_steps_per_env"]
    elapsed = iteration - config["start_iteration"]
    if elapsed < 0:
        return 0, config["discovery"]
    for index, stage in enumerate(config["stages"], 1):
        if elapsed < stage["updates"]:
            return index, stage
        elapsed -= stage["updates"]
    # Additional training stays at the final difficulty; it never restarts the ramp.
    return len(config["stages"]), config["stages"][-1]


def train_stages(runner, env, iterations):
    """Keep PPO state while resetting episodes at each fixed robustness boundary."""
    rollout = env.official.spec["rl"]["num_steps_per_env"]
    if runner.num_steps_per_env != rollout:
        raise ValueError("Running's iteration-based curriculum requires 24 rollout steps")
    config = env.official.spec["robustness"]
    boundary = config["start_iteration"]
    boundaries = [boundary]
    for stage in config["stages"]:
        boundary += stage["updates"]
        boundaries.append(boundary)
    current = runner.current_learning_iteration
    end = current + iterations
    while current < end:
        stop = min([end, *(value for value in boundaries if value > current)])
        # RSL-RL stores the last completed index. Repeated learn calls otherwise
        # repeat that update and shift both speed and robustness curricula.
        runner.current_learning_iteration = current
        env.official.common_step_counter = current * rollout
        # Observation delay caches produced by PPO are inference tensors.
        with torch.inference_mode():
            env.reset()
        processor = env.official.observations
        processor.cache = tuple(value.clone() for value in processor.cache)
        stage, parameters = robustness_stage(env.official.spec, current * rollout)
        print(f"Running phase {stage}: iterations [{current}, {stop}), {parameters}", flush=True)
        runner.learn(stop - current, init_at_random_ep_len=False)
        current = stop


class RunningRandomization:
    def __init__(self, env, spec):
        self.env, self.spec = env, spec
        self.enabled = env.reset_on_fall and spec.get("running_randomization_enabled", True)
        self.disturbances_enabled = env.reset_on_fall and spec.get(
            "running_disturbances_enabled", spec.get("running_randomization_enabled", True)
        )
        model = env.model
        self.mass = wp.to_torch(model.body_mass).reshape(env.num_envs, -1)
        self.inertia = wp.to_torch(model.body_inertia).reshape(env.num_envs, -1, 3, 3)
        self.com = wp.to_torch(model.body_com).reshape(env.num_envs, -1, 3)
        self.armature = wp.to_torch(model.joint_armature).reshape(env.num_envs, -1)
        self.nominal_mass, self.nominal_inertia = self.mass.clone(), self.inertia.clone()
        self.nominal_com, self.nominal_armature = self.com.clone(), self.armature.clone()
        names = [name.rsplit("/", 1)[-1] for name in model.body_label[:self.mass.shape[1]]]
        self.trunk = names.index("trunk_base")
        patterns = spec["events"]["randomize_head_com"]["params"]["asset_cfg"]["body_names"]
        # Preserve upstream's authored selectors, including bearing_roll (a hip link).
        self.head = [i for i, name in enumerate(names) if any(re.fullmatch(p, name) for p in patterns)]
        if not self.head:
            raise ValueError("Running head COM selectors matched no bodies")
        if self.enabled:
            # Alpha is a log-density perturbation of the Cholesky factor, so
            # both mass and inertia scale by exp(2*alpha), not uniform(0.95,1.05).
            bounds = spec["events"]["randomize_mass_inertia"]["params"]["alpha_range"]
            scale = torch.empty(env.num_envs, device=env.device).uniform_(*bounds).mul_(2).exp_()
            self.mass[:, self.trunk] = self.nominal_mass[:, self.trunk] * scale
            self.inertia[:, self.trunk] = self.nominal_inertia[:, self.trunk] * scale[:, None, None]
            wp.to_torch(model.body_inv_mass).reshape_as(self.mass)[:, self.trunk] = self.mass[:, self.trunk].reciprocal()
            wp.to_torch(model.body_inv_inertia).reshape_as(self.inertia)[:, self.trunk] = torch.linalg.inv(
                self.inertia[:, self.trunk]
            )
            env._sync()
        env.cfg["running_randomization"] = {
            "enabled": self.enabled, "disturbances_enabled": self.disturbances_enabled,
            "mass_inertia": "startup exp(2*uniform(alpha_range)) on trunk",
            "com_armature": "reset from immutable nominal values; selected worlds only",
            "robustness": spec["robustness"], "head_body_names": [names[i] for i in self.head],
        }

    def reset(self, ids, recipe, q):
        if not (self.enabled or self.disturbances_enabled) or not len(ids):
            return
        env = self.env
        _, stage = robustness_stage(self.spec, recipe.common_step_counter)
        tilt = math.radians(stage["initial_tilt_deg"])
        if self.disturbances_enabled and tilt:
            # Upstream randomize_base_orientation replaces yaw with zero; it
            # does not compose the tilt with reset_base's randomized yaw.
            pitch, roll = torch.empty(2, len(ids), device=env.device).uniform_(-tilt, tilt) / 2
            q[ids, 3:7] = torch.stack((roll.sin() * pitch.cos(), roll.cos() * pitch.sin(),
                                      -roll.sin() * pitch.sin(), roll.cos() * pitch.cos()), -1)
        if not self.enabled:
            return
        # Overwrite from the original model each time, never add to the previous draw.
        self.com[ids, self.trunk] = self.nominal_com[ids, self.trunk] + torch.empty(
            len(ids), 3, device=env.device
        ).uniform_(-stage["trunk_com_m"], stage["trunk_com_m"])
        head = torch.tensor(self.head, device=env.device)
        self.com[ids[:, None], head] = self.nominal_com[ids[:, None], head] + torch.empty(
            len(ids), len(head), 3, device=env.device
        ).uniform_(-stage["head_com_m"], stage["head_com_m"])
        bounds = self.spec["events"]["randomize_armature"]["params"]["ranges"]
        self.armature[ids[:, None], env.qdi] = self.nominal_armature[ids[:, None], env.qdi] * torch.empty(
            len(ids), 14, device=env.device
        ).uniform_(*bounds)
        env._sync()
        # COM changes require eager proxy/constraint-weight reconstruction, not
        # just edits to the public model arrays. Do this before selected-state FK.
        env.solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES | newton.ModelFlags.JOINT_DOF_PROPERTIES)
        if env.backend == "synreal":
            env.solver.prepare_contacts(env.contacts)
        env._sync()

    def push_range(self, recipe):
        _, stage = robustness_stage(self.spec, recipe.common_step_counter)
        magnitude = stage["push_m_s"] if self.disturbances_enabled else 0.0
        return {axis: (-magnitude, magnitude) for axis in ("x", "y")}
