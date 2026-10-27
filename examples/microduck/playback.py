"""Actor-only playback, isolated from the fingerprinted training recipe."""

from types import SimpleNamespace

import newton
import torch
import warp as wp

from .official_events import termination
from .policy_spec import EFFORT, KD, KP
from .stability import check_physics
from .tasks import rotate


class PolicyPlayback:
    def __init__(self, env):
        self.env = env
        self.observations = env.official.observations
        allowed = {
            "base_ang_vel",
            "projected_gravity",
            "joint_pos",
            "joint_vel",
            "actions",
            "command",
            "head_command",
            "body_command",
        }
        if set(self.observations.spec["actor"]["terms"]) - allowed:
            raise ValueError("Fast playback requires the current proprioceptive actor contract")
        # A skipped critic must not consume RNG or carry delays that influence actor sampling.
        critic = self.observations.spec["critic"]
        if critic["enable_corruption"] or any(group == "critic" for group, _ in self.observations.delays):
            raise ValueError("Fast playback requires a clean, undelayed critic")

    def observe(self):
        env = self.env
        processor = self.observations
        if processor.cache is not None:
            processor.cache[0][:, 48:] = env.commands
            return processor.cache[0]
        q, qd, _, _, angular, gravity = env._kinematics()
        values = {
            "base_ang_vel": angular,
            "projected_gravity": gravity,
            "joint_pos": q[:, env.qi] - env.home,
            "joint_vel": qd[:, env.qdi],
            "actions": env.actions,
            "command": env.commands[:, :3],
            "head_command": env.commands[:, 3:7],
            "body_command": env.commands[:, 7:13],
        }
        cfg = processor.spec["actor"]
        parts = []
        for name, term in cfg["terms"].items():
            value = values[name].clone()
            func = term["func"].rsplit(".", 1)[-1]
            if func == "zero_command_padding":
                value.zero_()
            if func.endswith("imu_misaligned"):
                value = rotate(processor.imu_rotation, value)
            if name == "joint_pos" and term["params"].get("biased", False):
                value += processor.encoder_bias
            noise = term["noise"] if cfg["enable_corruption"] else None
            if noise:
                if noise["operation"] != "add":
                    raise NotImplementedError("Non-additive observation noise")
                value += torch.empty_like(value).uniform_(noise["n_min"], noise["n_max"])
            if term["clip"] is not None:
                value.clamp_(*term["clip"])
            if term["scale"] is not None:
                value *= torch.as_tensor(term["scale"], device=env.device)
            if ("actor", name) in processor.delays:
                value = processor.delays["actor", name](value)
            parts.append(value)
        actor = torch.cat(parts, -1)
        if actor.shape != (env.num_envs, 61) or not torch.isfinite(actor).all():
            raise FloatingPointError("Invalid playback actor observations")
        processor.cache = actor, None
        return actor

    @torch.no_grad()
    def step(self, actions):
        env = self.env
        if env.task not in ("velocity", "velstand", "groundpick", "ballkick"):
            return env.step(actions)[0]
        if actions.shape != env.actions.shape or not torch.isfinite(actions).all():
            raise ValueError("Expected finite actions with shape (num_envs, 14)")
        self.observations.cache = None
        env.actions.copy_(actions.to(env.device))
        desired = env.home + env.actions
        targets = wp.to_torch(env.control.joint_target_q).reshape(env.num_envs, -1)
        for substep in range(env.decimation):
            env._sync()
            newton.eval_ik(env.model, env.states[0], env.states[0].joint_q, env.states[0].joint_qd)
            env._sync()
            q, qd, _, _ = env._views()
            torque = (KP * (desired - q[:, env.qi]) - KD * qd[:, env.qdi]).clamp(-EFFORT, EFFORT)
            targets[:, env.qdi] = q[:, env.qi] + (torque + KD * qd[:, env.qdi]) / KP
            env._sync()
            env.states[0].clear_forces()
            if env.viewer is not None:
                # ViewerGL writes mouse-drag perturbations into the current simulation state.
                env.viewer.apply_forces(env.states[0])
            if env.backend == "synreal":
                env.collision.collide(env.states[0], env.contacts)
            env.solver.step(env.states[0], env.states[1], env.control, env.contacts, env.physics_dt)
            env.states.reverse()
            check_physics(env, substep)
            # Post-solve contacts serve reward/critic sensors only; the next solve rebuilds contacts.
        env._sync()
        newton.eval_ik(env.model, env.states[0], env.states[0].joint_q, env.states[0].joint_qd)
        q, qd, pose, linear, angular, gravity = env._kinematics()
        if not all(torch.isfinite(x).all() for x in (q, qd, pose, linear, angular)):
            raise FloatingPointError(f"{env.backend}: non-finite playback state")
        env.episode_length_buf += 1
        # Training advances the curriculum in reward(), before evaluating termination/events.
        env.official.common_step_counter += 1
        env.official.robot.data = SimpleNamespace(root_link_pos_w=pose[:, :3], root_link_quat_w=pose[:, [6, 3, 4, 5]])
        done = termination(env.official, gravity) | (env.episode_length_buf >= env.max_episode_length)
        env.sim_time += env.step_dt
        ids = done.nonzero().flatten()
        if len(ids):
            env.total_resets += len(ids)
            env._reset(ids)
        if env.viewer is None or env.task == "groundpick":
            env.official.update_commands()
        # env.official.pushes.apply()     #随机推力
        obs = self.observe()
        if env.viewer:
            env.viewer.begin_frame(env.sim_time)
            env.viewer.log_state(env.states[0])
            env.viewer.end_frame()
        return obs
