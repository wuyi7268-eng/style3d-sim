"""Scene assets for the four official-adapted tasks; rewards live in official_recipe.py."""

import math

import torch
import warp as wp

from examples.locomotion.math_utils import quat_rotate_inverse

TASKS = ("velocity", "velstand", "groundpick", "ballkick")


def rotate(quat, vector):
    inverse = torch.cat((-quat[..., :3], quat[..., 3:]), dim=-1)
    return quat_rotate_inverse(inverse, vector)


class TaskLogic:
    def __init__(self, task, robot):
        self.task = task
        self.ball_body = None
        if task == "ballkick":
            self.ball_q = len(robot.joint_q)
            self.ball_qd = len(robot.joint_qd)
            self.ball_body = robot.add_link(
                xform=wp.transform(wp.vec3(0.09, -0.042, 0.035), wp.quat_identity()), label="kick_ball"
            )
            joint = robot.add_joint_free(child=self.ball_body)
            robot.add_articulation([joint], label="ball")
            cfg = robot.default_shape_cfg.copy()
            cfg.density = 0.015 / (4.0 / 3.0 * math.pi * 0.035**3)
            cfg.mu = 0.5
            robot.add_shape_sphere(self.ball_body, radius=0.035, cfg=cfg, color=(1.0, 0.4, 0.05))

    def initialize(self, env):
        self.spawn = env.initial_q.clone()

    def reset_state(self, env, ids, q):
        q[ids] = self.spawn[ids]
