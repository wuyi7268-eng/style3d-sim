"""Pose-feedback commands and terminal accounting for an inlet-to-outlet bridge crossing."""

import math

import torch


class BridgePath:
    def __init__(self, recipe, config):
        self.recipe, self.config = recipe, config
        env = recipe.env
        scene = env.terrain.cfg["config"]
        self.half_length = scene["elastic_bridge_length"] / 2
        self.goal_x = self.half_length + scene["elastic_bridge_approach_length"] / 2
        self.speed = torch.zeros(env.num_envs, device=env.device)
        self.previous_x = torch.zeros_like(self.speed)
        self.hold = torch.zeros_like(self.speed)
        self.completed = torch.zeros_like(self.speed, dtype=torch.bool)
        self.at_goal = torch.zeros_like(self.completed)

    def reset(self, ids):
        lo, hi = self.config["speed_range"]
        self.speed[ids] = torch.empty(len(ids), device=self.speed.device).uniform_(lo, hi)
        self.previous_x[ids] = self.recipe.env._views()[2][ids, 0]
        self.hold[ids] = 0
        self.completed[ids] = False
        self.at_goal[ids] = False

    def update(self, ids=None):
        env, cfg = self.recipe.env, self.config
        ids = torch.arange(env.num_envs, device=env.device) if ids is None else ids
        pose = env._views()[2][ids]
        x, y, z, w = pose[:, 3:7].unbind(-1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
        heading = torch.atan2(-pose[:, 1], torch.full_like(yaw, cfg["lookahead_m"]))
        error = torch.atan2((heading - yaw).sin(), (heading - yaw).cos())
        # Translate a world-frame centerline velocity into the policy's body-frame commands.
        # Goal distance does not taper walking speed; only the goal-region stop below requests zero.
        world_x = self.speed[ids] * error.cos().clamp_min(0)
        world_y = (-cfg["lateral_gain"] * pose[:, 1]).clamp(-cfg["lateral_limit"], cfg["lateral_limit"])
        commands = torch.stack((yaw.cos() * world_x + yaw.sin() * world_y,
                                -yaw.sin() * world_x + yaw.cos() * world_y,
                                (cfg["yaw_gain"] * error).clamp(-cfg["yaw_limit"], cfg["yaw_limit"])), -1)
        # Apply the floor after rotation so it bounds the actual body-forward policy input.
        commands[:, 0].clamp_(min=cfg["minimum_forward_speed"])
        at_goal = torch.linalg.vector_norm(pose[:, :2] - pose.new_tensor([self.goal_x, 0]), dim=-1)
        at_goal = at_goal <= cfg["goal_tolerance_m"]
        at_goal &= yaw.abs() <= math.radians(cfg["goal_heading_deg"])
        self.at_goal[ids] = at_goal
        commands[at_goal | self.completed[ids]] = 0
        env.commands[ids, :3] = commands
        self.recipe.standing[ids] = at_goal | self.completed[ids]

    def evaluate(self, pose, linear, gravity, failure):
        """Return step reward and true-terminal success before any automatic reset."""
        env, cfg = self.recipe.env, self.config
        data = self.recipe.robot.data
        upright = -gravity[:, 2] >= math.cos(math.radians(cfg["max_tilt_deg"]))
        feet = data.site_pos_w[:, 1:3]
        on_outlet = (feet[:, :, 0] > self.half_length + cfg["foot_margin_m"]).all(-1)
        contact = (self.recipe.sensors.feet.data.found[:, :2] > 0).all(-1)
        # Test current geometry, not the previous observation's goal flag.
        distance = torch.linalg.vector_norm(pose[:, :2] - pose.new_tensor([self.goal_x, 0]), dim=-1)
        qx, qy, qz, qw = pose[:, 3:7].unbind(-1)
        yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy.square() + qz.square()))
        stable = (distance <= cfg["goal_tolerance_m"]) & (yaw.abs() <= math.radians(cfg["goal_heading_deg"]))
        stable &= on_outlet & contact & upright & (linear.norm(dim=-1) <= cfg["stop_speed_m_s"]) & ~failure
        self.hold.copy_(torch.where(stable, self.hold + env.step_dt, 0.0))
        success = (self.hold >= cfg["hold_s"]) & ~self.completed & ~failure
        self.completed |= success
        # Signed progress prevents back-and-forth motion from accumulating free distance reward.
        dx = (pose[:, 0].clamp(max=self.goal_x) - self.previous_x.clamp(max=self.goal_x)).clamp(-.05, .05)
        self.previous_x.copy_(pose[:, 0])
        progress = torch.where(upright & ~failure, dx, dx.clamp(max=0))
        center_cost = (pose[:, 1] / cfg["center_scale_m"]).square().clamp(max=4)
        reward = cfg["progress_weight"] * progress - cfg["center_weight"] * center_cost * env.step_dt
        reward += cfg["success_bonus"] * success - cfg["failure_cost"] * failure
        log = {"Bridge/success_fraction": success.float().mean(), "Bridge/failure_fraction": failure.float().mean(),
               "Bridge/cross_track_m": pose[:, 1].abs().mean(), "Bridge/goal_distance_m": distance.mean(),
               "Bridge/forward_command": env.commands[:, 0].mean()}
        return reward, success, log
