"""Reset distributions and termination curricula for the flat official task recipes."""

import copy
from types import SimpleNamespace

import torch

from . import official_reset_functions as upstream
from .official_recipe import RobotView


def scheduled_params(recipe, name, kind):
    params = copy.deepcopy(recipe.spec[kind][name]["params"])
    function = "event_param_curriculum" if kind == "events" else "termination_param_curriculum"
    key = "event_name" if kind == "events" else "term_name"
    for cfg in recipe.spec["curricula"].values():
        if cfg["func"].endswith("." + function) and cfg["params"][key] == name:
            for stage in cfg["params"]["param_stages"]:
                if recipe.common_step_counter >= stage["step"]:
                    params.update(copy.deepcopy(stage["params"]))
    return params


class ResetRobot(RobotView):
    def write_joint_position_to_sim(self, pos, env_ids):
        self.qpos[env_ids, 7:] = pos

    def write_joint_velocity_to_sim(self, vel, env_ids):
        self.qvel[env_ids, 6:] = vel


def prepare_reset(recipe, ids):
    """Run upstream reset formulas in a canonical servo-order view before backend FK."""
    env = recipe.env
    q = env.initial_q.clone()
    count = env.num_envs
    robot = ResetRobot()
    robot.qpos = torch.cat((q[:, :3], q[:, [6, 3, 4, 5]], q[:, env.qi]), -1)
    robot.qvel = torch.zeros(count, 20, device=env.device)
    robot.data = SimpleNamespace(
        joint_pos=robot.qpos[:, 7:],
        default_joint_pos=env.home.expand(count, -1),
        joint_pos_limits=recipe.limit.expand(count, -1, -1),
    )
    proxy = SimpleNamespace(
        num_envs=count,
        device=env.device,
        scene=SimpleScene(robot, recipe.scene.env_origins),
        sim=SimpleNamespace(data=SimpleNamespace(qpos=robot.qpos, qvel=robot.qvel)),
    )
    bounds = recipe.spec["events"]["reset_base"]["params"]["pose_range"]
    # The upstream default root position is (0, 0, 0), unlike the local XML standing root.
    for column, axis in enumerate(("x", "y", "z")):
        robot.qpos[ids, column] = torch.empty(len(ids), device=env.device).uniform_(*bounds.get(axis, (0.0, 0.0)))
    yaw = torch.empty(len(ids), device=env.device).uniform_(*bounds["yaw"])
    robot.qpos[ids, 3:7] = torch.stack((torch.cos(yaw / 2), yaw * 0, yaw * 0, torch.sin(yaw / 2)), -1)
    for term in recipe.spec["events"].values():
        if term["func"].endswith(".reset_joints_by_offset"):
            lo, hi = term["params"]["position_range"]
            joints = env.home + torch.empty(len(ids), 14, device=env.device).uniform_(lo, hi)
            mid = recipe.limit.mean(-1)
            half = (recipe.limit[:, 1] - recipe.limit[:, 0]) * 0.45
            robot.qpos[ids, 7:] = joints.clamp(mid - half, mid + half)
    if "random_prone_init" in recipe.spec["events"]:
        params = scheduled_params(recipe, "random_prone_init", "events")
        if "sit_params" in params:
            overrides = params["sit_params"].get("sitting_joint_overrides", {})
            params["sit_params"]["sitting_joint_overrides"] = {int(k): v for k, v in overrides.items()}
        upstream.maybe_set_random_prone_orientation(proxy, ids, **params)
    if "set_ground_state" in recipe.spec["events"]:
        upstream.set_random_ground_state(proxy, ids, **scheduled_params(recipe, "set_ground_state", "events"))
    q[ids, :3] = robot.qpos[ids, :3]
    q[ids, 3:7] = robot.qpos[ids][:, [4, 5, 6, 3]]
    q[ids[:, None], env.qi[None, :]] = robot.qpos[ids, 7:]
    if env.task == "ballkick":
        p = recipe.spec["events"]["reset_ball"]["params"]
        quat = q[ids, 3:7]
        yaw = torch.atan2(
            2 * (quat[:, 3] * quat[:, 2] + quat[:, 0] * quat[:, 1]), 1 - 2 * (quat[:, 1].square() + quat[:, 2].square())
        )
        offset = torch.tensor(p["offset"], device=env.device) + torch.empty(len(ids), 2, device=env.device).uniform_(
            -p["noise_xy"], p["noise_xy"]
        )
        c, s = yaw.cos(), yaw.sin()
        rotated = torch.stack((c * offset[:, 0] - s * offset[:, 1], s * offset[:, 0] + c * offset[:, 1]), -1)
        start = env.task_logic.ball_q
        q[ids, start : start + 2] = q[ids, :2] + rotated
        q[ids, start + 2] = p["ball_radius"]
        q[ids, start + 3 : start + 7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device)
        if not hasattr(recipe, "_ball_kick_dir_w"):
            recipe._ball_kick_dir_w = torch.zeros(count, 2, device=env.device)
        recipe._ball_kick_dir_w[ids] = torch.stack((c, s), -1)
    env.task_logic.spawn[ids] = q[ids]


class SimpleScene(dict):
    def __init__(self, robot, origins):
        super().__init__(robot=robot)
        self.env_origins = origins


def termination(recipe, gravity):
    params = scheduled_params(recipe, "fell_over", "terminations")
    failed = torch.acos((-gravity[:, 2]).clamp(-1, 1)) > params["limit_angle"]
    if "fallen_too_long" in recipe.spec["terminations"]:
        params = recipe.spec["terminations"]["fallen_too_long"]["params"]
        failed |= upstream.fallen_too_long(recipe, **params)
    return failed


class PushEvents:
    """Per-environment interval pushes; upstream adds velocity despite the function's name."""

    def __init__(self, recipe):
        self.recipe = recipe
        self.events = {
            name: cfg
            for name, cfg in recipe.spec["events"].items()
            if cfg["mode"] == "interval" and cfg["func"].endswith(".push_by_setting_velocity")
        }
        self.timers = {}
        for name, cfg in self.events.items():
            self.timers[name] = torch.empty(recipe.num_envs, device=recipe.device).uniform_(*cfg["interval_range_s"])

    def apply(self):
        import newton

        recipe = self.recipe
        env = recipe.env
        changed = False
        for name, cfg in self.events.items():
            self.timers[name] -= env.step_dt
            ids = (self.timers[name] <= 0).nonzero().flatten()
            if not len(ids):
                continue
            ranges = cfg["params"]["velocity_range"]
            for course in recipe.spec["curricula"].values():
                p = course["params"]
                if course["func"].endswith(".push_curriculum") and p["event_name"] == name:
                    for stage in p["push_stages"]:
                        if recipe.common_step_counter >= stage["step"]:
                            ranges = stage["velocity_range"]
            _, qd, _, _ = env._views()
            for column, axis in enumerate(("x", "y")):
                qd[ids, column] += torch.empty(len(ids), device=env.device).uniform_(*ranges.get(axis, (0.0, 0.0)))
            if any(key not in ("x", "y") for key in ranges):
                raise NotImplementedError("Push adapter currently supports the snapshot's planar linear impulses")
            self.timers[name][ids] = torch.empty(len(ids), device=env.device).uniform_(*cfg["interval_range_s"])
            changed = True
        if changed:
            env._sync()
            # Root free-joint linear velocity is world-frame COM velocity. With unchanged angular
            # velocity, adding a world linear impulse is identical at COM and at the link origin.
            newton.eval_fk(env.model, env.states[0].joint_q, env.states[0].joint_qd, env.states[0])
            if env.backend == "synreal":
                env.solver.sync_affine_body_state(env.states[0])
            env._sync()
