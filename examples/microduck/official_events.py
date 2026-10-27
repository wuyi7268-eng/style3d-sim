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
    def write_joint_position_to_sim(self, pos, env_ids, joint_ids=None):
        cols = (
            torch.arange(len(self.joint_names), device=pos.device)
            if joint_ids is None
            else torch.as_tensor(joint_ids, device=pos.device)
        )
        self.qpos[env_ids[:, None], 7 + cols[None, :]] = pos

    def write_joint_velocity_to_sim(self, vel, env_ids, joint_ids=None):
        cols = (
            torch.arange(len(self.joint_names), device=vel.device)
            if joint_ids is None
            else torch.as_tensor(joint_ids, device=vel.device)
        )
        self.qvel[env_ids[:, None], 6 + cols[None, :]] = vel

    def write_joint_state_to_sim(self, pos, vel, env_ids):
        self.write_joint_position_to_sim(pos, env_ids)
        self.write_joint_velocity_to_sim(vel, env_ids)

    def write_root_link_velocity_to_sim(self, vel, env_ids):
        from .official_adapter import quat_apply_inverse

        self.qvel[env_ids, :3] = vel[:, :3]
        self.qvel[env_ids, 3:6] = quat_apply_inverse(self.qpos[env_ids, 3:7], vel[:, 3:])


def prepare_reset(recipe, ids):
    """Run upstream reset formulas in a canonical servo-order view before backend FK."""
    env = recipe.env
    q = env.initial_q.clone()
    count = env.num_envs
    robot = ResetRobot()
    qi = getattr(env, "all_qi", env.qi)
    robot.joint_names = getattr(env, "joint_names", robot.joint_names)
    robot.qpos = torch.cat((q[:, :3], q[:, [6, 3, 4, 5]], q[:, qi]), -1)
    robot.qvel = torch.zeros(count, 6 + len(qi), device=env.device)
    robot.data = SimpleNamespace(
        joint_pos=robot.qpos[:, 7:],
        joint_vel=robot.qvel[:, 6:],
        default_joint_pos=q[:, qi],
        indexing=SimpleNamespace(free_joint_q_adr=torch.arange(7)),
        data=SimpleNamespace(qpos=robot.qpos),
        joint_pos_limits=recipe.limit.expand(count, -1, -1),
    )
    proxy = SimpleNamespace(
        num_envs=count,
        common_step_counter=recipe.common_step_counter,
        device=env.device,
        scene=SimpleScene(robot, recipe.scene.env_origins),
        sim=SimpleNamespace(data=SimpleNamespace(qpos=robot.qpos, qvel=robot.qvel)),
    )
    bounds = recipe.spec["events"]["reset_base"]["params"]["pose_range"]
    # The upstream default root position is (0, 0, 0), unlike the local XML standing root.
    for column, axis in enumerate(("x", "y", "z")):
        robot.qpos[ids, column] = torch.empty(len(ids), device=env.device).uniform_(*bounds.get(axis, (0.0, 0.0)))
    yaw = torch.empty(len(ids), device=env.device).uniform_(*bounds.get("yaw", (0.0, 0.0)))
    roll = torch.empty(len(ids), device=env.device).uniform_(*bounds.get("roll", (0.0, 0.0)))
    pitch = torch.empty(len(ids), device=env.device).uniform_(*bounds.get("pitch", (0.0, 0.0)))
    cr, sr = (roll / 2).cos(), (roll / 2).sin()
    cp, sp = (pitch / 2).cos(), (pitch / 2).sin()
    cy, sy = (yaw / 2).cos(), (yaw / 2).sin()
    robot.qpos[ids, 3:7] = torch.stack(
        (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy), -1
    )
    for term in recipe.spec["events"].values():
        if term["func"].endswith(".reset_joints_by_offset"):
            lo, hi = term["params"]["position_range"]
            joints = env.home + torch.empty(len(ids), 14, device=env.device).uniform_(lo, hi)
            mid = recipe.limit.mean(-1)
            half = (recipe.limit[:, 1] - recipe.limit[:, 0]) * 0.45
            robot.qpos[ids, 7:21] = joints.clamp(mid - half, mid + half)
    robot.qpos[ids, :3] += recipe.scene.env_origins[ids]
    velocity_ranges = recipe.spec["events"]["reset_base"]["params"].get("velocity_range", {})
    for column, axis in enumerate(("x", "y", "z", "roll", "pitch", "yaw")):
        robot.qvel[ids, column] = torch.empty(len(ids), device=env.device).uniform_(*velocity_ranges.get(axis, (0.0, 0.0)))
    # Keep episode-local roll progress on the reward facade, not on a disposable reset proxy.
    for key, value in vars(recipe).items():
        if key.startswith("_roulade_"):
            setattr(proxy, key, value)
    reset_functions = {
        "maybe_set_random_prone_orientation",
        "set_random_ground_state",
        "reset_roulade_state",
        "reset_rolling_entry",
        "reset_with_forward_velocity",
    }
    from .official_recipe import OfficialRecipe

    for name, term in recipe.spec["events"].items():
        function = term["func"].rsplit(".", 1)[-1]
        if function in reset_functions:
            params = scheduled_params(recipe, name, "events")

            # JSON object keys are strings, while upstream pose overrides index joints by integer.
            def numeric_keys(value):
                if isinstance(value, dict):
                    return {int(k) if k.lstrip("-").isdigit() else k: numeric_keys(v) for k, v in value.items()}
                return value

            params = numeric_keys(params)
            if "asset_cfg" in params:
                params["asset_cfg"] = OfficialRecipe.decode(recipe, params["asset_cfg"])
            getattr(upstream, function)(proxy, ids, **params)
    for key, value in vars(proxy).items():
        if key.startswith("_roulade_"):
            setattr(recipe, key, value)
    q[ids, :3] = robot.qpos[ids, :3]
    q[ids, 3:7] = robot.qpos[ids][:, [4, 5, 6, 3]]
    q[ids[:, None], qi[None, :]] = robot.qpos[ids, 7:]
    if env.task.startswith("ballkick"):
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
    if hasattr(env.task_logic, "spawn_velocity"):
        from .tasks import rotate

        velocity = env.task_logic.spawn_velocity
        velocity[ids] = 0.0
        velocity[ids, :3] = robot.qvel[ids, :3]
        velocity[ids, 3:6] = rotate(q[ids, 3:7], robot.qvel[ids, 3:6])
        velocity[ids[:, None], env.all_qdi[None, :]] = robot.qvel[ids, 6:]


class SimpleScene(dict):
    def __init__(self, robot, origins):
        super().__init__(robot=robot)
        self.env_origins = origins


def termination(recipe, gravity):
    failed = torch.zeros(gravity.shape[0], dtype=torch.bool, device=gravity.device)
    if "fell_over" in recipe.spec["terminations"]:
        params = scheduled_params(recipe, "fell_over", "terminations")
        failed = torch.acos((-gravity[:, 2]).clamp(-1, 1)) > params["limit_angle"]
    if "fell_into_void" in recipe.spec["terminations"]:
        params = recipe.spec["terminations"]["fell_into_void"]["params"]
        failed |= recipe.robot.data.root_link_pos_w[:, 2] < params["min_height"]
    if "fallen_too_long" in recipe.spec["terminations"]:
        params = recipe.spec["terminations"]["fallen_too_long"]["params"]
        failed |= upstream.fallen_too_long(recipe, **params)
    return failed


class PushEvents:
    """Per-environment interval pushes; upstream adds velocity despite the function's name."""

    def __init__(self, recipe):
        self.recipe = recipe
        self.events = {name: cfg for name, cfg in recipe.spec["events"].items() if cfg["mode"] == "interval"}
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
            self.timers[name][ids] = torch.empty(len(ids), device=env.device).uniform_(*cfg["interval_range_s"])
            function = cfg["func"].rsplit(".", 1)[-1]
            if function != "push_by_setting_velocity":
                changed |= self.seated_event(name, function, ids)
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
            newton.eval_fk(env.model, env.states[0].joint_q, env.states[0].joint_qd, env.states[0])
            if env.backend == "synreal":
                env.solver.sync_affine_body_state(env.states[0])
            env._sync()

    def seated_event(self, name, function, ids):
        from .tasks import rotate

        recipe, env = self.recipe, self.recipe.env
        params = scheduled_params(recipe, name, "events")
        seated = (env.commands[ids, 0] > 0.5) & (recipe.posture_alpha[ids] >= params.get("min_alpha", 0.98))
        ids = ids[seated & (torch.rand(len(ids), device=env.device) < params.get("prob", 1.0))]
        if not len(ids):
            return False
        if function == "seated_head_command_jerk":
            command = recipe.spec["commands"]["head_pose"]
            ranges = command["ranges"]
            for course in recipe.spec["curricula"].values():
                p = course["params"]
                if course["func"].endswith(".pose_command_range_curriculum") and p["command_name"] == "head_pose":
                    ranges = recipe.stage(p["range_stages"])["ranges"]
            bounds = torch.tensor(ranges, device=env.device)
            fresh = bounds[:, 0] + torch.rand(len(ids), 4, device=env.device) * (bounds[:, 1] - bounds[:, 0])
            flip = torch.rand(len(ids), device=env.device) < params.get("flip_frac", 0.5)
            env.commands[ids, 3:7] = torch.where(flip[:, None], -env.commands[ids, 3:7], fresh)
            return False
        if function != "seated_backward_tip":
            raise NotImplementedError(f"Unsupported interval event: {function}")
        lo, hi = params["ang_vel_range"]
        if hi <= 0:
            return False
        magnitude = torch.empty(len(ids), device=env.device).uniform_(lo, hi)
        lateral = (torch.rand(len(ids), device=env.device) * 2 - 1) * params.get("lateral_frac", 0.3)
        local = torch.stack((lateral * magnitude, -magnitude, torch.zeros_like(magnitude)), -1)
        _, velocity, pose, _ = env._views()
        omega = rotate(pose[ids, 3:], local)
        velocity[ids, 3:6] += omega
        # Preserve root-origin linear velocity when changing angular COM velocity.
        import warp as wp

        com = wp.to_torch(env.model.body_com)[0].expand(len(ids), -1)
        velocity[ids, :3] += torch.cross(omega, rotate(pose[ids, 3:], com), dim=-1)
        return True
