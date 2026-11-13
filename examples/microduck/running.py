"""Flat forward-running objectives from microduck-playground@828d950 (Apache-2.0).

The frozen training recipe lives in configs/running.json. No walking or bridge
settings are inherited at runtime. Physics, observations and PPO remain shared.
"""

import torch


def running_forward_progress(recipe, speed_cap):
    if speed_cap <= 0:
        raise ValueError("speed_cap must be positive")
    speed = recipe.robot.data.root_link_lin_vel_b[:, 0]
    recipe.extras["log"]["Metrics/running_forward_speed"] = speed.mean()
    if "robustness" in getattr(recipe, "spec", {}):
        from .running_randomization import robustness_stage

        stage, parameters = robustness_stage(recipe.spec, recipe.common_step_counter)
        recipe.extras["log"]["Curriculum/running_robustness_stage"] = float(stage)
        recipe.extras["log"]["Curriculum/running_push_m_s"] = (
            parameters["push_m_s"] if recipe.spec.get(
                "running_disturbances_enabled", recipe.spec.get("running_randomization_enabled", True)
            ) else 0.0
        )
    speed = torch.nan_to_num(speed, nan=0.0, posinf=speed_cap, neginf=0.0)
    # Upstream deliberately rewards exceeding the command, including the idle bucket.
    return speed.clamp(0.0, speed_cap) / speed_cap


def running_planar_drift_cost(recipe, command_name="twist", lateral_weight=4.0):
    command = recipe.get_command(command_name)
    data = recipe.robot.data
    lateral = torch.nan_to_num(data.root_link_lin_vel_b[:, 1] - command[:, 1], nan=0.0)
    yaw = torch.nan_to_num(data.root_link_ang_vel_b[:, 2] - command[:, 2], nan=0.0)
    return yaw.square() + lateral_weight * lateral.square()


def command_ranges(recipe):
    """Use upstream's inclusive speed stages without changing shared weight schedules."""
    stages = recipe.spec["curricula"]["running_speed_range"]["params"]["speed_stages"]
    stage = stages[0]
    for candidate in stages:
        if recipe.common_step_counter >= candidate["step"]:
            stage = candidate
    return {**recipe.spec["commands"]["twist"]["ranges"], "lin_vel_x": [stage["min_speed"], stage["max_speed"]]}


REWARDS = {
    "running_forward_progress": running_forward_progress,
    "running_planar_drift_cost": running_planar_drift_cost,
}
