"""Ball-relative rewards and failure criteria shared by both local solvers.

Basketball-specific formulas: Vottivott/microduck-playground, Apache-2.0,
Copyright 2026 Hannes von Essen / Vottivott.
052a2494dd6372289221c7a4f24a245b443dd75f. No ball features enter the actor.
"""

import math

import torch


def task_terms(config, root, gravity, linear, angular, ball, ball_velocity, feet, commands):
    rewards, geometry = config["rewards"], config["geometry"]
    offset = root[:, :2] - ball[:, :2]
    height = root[:, 2] - ball[:, 2]
    foot_gap = (feet - ball[:, None]).norm(dim=-1) - geometry["radius"] - rewards["feet_on_ball"]["sole_gap"]
    terms = {
        "track_linear": torch.exp(-(linear[:, :2] - commands[:, :2]).square().sum(-1)
                                   / rewards["track_linear"]["variance"]),
        "track_angular": torch.exp(-(angular[:, 2] - commands[:, 2]).square() / rewards["track_angular"]["variance"]),
        "upright": torch.exp(-gravity[:, :2].square().sum(-1) / rewards["upright"]["std"]**2),
        "centered": torch.exp(-offset.square().sum(-1) / rewards["centered"]["std"]**2),
        "feet_on_ball": torch.exp(-foot_gap.square() / rewards["feet_on_ball"]["std"]**2).mean(-1),
        "height": torch.exp(-(height - geometry["radius"] - geometry["root_height"]).square()
                            / rewards["height"]["std"]**2),
        "ball_speed": ball_velocity[:, :2].square().sum(-1),
        "body_ang_vel": angular[:, :2].square().sum(-1),
    }
    end = config["termination"]
    failed = ((height < geometry["radius"] + end["min_height"])
              | (offset.norm(dim=-1) > end["max_offset"])
              | (-gravity[:, 2] < math.cos(math.radians(end["tilt_deg"]))))
    metrics = {"offset": offset.norm(dim=-1), "height": height, "upright": -gravity[:, 2],
               "ball_position": ball, "ball_velocity": ball_velocity, "failed": failed}
    return terms, failed, metrics


def advance_hold(level, durations, config):
    promoted = level + (durations >= config["promote_s"]).long() - (durations < config["demote_s"]).long()
    return promoted.clamp(0, len(config["levels"]) - 1)
