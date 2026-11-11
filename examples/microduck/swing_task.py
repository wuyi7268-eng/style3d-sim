"""Vectorized SwingPump objectives adapted from upstream c5fcc50.

Copyright 2026 Pollen Robotics. Apache-2.0; see assets/swing/LICENSE.
Modified: explicit per-world state/reset and a single update per control step.
"""

import math

import torch


def barrier(value, safe, scale):
    return ((value - safe).clamp_min(0) / scale).square()


class SwingObjective:
    def __init__(self, count, device, config):
        self.config = config
        self.peaks = torch.zeros(count, 2, device=device)
        self.invalid_steps = torch.zeros(count, dtype=torch.long, device=device)
        self.invalid = torch.zeros(count, dtype=torch.bool, device=device)

    def reset(self, ids):
        self.peaks[ids] = 0
        self.invalid_steps[ids] = 0
        self.invalid[ids] = False

    def update(self, state, progress, dt):
        s, cfg = state, self.config
        length = s["lengths"].amax(-1)
        # Keep upstream's full-horizon debt: violations do not restart the episode.
        bad = ((s["lateral"].abs() > 0.020) | (s["alignment"] > 0.050)
               | (length > 0.394) | (length < 0.370))
        self.invalid_steps = torch.where(bad, self.invalid_steps + 1, 0)
        self.invalid |= self.invalid_steps >= 2
        cost = (barrier(s["lateral"].abs(), 0.012, 0.004) + barrier(s["alignment"], 0.04, 0.005)
                + barrier(s["slack"], 0.0095, 0.003) + barrier(length - 0.38, 0.012, 0.001))
        gate = torch.exp(-cost.clamp(max=20)) * (~self.invalid)
        p = cfg["rewards"]["swing_peak_progress"]["params"]
        peaks = torch.stack((s["angle"], -s["angle"]), -1).clamp(0, p["target_angle"])
        paid = self.peaks + (peaks - self.peaks).clamp(0, p["max_paid_rate"] * dt)
        frontier = ((paid / p["target_angle"]).pow(p["frontier_power"])
                    - (self.peaks / p["target_angle"]).pow(p["frontier_power"])).sum(-1) / (2 * dt)
        self.peaks = torch.maximum(self.peaks, peaks)
        height = (1 - s["angle"].cos()).clamp(0, 2)
        severity = torch.maximum((s["lateral"] / 0.020).square(), s["alignment"] / 0.050)
        severity = torch.maximum(severity, barrier(length, 0.392, 0.002)).clamp(max=8)
        terms = {
            "swing_peak_progress": frontier * gate,
            "swing_height": height * gate,
            "swing_late_height": height * progress.pow(cfg["rewards"]["swing_late_height"]["params"]["time_power"]) * gate,
            "swing_energy": (height + 0.5 * (0.38 / 9.81) * s["rate"].square()).clamp(
                max=cfg["rewards"]["swing_energy"]["params"]["max_equivalent_height"]) * gate,
            "swing_lateral": (s["lateral"] / cfg["rewards"]["swing_lateral"]["params"]["tolerance_m"]).square(),
            "swing_lateral_velocity": (s["linear"][:, 1]
                / cfg["rewards"]["swing_lateral_velocity"]["params"]["tolerance_m_s"]).square(),
            "swing_out_of_plane_angular_velocity": s["angular"][:, [0, 2]].square().sum(-1)
                / cfg["rewards"]["swing_out_of_plane_angular_velocity"]["params"]["tolerance_rad_s"] ** 2,
            "swing_alignment": s["alignment"],
            "string_slack": (s["slack"] / 0.01).square() + (s["imbalance"] / 0.01).square(),
            "invalid_episode_debt": self.invalid.float() * (1 + severity),
        }
        p = cfg["rewards"]["swing_lateral_barrier"]["params"]
        terms["swing_lateral_barrier"] = barrier(s["lateral"].abs(), p["safe_m"], p["scale_m"])
        p = cfg["rewards"]["swing_alignment_barrier"]["params"]
        terms["swing_alignment_barrier"] = barrier(s["alignment"], p["safe_penalty"], p["scale"])
        p = cfg["rewards"]["swing_predictive_geometry_barrier"]["params"]
        terms["swing_predictive_geometry_barrier"] = (
            barrier((s["lateral"] + p["horizon_s"] * s["linear"][:, 1]).abs(), p["lateral_safe_m"], p["lateral_scale_m"])
            + barrier(s["predicted_alignment"], p["alignment_safe_penalty"], p["alignment_scale"])
        ).clamp(max=p["max_penalty"])
        p = cfg["rewards"]["string_extension_barrier"]["params"]
        terms["string_extension_barrier"] = barrier(length, p["safe_length_m"], p["scale_m"]).clamp(max=p["max_penalty"])
        return terms

    def weighted(self, terms, step):
        result = {}
        for name, value in terms.items():
            weight = self.config["rewards"][name]["weight"]
            for stage in self.config["curriculum"].get(name, []):
                if step > stage["step"]:
                    weight = stage["weight"]
            result[name] = value * weight
        return result

    def critic_context(self, state, progress):
        s = state
        fault = torch.maximum(s["slack"] / 0.01, s["imbalance"] / 0.01)
        fault = torch.maximum(fault, (s["lengths"].amax(-1) - 0.392).clamp_min(0) / 0.002)
        validity = torch.stack((s["lateral"] / 0.02, s["linear"][:, 1] / 0.2,
                                s["angular"][:, 0] / 2, s["angular"][:, 2] / 2, s["alignment"] / 0.05,
                                torch.maximum(fault, 2 * self.invalid.float())), -1)
        frontier = torch.cat((self.peaks / math.pi, progress[:, None]), -1)
        mechanism = torch.stack((s["angle"] / math.pi, s["rate"] / 8, s["lateral"] / 0.38, s["slack"] / 0.02), -1)
        return frontier, validity, mechanism
