"""Continuous rebound events and hardware-observable input; independent of legacy jump recipes."""

import math

import torch

from examples.locomotion.math_utils import quat_rotate_inverse

SENSOR_DIM = 51


def specific_force(previous_velocity, velocity, quat, dt):
    """Average IMU-point specific force in trunk axes, including gravity at rest."""
    acceleration = (velocity - previous_velocity) / dt
    acceleration = acceleration.clone()
    acceleration[:, 2] += 9.81
    return quat_rotate_inverse(quat, acceleration)


def sensor_frame(state, acceleration, previous_action):
    # No contact, bed state, world velocity, height, or oracle phase may enter the actor.
    return torch.cat((state.angular * 0.2, state.gravity, acceleration / 9.81,
                      state.q, state.qd * 0.05, previous_action), -1).clamp(-10, 10)


class ReboundCycle:
    """Reward bookkeeping only: the policy never receives this oracle state machine."""

    shaped = True

    def __init__(self, count, device, settings):
        self.cfg = settings
        self.phase = torch.zeros(count, dtype=torch.long, device=device)
        self.failed = torch.zeros(count, dtype=torch.bool, device=device)
        self.confirmed = self.failed.clone()
        self.apex_seen = self.failed.clone()
        self.airborne_paid = self.failed.clone()
        self.touch_time = torch.full((count, 2), -1.0, device=device)
        for name in ("target", "reference", "peak", "age", "phase_age", "stable_time", "air_time",
                     "landing_age", "compression", "score", "crouch", "push_progress", "height_paid",
                     "launch_progress", "contact_time", "flight_time", "max_clearance", "last_height",
                     "previous_height", "bounce_count", "contact_to_takeoff", "support_leg", "support_bed", "stall_age"):
            setattr(self, name, torch.zeros(count, device=device))

    def reset(self, ids, targets, com_z):
        for value in vars(self).values():
            if isinstance(value, torch.Tensor):
                value[ids] = 0
        self.touch_time[ids] = -1
        # Lock one reference for the whole episode, so crouching cannot lower the height target.
        self.reference[ids] = com_z
        self.peak[ids] = com_z

    def advance(self, s, dt):
        cfg = self.cfg
        active = ~self.failed
        phase = self.phase.clone()
        any_contact, both = s.contact.any(-1), s.contact.all(-1)
        self.age += dt
        self.phase_age += dt
        self.air_time = torch.where(~any_contact, self.air_time + dt, 0.0)
        self.contact_time = torch.where(any_contact, self.contact_time + dt, 0.0)
        safe = (~s.bad_contact & ~s.outside & (s.tilt < math.radians(cfg["launch_tilt_deg"]))
                & (s.root_clearance > cfg["stable_root_height"]))
        ready = (phase == 0) & both & safe
        self.stable_time = torch.where(ready, self.stable_time + dt, 0.0)
        armed = ready & (self.stable_time >= cfg["settle_s"])
        self.phase[armed] = 1
        self.phase_age[armed] = 0
        self.support_leg[armed] = s.leg_height[armed]
        self.support_bed[armed] = s.bed_z[armed]

        # Confirm foot support without asking the body to stop translating vertically.
        touchdown = (phase == 2) & any_contact & active
        self.phase[touchdown] = 3
        self.phase_age[touchdown] = 0
        self.contact_time[touchdown] = dt
        self.touch_time[touchdown] = -1
        self.confirmed[touchdown] = False
        self.support_leg[touchdown] = s.leg_height[touchdown]
        self.support_bed[touchdown] = s.bed_z[touchdown]
        landing = (phase == 3) | touchdown
        self.landing_age = torch.where(touchdown, 0.0, self.landing_age + dt)
        new_touch = landing[:, None] & s.contact & (self.touch_time < 0)
        self.touch_time = torch.where(new_touch, self.landing_age[:, None], self.touch_time)
        paired = (self.touch_time >= 0).all(-1)
        valid_flight = ((self.flight_time >= cfg["valid_flight_s"])
                        & (self.max_clearance >= cfg["valid_clearance_m"]))
        landed = (landing & ~self.confirmed & paired & valid_flight & safe & active
                  & (self.contact_time >= cfg["contact_confirm_s"]))
        height = (self.peak - self.reference).clamp_min(0)
        qualified = landed & (height >= cfg["valid_height_m"])
        continuity = torch.minimum(height, self.last_height) / cfg["height_reward_scale"]
        continuity *= qualified & (self.bounce_count > 0)
        self.previous_height[landed] = self.last_height[landed]
        self.last_height[landed] = height[landed]
        self.bounce_count += qualified
        self.confirmed |= landed
        self.airborne_paid[landed] = False
        # A grazing contact that never became support remains part of the same flight.
        grazing = landing & ~self.confirmed & ~any_contact & (self.air_time >= cfg["minimum_air_s"])
        self.phase[grazing] = 2
        # A safely completed flight settles its height credit; later failures cannot erase older flights.
        self.height_paid[landed] = 0
        self.launch_progress[qualified] = 0
        self.crouch[qualified] = 0
        self.compression[qualified] = 0

        support = ((phase == 1) | (landing & self.confirmed)) & active
        controlled = support & both & safe
        old_crouch, old_compression = self.crouch.clone(), self.compression.clone()
        self.crouch = torch.where(controlled, torch.maximum(
            self.crouch, (self.support_leg - s.leg_height).clamp(0, cfg["crouch_target"])), self.crouch)
        self.compression = torch.where(controlled, torch.maximum(
            self.compression, (self.support_bed - s.bed_z).clamp(0, cfg["compression_target"])), self.compression)
        speed = torch.minimum(s.com_vz, s.leg_extension_speed).clamp(0, cfg["launch_velocity_cap"])
        progress = torch.where(controlled, speed / cfg["launch_velocity_cap"], 0.0)
        best = torch.maximum(self.launch_progress, progress)
        launch_progress = best - self.launch_progress
        self.launch_progress = best
        takeoff = (support & (self.air_time >= cfg["minimum_air_s"])
                   & (s.com_vz > 0) & (s.foot_clearance.min(-1).values > cfg["takeoff_clearance_m"]))
        rebound = takeoff & (phase == 3)
        self.contact_to_takeoff[rebound] = self.landing_age[rebound]
        self.phase[takeoff] = 2
        self.phase_age[takeoff] = 0
        self.peak[takeoff] = self.reference[takeoff]
        self.height_paid[takeoff] = 0
        self.apex_seen[takeoff] = False
        self.flight_time[takeoff] = self.air_time[takeoff]
        self.max_clearance[takeoff] = 0
        flying = ((phase == 2) & ~touchdown) | takeoff | grazing
        self.flight_time += (flying & ~takeoff) * dt
        self.max_clearance = torch.where(flying, torch.maximum(
            self.max_clearance, s.foot_clearance.min(-1).values), self.max_clearance)
        self.peak = torch.where(flying, torch.maximum(self.peak, s.com_z), self.peak)
        progress = (self.peak - self.reference).clamp_min(0) / cfg["height_reward_scale"]
        increment = torch.where(flying & active, (progress - self.height_paid).clamp_min(0), 0.0)
        self.height_paid += increment
        apex = flying & ~self.apex_seen & (s.com_vz <= 0)
        self.apex_seen |= apex

        if cfg.get("reward_revision", 1) >= 2:
            # Require uninterrupted flight and current clearance, not accumulated grazing contacts.
            airborne = (flying & ~any_contact & ~self.airborne_paid & safe & active
                        & (self.air_time >= cfg["valid_flight_s"])
                        & (s.foot_clearance.min(-1).values >= cfg["valid_clearance_m"]))
            self.airborne_paid |= airborne
            # Crouching, brief loss of contact, and oracle phase changes do not restart this clock.
            self.stall_age += active * dt
            self.stall_age[armed | landed | (flying & ~any_contact & self.airborne_paid)] = 0
            stall = ((self.stall_age - cfg["stall_grace_s"]) / cfg["stall_ramp_s"]).clamp(0, 1)
            stall *= any_contact * dt

        failure = active & (
            s.bad_contact | s.outside | (s.tilt > math.radians(cfg["failure_tilt_deg"]))
            | (s.root_clearance < 0.035)
            | ((self.phase == 0) & (self.phase_age > cfg["prepare_timeout_s"]))
            | (((self.phase == 1) | (self.phase == 3)) & (self.phase_age > cfg["takeoff_timeout_s"]))
            | ((self.phase == 2) & (self.phase_age > cfg["flight_timeout_s"]))
        )
        events = {
            "prepare": armed.float(),
            "crouch": (self.crouch - old_crouch).clamp_min(0) / cfg["crouch_target"],
            "compression": (self.compression - old_compression).clamp_min(0) / cfg["compression_target"],
            "launch_velocity": launch_progress,
            "apex_progress": increment - torch.where(failure, self.height_paid, 0.0),
            "launch": takeoff.float(),
            "landing": qualified.float() * height / cfg["height_reward_scale"],
            "continuity": continuity,
            "rebound": rebound.float() * (self.last_height / cfg["height_reward_scale"]).clamp(max=2),
            "upright": torch.cos(s.tilt).clamp_min(0) * dt,
            "failure": failure.float(),
        }
        if cfg.get("reward_revision", 1) >= 2:
            events.update(airborne=airborne.float(), stall=stall)
        for name, value in events.items():
            if name not in ("failure", "apex_progress"):
                events[name] = torch.where(active & ~failure, value, 0.0)
        self.height_paid[failure] = 0
        self.failed |= failure
        self.score = qualified.float()
        self.events = {"takeoff": takeoff & ~failure, "touchdown": touchdown & ~failure,
                       "apex": apex & ~failure, "landing": qualified & ~failure}
        return events, qualified & ~failure, failure
