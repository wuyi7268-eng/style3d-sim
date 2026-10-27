"""Upstream reset and termination formulas; generated from the bundled mdp.py."""

# ruff: noqa: F841, UP037, UP045, B008
from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import torch

from .official_adapter import SceneEntityCfg

ManagerBasedRlEnv = Entity = Any
_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

_ROULADE_FWD_SIGN = 1.0


def _servo_joint_ids(env: "ManagerBasedRlEnv", asset: Entity) -> list:
    """Entity-local indices of the servo (non-``passive_``) joints, cached.

    All joint-index-based reward/event params in this module (``joint_indices``,
    ``target_overrides``, qpos-column math) are written against the canonical
    14-servo layout. On models with extra unactuated joints — backlash hinges,
    roller wheels, the jaw linkage, all named ``passive_*`` — the entity joint
    array is wider and interleaved, so raw indices would select the wrong
    joints. Index through this list to recover the servo-only view; on plain
    models it is the identity.
    """
    cache = env.__dict__.setdefault("_servo_joint_ids_cache", {})
    key = id(asset)
    ids = cache.get(key)
    if ids is None:
        ids, _ = asset.find_joints("^(?!passive_).*")
        cache[key] = ids
    return ids


def _env_origin_z(env: "ManagerBasedRlEnv", env_ids: torch.Tensor) -> torch.Tensor:
    """Terrain height of each env's origin. Every spawn function that writes an
    ABSOLUTE trunk z must add this: on rough terrain (slope pyramids put the
    platform well above z=0) an absolute z of 0.05-0.09 m puts the robot inside
    the terrain → contact explosion → returns/value/std blow up. This is what
    killed the first rough+backlash velstand run (wy8gcaus, 2026-09-10): stable
    until prone spawns switched on at iter 700, then value loss 0.6 → 42 and
    entropy 5 → 33, in lockstep with prone_prob. Flat runs never noticed
    (origin z = 0)."""
    return env.scene.env_origins[env_ids.long(), 2]


def reset_with_forward_velocity(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    velocity_range: tuple[float, float] = (0.3, 0.8),
    fraction_stages: list[dict] | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Warm-start a fraction of reset environments with a random forward velocity.

    The robot spawns already moving in its body-forward direction, so it first
    discovers what coasting at speed feels like. The fraction decreases over
    training, forcing it to progressively earn that speed from rest.

    Args:
        velocity_range: (min, max) forward speed in m/s.
        fraction_stages: list of {"step": int, "fraction": float} dicts, sorted by step.
            The fraction active at the current training step is used.
            Example: [{"step":0,"fraction":0.8}, {"step":2000*24,"fraction":0.0}]
        asset_cfg: robot entity config.
    """
    if fraction_stages is None:
        fraction_stages = [{"step": 0, "fraction": 0.8}]
    step = env.common_step_counter
    fraction = fraction_stages[0]["fraction"]
    for stage in fraction_stages:
        if step >= stage["step"]:
            fraction = stage["fraction"]
    if len(env_ids) == 0 or fraction <= 0.0:
        return
    n_warmstart = max(1, int(len(env_ids) * fraction))
    perm = torch.randperm(len(env_ids), device=env.device)[:n_warmstart]
    warmstart_ids = env_ids[perm]
    lo, hi = velocity_range
    vx = lo + torch.rand(n_warmstart, device=env.device) * (hi - lo)
    asset: Entity = env.scene[asset_cfg.name]
    qpos_q_adr = asset.data.indexing.free_joint_q_adr[3:7]
    q = asset.data.data.qpos[warmstart_ids][:, qpos_q_adr]
    w, x, y, z = (q[:, 0], q[:, 1], q[:, 2], q[:, 3])
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    forward_world = torch.stack([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)], dim=-1)
    velocities = torch.zeros(n_warmstart, 6, device=env.device)
    velocities[:, :3] = vx.unsqueeze(-1) * forward_world
    asset.write_root_link_velocity_to_sim(velocities, env_ids=warmstart_ids)
    _WHEEL_RADIUS = 0.0175
    all_wheel_ids, _ = asset.find_joints("^passive_.*")
    if all_wheel_ids:
        joint_pos = asset.data.joint_pos[warmstart_ids].clone()
        joint_vel = asset.data.joint_vel[warmstart_ids].clone()
        omega = vx / _WHEEL_RADIUS
        joint_vel[:, all_wheel_ids] = omega.unsqueeze(-1).expand(-1, len(all_wheel_ids))
        asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=warmstart_ids)


def _fallen_mask(env: ManagerBasedRlEnv, asset, gate_z_below: float, gate_tilt_above_deg: float) -> torch.Tensor:
    """Per-env float mask: 1.0 where the robot counts as FALLEN — trunk height
    below `gate_z_below` OR tilt beyond `gate_tilt_above_deg`. Used to gate the
    recovery rewards so they only steer while actually fallen and contribute
    exactly zero during clean walking (no walk tax / bounce farming)."""
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    quat = asset.data.root_link_quat_w
    cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    fallen = (z < gate_z_below) | (cos_tilt < math.cos(math.radians(gate_tilt_above_deg)))
    return fallen.float()


def fallen_too_long(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    gate_z_below: float = 0.1,
    gate_tilt_above_deg: float = 40.0,
    max_duration_s: float = 5.0,
) -> torch.Tensor:
    """Terminate envs that have been continuously FALLEN for `max_duration_s`.

    For envs that mix walking with fall recovery (velstand): the fell_over
    termination gets disabled by curriculum so the policy can attempt recovery,
    but without a backstop a failed recovery farms recovery-reward for the whole
    20 s episode, starving the walk of data (audit: ~25% walking share). This
    gives every fall a fair recovery window, then recycles the env.
    """
    asset: Entity = env.scene[asset_cfg.name]
    fallen = _fallen_mask(env, asset, gate_z_below, gate_tilt_above_deg).bool()
    if not hasattr(env, "_fallen_timer_s"):
        env._fallen_timer_s = torch.zeros(env.num_envs, device=env.device)
    env._fallen_timer_s[env.episode_length_buf <= 1] = 0.0
    env._fallen_timer_s = torch.where(fallen, env._fallen_timer_s + env.step_dt, torch.zeros_like(env._fallen_timer_s))
    return env._fallen_timer_s >= max_duration_s


def reset_rolling_entry(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    speed_range: tuple = (0.25, 0.45),
    wheel_radius: float = 0.0175,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Départ en ROULEMENT sans glissement (élan aux roues).

    Tire une vitesse d'avance v par env ; met la vitesse LINÉAIRE de base (x
    monde) = v ET la vitesse de ROTATION des 4 roues passives = v / r, donc
    ω·r = v => zéro glissement au contact. Évite l'à-coup de l'ancienne poussée
    base-seule (base qui bouge, roues immobiles = patinage brutal au 1er pas).
    À exécuter APRÈS reset_base (qui pose la base ; ne plus lui donner de
    velocity_range).
    """
    asset: Entity = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = int(env_ids.shape[0])
    lo, hi = speed_range
    v = torch.rand(n, device=env.device) * (hi - lo) + lo
    root_vel = torch.zeros(n, 6, device=env.device)
    root_vel[:, 0] = v
    asset.write_root_link_velocity_to_sim(root_vel, env_ids=env_ids)
    wheel_ids = []
    for name in ("passive_LF_?wheel", "passive_LR_?wheel", "passive_RF_?wheel", "passive_RR_?wheel"):
        ids, _ = asset.find_joints(name)
        wheel_ids.append(ids[0])
    wheel_ids_t = torch.tensor(wheel_ids, device=env.device)
    omega = (v / wheel_radius).unsqueeze(1).repeat(1, len(wheel_ids))
    asset.write_joint_velocity_to_sim(omega, joint_ids=wheel_ids_t, env_ids=env_ids)


def set_random_prone_orientation(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    face_down_prob: float = 0.5,
    side_prob: float = 0.0,
):
    """Randomly initialize each env lying down, with random yaw.

    First ``side_prob`` of the envs lie on a SIDE (±90° roll, left/right 50/50);
    the rest split face-down / face-up by ``face_down_prob``.

    Face-down:  +90° pitch → quat = [s*cy, -s*sy,  s*cy,  s*sy]
    Face-up:    -90° pitch → quat = [s*cy,  s*sy, -s*cy,  s*sy]
    On side:    ±90° roll  → quat = yaw ⊗ roll = [s*cy, ±s*cy, ±s*sy, s*sy]

    Side spawns were added 2026-09 (protective-fall velstand run 1): a headless
    eval showed 79% of pushed falls end ON THE SIDE, 20% face-up, 0% face-down —
    the dominant fallen state had no reverse-curriculum data at all.

    Args:
        face_down_prob: probability of face-down (vs face-up) among the non-side envs.
        side_prob: fraction of envs placed on a side.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.int)
    num = len(env_ids)
    yaw = torch.rand(num, device=env.device) * 2 * np.pi - np.pi
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    s = 2.0 ** (-0.5)
    face_down = torch.stack([s * cy, -s * sy, s * cy, s * sy], dim=1)
    face_up = torch.stack([s * cy, s * sy, -s * cy, s * sy], dim=1)
    sign = torch.where(torch.rand(num, device=env.device) < 0.5, 1.0, -1.0)
    on_side = torch.stack([s * cy, sign * s * cy, sign * s * sy, s * sy], dim=1)
    u = torch.rand(num, device=env.device)
    side_mask = u < side_prob
    mask = torch.rand(num, device=env.device) < face_down_prob
    new_quat = torch.where(mask.unsqueeze(1), face_down, face_up)
    new_quat = torch.where(side_mask.unsqueeze(1), on_side, new_quat)
    env.sim.data.qpos[env_ids, 3:7] = new_quat
    env.sim.data.qvel[env_ids, :6] = 0.0


def set_random_ground_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    face_down_prob: float = 0.4,
    face_up_prob: float = 0.4,
    sitting_prob: float = 0.2,
    standing_prob: float = 0.0,
    prone_z_min: float = 0.2,
    prone_z_max: float = 0.25,
    sitting_z_min: float = 0.07,
    sitting_z_max: float = 0.09,
    standing_z_min: float = 0.11,
    standing_z_max: float = 0.12,
    sitting_joint_overrides: Optional[dict] = None,
    sitting_joint_noise_std: float = 0.0,
    sitting_tilt_max: float = 0.0,
    face_up_roll_max: float = 0.0,
):
    """Reset to a random ground state: face-down, face-up, sitting, or standing.

    Broader than ``set_random_prone_orientation`` — used by the stand-up env so
    the policy learns to recover from any plausible pose, including the sitting
    keyframe (rest state of the sit policy) and an already-standing pose (so it
    also learns to *hold* a stand, not only to rise).

    Modes (probabilities are normalized; they need not sum to 1.0):
      - face-down (belly to floor): +90° pitch, random yaw, z in [prone_z_min, prone_z_max].
      - face-up   (back to floor):  -90° pitch, random yaw, z in [prone_z_min, prone_z_max].
      - sitting:                    upright (±sitting_tilt_max), random yaw, z low,
                                    joints set to ``sitting_joint_overrides``.
      - standing:                   upright (±sitting_tilt_max), random yaw, z in
                                    [standing_z_min, standing_z_max], joints left at
                                    HOME (whatever ``reset_robot_joints`` set).

    Args:
        sitting_joint_overrides: ``{qpos_joint_index: angle_rad}`` to write into
            ``qpos[7+idx]`` for envs sampled into the sitting bucket. ``None``
            keeps joints at whatever ``reset_robot_joints`` already set.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.int)
    num = len(env_ids)
    total = face_down_prob + face_up_prob + sitting_prob + standing_prob
    p_fd = face_down_prob / total
    p_fu = (face_down_prob + face_up_prob) / total
    p_sit = (face_down_prob + face_up_prob + sitting_prob) / total
    yaw = torch.rand(num, device=env.device) * 2 * np.pi - np.pi
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    s = 2.0 ** (-0.5)
    face_down = torch.stack([s * cy, -s * sy, s * cy, s * sy], dim=1)
    face_up = torch.stack([s * cy, s * sy, -s * cy, s * sy], dim=1)
    if sitting_tilt_max > 0.0:
        pitch = (torch.rand(num, device=env.device) * 2 - 1) * sitting_tilt_max
        roll = (torch.rand(num, device=env.device) * 2 - 1) * sitting_tilt_max
        cp = torch.cos(pitch * 0.5)
        sp = torch.sin(pitch * 0.5)
        cr = torch.cos(roll * 0.5)
        sr = torch.sin(roll * 0.5)
        sit_w = cr * cp * cy + sr * sp * sy
        sit_x = sr * cp * cy - cr * sp * sy
        sit_y = cr * sp * cy + sr * cp * sy
        sit_z = cr * cp * sy - sr * sp * cy
        sitting = torch.stack([sit_w, sit_x, sit_y, sit_z], dim=1)
    else:
        sitting = torch.stack([cy, torch.zeros_like(cy), torch.zeros_like(cy), sy], dim=1)
    u = torch.rand(num, device=env.device)
    is_fd = u < p_fd
    is_fu = (u >= p_fd) & (u < p_fu)
    is_sit = (u >= p_fu) & (u < p_sit)
    is_stand = u >= p_sit
    if face_up_roll_max > 0.0:
        theta = (torch.rand(num, device=env.device) * 2 - 1) * face_up_roll_max
        ct = torch.cos(theta * 0.5)
        st = torch.sin(theta * 0.5)
        w, x, y, z = (face_up[:, 0], face_up[:, 1], face_up[:, 2], face_up[:, 3])
        face_up = torch.stack([w * ct - z * st, x * ct + y * st, y * ct - x * st, w * st + z * ct], dim=1)
    new_quat = face_down.clone()
    new_quat[is_fu] = face_up[is_fu]
    new_quat[is_sit] = sitting[is_sit]
    new_quat[is_stand] = sitting[is_stand]
    z_prone = torch.rand(num, device=env.device) * (prone_z_max - prone_z_min) + prone_z_min
    z_sit = torch.rand(num, device=env.device) * (sitting_z_max - sitting_z_min) + sitting_z_min
    z_stand = torch.rand(num, device=env.device) * (standing_z_max - standing_z_min) + standing_z_min
    new_z = z_prone.clone()
    new_z = torch.where(is_sit, z_sit, new_z)
    new_z = torch.where(is_stand, z_stand, new_z)
    env.sim.data.qpos[env_ids, 2] = new_z + _env_origin_z(env, env_ids)
    env.sim.data.qpos[env_ids, 3:7] = new_quat
    env.sim.data.qvel[env_ids, :6] = 0.0
    asset: Entity = env.scene[asset_cfg.name]
    servo_ids = _servo_joint_ids(env, asset)
    if sitting_joint_overrides:
        sit_env_ids = env_ids[is_sit]
        if len(sit_env_ids) > 0:
            for jnt_idx, angle in sitting_joint_overrides.items():
                env.sim.data.qpos[sit_env_ids, 7 + servo_ids[jnt_idx]] = angle
    if sitting_joint_noise_std > 0.0:
        sit_env_ids = env_ids[is_sit]
        if len(sit_env_ids) > 0:
            n_sit = len(sit_env_ids)
            cols = torch.tensor([7 + j for j in servo_ids], device=env.device, dtype=torch.long)
            noise = torch.randn(n_sit, len(cols), device=env.device) * sitting_joint_noise_std
            env.sim.data.qpos[sit_env_ids.unsqueeze(1).long(), cols.unsqueeze(0)] += noise


_CROUCH_ANCHOR_BY_NAME = {
    "left_hip_pitch": -1.15,
    "left_knee": 1.25,
    "left_ankle": 1.05,
    "right_hip_pitch": 1.15,
    "right_knee": -1.25,
    "right_ankle": -1.05,
}


def set_random_crouch_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    depth_min: float = 0.35,
    depth_max: float = 1.0,
    pitch_max_deg: float = 55.0,
    joint_noise: float = 0.12,
    z_stand: float = 0.115,
    z_deep: float = 0.06,
):
    """Reset selected envs into a random mid-recovery crouch.

    Reverse curriculum for the recovery last mile (velstand run-5 lesson):
    prone-init episodes spend most of their fallen budget getting TO the deep
    crouch and are recycled shortly after reaching it, so the crouch→stand
    mile gets almost no on-policy data — the policy converged to parking
    there. Seeding resets ACROSS that mile (depth λ ∈ [depth_min, depth_max]
    between standing and the deep-crouch anchor, trunk pitch and z scaled
    with λ) makes the frontier dense from step 0 of the episode.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.long)
    num = len(env_ids)
    asset: Entity = env.scene[asset_cfg.name]
    lam = torch.rand(num, device=env.device) * (depth_max - depth_min) + depth_min
    joints = asset.data.default_joint_pos[env_ids].clone()
    for name, anchor in _CROUCH_ANCHOR_BY_NAME.items():
        ids, _ = asset.find_joints(f"^{name}$")
        j = ids[0]
        joints[:, j] = joints[:, j] + lam * (anchor - joints[:, j])
    noise_mask = torch.zeros(joints.shape[1], device=joints.device)
    noise_mask[_servo_joint_ids(env, asset)] = 1.0
    joints += (torch.rand_like(joints) * 2 - 1) * joint_noise * noise_mask
    pitch = lam * math.radians(pitch_max_deg) + (torch.rand(num, device=env.device) * 2 - 1) * math.radians(10.0)
    pitch = torch.clamp(pitch, min=math.radians(5.0))
    roll = (torch.rand(num, device=env.device) * 2 - 1) * math.radians(8.0)
    yaw = torch.rand(num, device=env.device) * 2 * np.pi - np.pi
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    quat = torch.stack([qw, qx, qy, qz], dim=1)
    z = z_stand + lam * (z_deep - z_stand) + torch.rand(num, device=env.device) * 0.01
    env.sim.data.qpos[env_ids, 2] = z + _env_origin_z(env, env_ids)
    env.sim.data.qpos[env_ids, 3:7] = quat
    env.sim.data.qpos[env_ids, 7:] = joints
    env.sim.data.qvel[env_ids, :] = 0.0


def randomize_servo_joints_uniform(
    env: ManagerBasedRlEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, range_frac: float = 0.8
):
    """Sample every servo joint uniformly over the central ``range_frac`` of its
    limits for ``env_ids``; zero joint velocities. Uses the entity write API, so it
    is correct on backlash/roller models where passive joints interleave."""
    asset: Entity = env.scene[asset_cfg.name]
    ids = torch.as_tensor(_servo_joint_ids(env, asset), device=env.device, dtype=torch.long)
    lim = asset.data.joint_pos_limits[env_ids][:, ids]
    lo, hi = (lim[..., 0], lim[..., 1])
    mid, half = (0.5 * (lo + hi), 0.5 * (hi - lo) * range_frac)
    sample = mid + (torch.rand(len(env_ids), len(ids), device=env.device) * 2.0 - 1.0) * half
    pos = asset.data.joint_pos[env_ids].clone()
    pos[:, ids] = sample
    asset.write_joint_position_to_sim(pos, env_ids=env_ids)
    asset.write_joint_velocity_to_sim(torch.zeros_like(pos), env_ids=env_ids)


def maybe_set_random_prone_orientation(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    prone_prob: float = 0.0,
    face_down_prob: float = 0.5,
    prone_z_min: float = 0.2,
    prone_z_max: float = 0.25,
    crouch_prob: float = 0.0,
    side_prob: float = 0.0,
    joint_random_prob: float = 0.0,
    joint_range_frac: float = 0.8,
    joint_random_extra_z: float = 0.06,
    so3_prob: float = 0.0,
    so3_extra_z: float = 0.04,
    sit_prob: float = 0.0,
    sit_params: Optional[dict] = None,
):
    """Reset event that overrides orientation to prone with probability `prone_prob`.

    ``so3_prob`` (2026-10, velstand stand-up coverage): fraction of the prone
    slice that gets a UNIFORMLY random orientation instead of an exact ±90°
    one, always with randomized servo joints (head included) and
    ``joint_random_extra_z + so3_extra_z`` of extra clearance — the robot drops
    a few cm and lands in whatever rest state that pose gives. Covers the
    "put down / fell somewhere odd" starts the ±90° HOME-joint spawns never show.

    ``sit_prob``: an exclusive slice (after prone and crouch) reset into the
    sitting bucket of ``set_random_ground_state`` with ``sit_params`` (joint
    overrides, noise, tilt, z band).

    ``joint_random_extra_z``: extra spawn clearance for the joint-randomized
    envs. Randomly folded legs reach ~10 cm below the trunk; on rough terrain
    they interpenetrate a step/slope at spawn (measured 800-1200 N spikes vs
    ~110 N for HOME joints). The short extra drop costs ~70 N.

    ``side_prob`` (fraction of the prone slice lying on a side) is passed through
    to set_random_prone_orientation.

    ``joint_random_prob``: fraction of the prone slice whose SERVO joints are
    re-sampled uniformly over the central ``joint_range_frac`` of each joint's
    limits (velocities zeroed). Post-fall-like spawns (2026-09-10, protective
    fall run 4): a real fall leaves the legs wherever the gait had them — hip
    roll pinned at its limit, head yaw 1.7 rad off, knees anywhere — and the
    policy recovered only 58-77% of face-up landings after a push vs 100% from
    HOME-pose face-up spawns. The remaining fraction keeps the HOME pose (the
    daemon's stand-from-init case).

    With prob `prone_prob`, replaces the upright orientation (already set by
    reset_base) with a prone orientation; otherwise leaves it upright. Among the
    overridden envs, `face_down_prob` picks face-down (belly) vs face-up (back).

    Also lifts z to [prone_z_min, prone_z_max] for the overridden envs so the
    head/neck clearance is sufficient — the vel-env reset z (~0.125) would
    clip the head through the ground at 90° pitch.

    At prone_prob=2/3 and face_down_prob=0.5 you get a balanced 33/33/33 split
    of upright/face-down/face-up resets, which is the standard mixture for
    learning fall recovery alongside normal upright start.

    With ``crouch_prob`` > 0, an additional exclusive slice of envs is reset
    into a random mid-recovery crouch via ``set_random_crouch_state`` (reverse
    curriculum for the recovery last mile — see its docstring).
    """
    if prone_prob <= 0.0 and crouch_prob <= 0.0 and (sit_prob <= 0.0):
        return
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if len(env_ids) == 0:
        return
    env_ids_t = (
        env_ids.to(env.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor)
        else torch.tensor(env_ids, device=env.device, dtype=torch.long)
    )
    u = torch.rand(len(env_ids_t), device=env.device)
    selected = env_ids_t[u < prone_prob]
    crouch_selected = env_ids_t[(u >= prone_prob) & (u < prone_prob + crouch_prob)]
    sit_selected = env_ids_t[(u >= prone_prob + crouch_prob) & (u < prone_prob + crouch_prob + sit_prob)]
    if len(selected) > 0:
        set_random_prone_orientation(env, selected, asset_cfg=asset_cfg, face_down_prob=face_down_prob, side_prob=side_prob)
        z = torch.rand(len(selected), device=env.device) * (prone_z_max - prone_z_min) + prone_z_min
        env.sim.data.qpos[selected, 2] = z + _env_origin_z(env, selected)
        so3 = torch.rand(len(selected), device=env.device) < so3_prob
        if so3.any():
            ids = selected[so3]
            env.sim.data.qpos[ids, 3:7] = _uniform_random_quat(len(ids), env.device)
            env.sim.data.qpos[ids, 2] += so3_extra_z
        if joint_random_prob > 0.0 or so3.any():
            jr = selected[(torch.rand(len(selected), device=env.device) < joint_random_prob) | so3]
            if len(jr) > 0:
                randomize_servo_joints_uniform(env, jr, asset_cfg=asset_cfg, range_frac=joint_range_frac)
                env.sim.data.qpos[jr, 2] += joint_random_extra_z
    if len(crouch_selected) > 0:
        set_random_crouch_state(env, crouch_selected, asset_cfg=asset_cfg)
    if len(sit_selected) > 0:
        set_random_ground_state(
            env,
            sit_selected,
            asset_cfg=asset_cfg,
            face_down_prob=0.0,
            face_up_prob=0.0,
            sitting_prob=1.0,
            standing_prob=0.0,
            **sit_params or {},
        )


def _uniform_random_quat(n: int, device) -> torch.Tensor:
    """Uniform random unit quaternions (w, x, y, z) — Shoemake's method."""
    a, b, c = torch.rand(n, 3, device=device).unbind(1)
    return torch.stack(
        [
            torch.sqrt(1 - a) * torch.sin(2 * math.pi * b),
            torch.sqrt(1 - a) * torch.cos(2 * math.pi * b),
            torch.sqrt(a) * torch.sin(2 * math.pi * c),
            torch.sqrt(a) * torch.cos(2 * math.pi * c),
        ],
        dim=1,
    )


def _roulade_state(env: ManagerBasedRlEnv) -> tuple:
    if not hasattr(env, "_roulade_accum"):
        z = torch.zeros(env.num_envs, device=env.device)
        env._roulade_accum = z.clone()
        env._roulade_max = z.clone()
        env._roulade_paid = z.clone()
        env._roulade_head_latch = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._roulade_last_update_step = -1
    return (env._roulade_accum, env._roulade_max, env._roulade_paid)


def reset_roulade_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    standing_prob: float = 0.5,
    midroll_prob: float = 0.5,
    standing_z_min: float = 0.11,
    standing_z_max: float = 0.12,
    standing_tilt_max: float = 0.0,
    forward_vel_range: tuple = (0.0, 0.0),
    midroll_pitch_min: float = math.radians(50.0),
    midroll_pitch_max: float = math.radians(185.0),
    midroll_z_min: float = 0.05,
    midroll_z_max: float = 0.1,
    midroll_omega_range: tuple = (0.0, 0.0),
    tuck_overrides: Optional[dict] = None,
    tuck_factor_range: tuple = (0.3, 1.0),
    joint_noise_std: float = 0.0,
):
    """Reset to a standing start or a mid-roll state (reverse curriculum).

    Standing bucket: upright (±standing_tilt_max pitch/roll noise), random yaw,
    HOME joints (left from reset_robot_joints), z in [standing_z_min, _max].
    ``forward_vel_range`` is the élan hook: a per-env forward base velocity
    (body x, mapped to world through the spawn yaw) sampled uniformly — 0 for
    a standstill roll, widen it later to train rolls out of a walk.

    Mid-roll bucket: pitched ``midroll_pitch_min..max`` into the roll (90° =
    on the head, 180° = on the back), random yaw, legs lerped HOME→tuck by a
    per-env factor in ``tuck_factor_range``, z in [midroll_z_min, _max],
    optional forward angular momentum from ``midroll_omega_range``. The
    rotation accumulator is initialized to the spawn pitch so progress
    accounting (and the completion gates) stay consistent: a 170° spawn only
    gets paid for the remaining ~190°.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.long)
    num = len(env_ids)
    asset: Entity = env.scene[asset_cfg.name]
    accum, max_accum, paid = _roulade_state(env)
    total = standing_prob + midroll_prob
    is_mid = torch.rand(num, device=env.device) < midroll_prob / max(total, 1e-06)
    yaw = torch.rand(num, device=env.device) * 2 * np.pi - np.pi
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    pitch = (torch.rand(num, device=env.device) * 2 - 1) * standing_tilt_max
    mid_pitch = torch.rand(num, device=env.device) * (midroll_pitch_max - midroll_pitch_min) + midroll_pitch_min
    pitch = torch.where(is_mid, mid_pitch, pitch)
    roll = (torch.rand(num, device=env.device) * 2 - 1) * max(standing_tilt_max, math.radians(5.0))
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    quat = torch.stack([qw, qx, qy, qz], dim=1)
    z_stand = torch.rand(num, device=env.device) * (standing_z_max - standing_z_min) + standing_z_min
    z_mid = torch.rand(num, device=env.device) * (midroll_z_max - midroll_z_min) + midroll_z_min
    new_z = torch.where(is_mid, z_mid, z_stand)
    env.sim.data.qpos[env_ids, 2] = new_z + _env_origin_z(env, env_ids)
    env.sim.data.qpos[env_ids, 3:7] = quat
    env.sim.data.qvel[env_ids, :6] = 0.0
    servo_ids = _servo_joint_ids(env, asset)
    mid_env_ids = env_ids[is_mid]
    if len(mid_env_ids) > 0 and tuck_overrides:
        u = (
            torch.rand(len(mid_env_ids), device=env.device) * (tuck_factor_range[1] - tuck_factor_range[0])
            + tuck_factor_range[0]
        )
        for jnt_idx, angle in tuck_overrides.items():
            col = 7 + servo_ids[jnt_idx]
            home = env.sim.data.qpos[mid_env_ids, col]
            env.sim.data.qpos[mid_env_ids, col] = home + u * (angle - home)
    if len(mid_env_ids) > 0 and joint_noise_std > 0.0:
        cols = torch.tensor([7 + j for j in servo_ids], device=env.device, dtype=torch.long)
        noise = torch.randn(len(mid_env_ids), len(cols), device=env.device) * joint_noise_std
        env.sim.data.qpos[mid_env_ids.unsqueeze(1), cols.unsqueeze(0)] += noise
    if len(mid_env_ids) > 0 and midroll_omega_range[1] > 0.0:
        omega = (
            torch.rand(len(mid_env_ids), device=env.device) * (midroll_omega_range[1] - midroll_omega_range[0])
            + midroll_omega_range[0]
        )
        env.sim.data.qvel[mid_env_ids, 4] = _ROULADE_FWD_SIGN * omega
    stand_env_ids = env_ids[~is_mid]
    if len(stand_env_ids) > 0 and forward_vel_range[1] > 0.0:
        vx = (
            torch.rand(len(stand_env_ids), device=env.device) * (forward_vel_range[1] - forward_vel_range[0])
            + forward_vel_range[0]
        )
        yaw_s = yaw[~is_mid]
        env.sim.data.qvel[stand_env_ids, 0] = vx * torch.cos(yaw_s)
        env.sim.data.qvel[stand_env_ids, 1] = vx * torch.sin(yaw_s)
    spawn_angle = torch.where(is_mid, mid_pitch, torch.zeros_like(mid_pitch))
    accum[env_ids] = spawn_angle
    max_accum[env_ids] = spawn_angle
    paid[env_ids] = spawn_angle
    env._roulade_head_latch[env_ids] = is_mid
