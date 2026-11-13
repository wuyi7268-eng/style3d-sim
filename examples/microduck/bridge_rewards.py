"""Bridge-only locomotion shaping; common scalar locomotion primitives remain shared.

These implementations are owned by the bridge recipe. Editing them must not change
flat-ground or four-corner-cloth objectives. Keep existing term names for log continuity.
"""

import math

import torch


def is_bridge(recipe):
    return getattr(getattr(recipe.env, "terrain", None), "kind", None) == "elastic_bridge"


def cloth_velocity_tracking(recipe, command_name, min_std, relative_std, max_std, yaw_std,
                            command_threshold, max_tilt_deg, min_height, broad_fraction):
    """Reward directed walking on cloth with a useful gradient even far from the target."""
    data = recipe.scene["robot"].data
    if not is_bridge(recipe):
        return torch.zeros_like(data.root_link_pos_w[:, 0])
    command = recipe.command_manager.get_command(command_name)
    speed = command[:, :2].norm(dim=-1)
    std = (relative_std * speed).clamp(min_std, max_std)
    error = (data.root_link_lin_vel_b[:, :2] - command[:, :2]).square().sum(-1) / std.square()
    error += ((data.root_link_ang_vel_b[:, 2] - command[:, 2]) / yaw_std).square()
    active = speed > command_threshold
    active &= -data.projected_gravity_b[:, 2] >= math.cos(math.radians(max_tilt_deg))
    active &= data.root_link_pos_w[:, 2] - recipe.scene.terrain.env_origins[:, 2] >= min_height
    active &= ~recipe.env.terrain.out_of_bounds(data.root_link_pos_w)
    value = ((1 - broad_fraction) * torch.exp(-error) + broad_fraction / (1 + error)) * active
    recipe.extras["log"]["Metrics/cloth_fine_velocity_tracking"] = value.sum() / active.sum().clamp_min(1)
    return value


def cloth_command_progress(recipe, command_name, command_threshold, max_tilt_deg, min_height,
                           lateral_std, yaw_std):
    """Give dense credit for moving along a translational command on cloth."""
    data = recipe.scene["robot"].data
    if not is_bridge(recipe):
        return torch.zeros_like(data.root_link_pos_w[:, 0])
    command = recipe.command_manager.get_command(command_name)
    speed = command[:, :2].norm(dim=-1)
    direction = command[:, :2] / speed[:, None].clamp_min(1.0e-6)
    velocity = data.root_link_lin_vel_b[:, :2]
    along = (velocity * direction).sum(-1)
    lateral = velocity - along[:, None] * direction
    progress = (along / speed.clamp_min(1.0e-6)).clamp(0.0, 1.0)
    tracking_gate = torch.exp(-lateral.square().sum(-1) / lateral_std**2
                              - ((data.root_link_ang_vel_b[:, 2] - command[:, 2]) / yaw_std).square())
    active = speed > command_threshold
    active &= -data.projected_gravity_b[:, 2] >= math.cos(math.radians(max_tilt_deg))
    active &= data.root_link_pos_w[:, 2] - recipe.scene.terrain.env_origins[:, 2] >= min_height
    active &= ~recipe.env.terrain.out_of_bounds(data.root_link_pos_w)
    value = progress * tracking_gate * active
    recipe.extras["log"]["Metrics/cloth_command_progress"] = value.sum() / active.sum().clamp_min(1)
    return value


def cloth_stall_penalty(recipe, command_name, command_threshold, max_tilt_deg, min_height,
                        lateral_std, yaw_std):
    """Keep the progress signal from being bypassed by a quiet upright policy."""
    if not is_bridge(recipe):
        return torch.zeros_like(recipe.scene["robot"].data.root_link_pos_w[:, 0])
    command = recipe.command_manager.get_command(command_name)
    progress = cloth_command_progress(recipe, command_name, command_threshold, max_tilt_deg,
                                      min_height, lateral_std, yaw_std)
    active = command[:, :2].norm(dim=-1) > command_threshold
    return (active.float() - progress).clamp_min(0.0)


def cloth_contact_slip(recipe, sensor_name, slip_std):
    """Bound the cost of stance slip relative to the cloth, including zero-command standing."""
    data = recipe.scene[sensor_name].data
    if not is_bridge(recipe):
        return torch.zeros_like(data.found[:, 0])
    contact = data.soft_found > 0
    squared = data.soft_slip_squared_sum / data.soft_found.clamp_min(1)
    cost = 1 - torch.exp(-squared / slip_std**2)
    recipe.extras["log"]["Metrics/cloth_relative_slip_m_s"] = (
        (squared.sqrt() * contact).sum() / contact.sum().clamp_min(1)
    )
    recipe.extras["log"]["Metrics/cloth_foot_contact_fraction"] = contact.float().mean()
    return (cost * contact).sum(-1) / contact.sum(-1).clamp_min(1)


def cloth_nonfoot_contact(recipe, sensor_name, body_sensor_name):
    """Do not let the trunk or shins substitute for foot support on a sagging sheet."""
    feet = recipe.scene[sensor_name].data.found
    if not is_bridge(recipe):
        return torch.zeros_like(feet[:, 0])
    body = recipe.scene[body_sensor_name].data.found.sum(-1)
    return (body > feet.sum(-1)).float()



REWARDS = {
    "cloth_velocity_tracking": cloth_velocity_tracking,
    "cloth_command_progress": cloth_command_progress,
    "cloth_stall_penalty": cloth_stall_penalty,
    "cloth_contact_slip": cloth_contact_slip,
    "cloth_nonfoot_contact": cloth_nonfoot_contact,
}
