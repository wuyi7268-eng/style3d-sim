"""XL330 quasi-static current telemetry for the local position-PD actuator.

The official BAM pickup pipeline uses actuator_force / kt. Here the available
input is the saturated PD drive torque, not BAM's motor torque. Consequently
this models equivalent current, not gearbox friction or voltage-loop dynamics.
"""

import torch

LEG_INDICES = (0, 1, 2, 3, 4, 9, 10, 11, 12, 13)
BUS_LEG_INDICES = (0, 1, 2, 3, 4, 10, 11, 12, 13, 14)
CURRENT_SPEC = {
    "motor": "dynamixel_xl330",
    "model": "quasistatic_pd_equivalent_current",
    "kt_nm_per_a": 0.36601349688984386,
    "parameter_source": "https://github.com/Rhoban/bam/blob/62bd8ce12154340be97e06f7f41a0ca8f116d967/bam/params/xl330/m6.json",
    "torque_source": "saturated_pd_drive_before_physics_step",
    "measurement": "absolute_present_current",
    "resolution_a": 0.001,
    "scale_a": 1.0,
    "sampling": "last_physics_substep_of_control_interval",
    "reset": "zero_until_first_complete_control_interval",
    "noise_delay": "none_hardware_calibration_pending",
}


def simulated_leg_current(torque):
    """Return absolute amperes, quantized like the official 1 mA bus reading.

    BAM's fitted kt is already referred to the joint output; applying another
    gear ratio would double-count the transmission. Preserve the local drive
    limit rather than introducing a different current limit only in telemetry.
    """
    if torque.shape[-1] != 14:
        raise ValueError("Expected 14 PD torques in policy joint order")
    if not torch.isfinite(torque).all():
        raise FloatingPointError("Non-finite torque in simulated current sensor")
    current = torque.detach()[..., LEG_INDICES].abs() / CURRENT_SPEC["kt_nm_per_a"]
    resolution = CURRENT_SPEC["resolution_a"]
    return torch.round(current / resolution) * resolution / CURRENT_SPEC["scale_a"]


def hardware_leg_current(currents_ma):
    """Convert official robotd's 15-joint bus array (mouth at index 9) to policy input.

    The runtime has already taken the absolute value, but accepting signed raw
    register readings here makes the conversion unambiguous in both cases.
    """
    if currents_ma.shape[-1] != 15:
        raise ValueError("Expected robotd's 15-joint currents_ma array")
    if not torch.isfinite(currents_ma).all():
        raise FloatingPointError("Non-finite hardware current")
    return currents_ma[..., BUS_LEG_INDICES].abs() / (1000.0 * CURRENT_SPEC["scale_a"])
