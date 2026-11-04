"""Local Velocity sampling for signed translation/turning and repeated starts/stops."""

import torch


class VelocityCommandSampler:
    def __init__(self, config, num_envs, device):
        self.config = config["local_sampling"]
        self.device = device
        bins_by_axis = []
        for axis in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
            bins = self.config["axis_velocity_bins"][axis]
            lower, upper = config["ranges"][axis]
            # Both signs use these magnitude bins; neither may exceed its task limit or enter the idle deadband.
            if not bins or any(not (0.01 < lo < hi <= min(-lower, upper)) for lo, hi in bins):
                raise ValueError(f"{axis} bins must exceed 0.01 and fit both signed task limits")
            bins_by_axis.append(bins)
        if len({len(bins) for bins in bins_by_axis}) != 1:
            raise ValueError("Each command axis must have the same number of speed bins")
        for name in ("start_stop_fraction", "pure_translation_fraction"):
            if not 0 <= self.config[name] <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        self.turn_fraction = config.get("rel_turn_in_place_envs", 0.0)
        if not 0 <= self.turn_fraction <= 1:
            raise ValueError("rel_turn_in_place_envs must be in [0, 1]")
        for name in ("standing_time_range_s", "moving_time_range_s"):
            lo, hi = self.config[name]
            if not 0 < lo <= hi:
                raise ValueError(f"Invalid {name}: expected 0 < minimum <= maximum")
        self.bins = torch.tensor(bins_by_axis, device=device)
        self.active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.moving = torch.zeros_like(self.active)
        self.motion_command = torch.zeros(num_envs, 3, device=device)

    def sample_axis_commands(self, count, axes=(0, 1, 2)):
        # Balance axes, signs, and speed bins independently; translation and yaw keep their own units and limits.
        choices = torch.tensor(axes, device=self.device)
        axis = choices[torch.randint(len(axes), (count,), device=self.device)]
        bucket = torch.randint(self.bins.shape[1], (count,), device=self.device)
        bounds = self.bins[axis, bucket]
        magnitude = bounds[:, 0] + torch.rand(count, device=self.device) * (bounds[:, 1] - bounds[:, 0])
        sign = torch.where(torch.rand(count, device=self.device) < 0.5, -1.0, 1.0)
        commands = torch.zeros(count, 3, device=self.device)
        commands.scatter_(1, axis[:, None], (sign * magnitude)[:, None])
        return commands

    def sample_random(self, values):
        # Keep general multi-axis samples, then override a subset with pure signed translation or turning.
        count = len(values)
        translation = torch.rand(count, device=self.device) < self.config["pure_translation_fraction"]
        values[translation] = self.sample_axis_commands(count, axes=(0, 1))[translation]
        turn = torch.rand(count, device=self.device) < self.turn_fraction
        values[turn] = self.sample_axis_commands(count, axes=(2,))[turn]
        return turn

    def reset(self, ids):
        # Select independently per episode, so the sequence is not tied to one friction/bias realization.
        self.active[ids] = torch.rand(len(ids), device=self.device) < self.config["start_stop_fraction"]
        self.motion_command[ids] = self.sample_axis_commands(len(ids))
        # resample() toggles the phase; the first command after reset must be standing.
        self.moving[ids] = True

    def resample(self, selected, values, durations):
        mask = self.active[selected]
        ids = selected[mask]
        self.moving[ids] = ~self.moving[ids]
        moving = self.moving[ids]
        # Preserve direction and speed after each stop, including pure yaw, without resetting the robot's state.
        values[mask] = torch.where(moving[:, None], self.motion_command[ids], 0.0)
        idle_lo, idle_hi = self.config["standing_time_range_s"]
        move_lo, move_hi = self.config["moving_time_range_s"]
        lo = torch.where(moving, move_lo, idle_lo)
        hi = torch.where(moving, move_hi, idle_hi)
        durations[mask] = lo + torch.rand(len(ids), device=self.device) * (hi - lo)
