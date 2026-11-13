"""Low-speed coverage and repeated starts/stops must survive timers and partial resets."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from examples.microduck.recipe import SNAPSHOT, OfficialRecipe
from examples.microduck.reward_functions import CommandProgress


def make_recipe(count=4096):
    recipe = OfficialRecipe.__new__(OfficialRecipe)
    recipe.spec = deepcopy(json.loads(SNAPSHOT.read_text())["tasks"]["velocity"])
    recipe.spec["commands"] = {"twist": recipe.spec["commands"]["twist"]}
    recipe.device = torch.device("cpu")
    recipe.env = SimpleNamespace(commands=torch.zeros(count, 13), family="velocity")
    recipe.standing = torch.zeros(count, dtype=torch.bool)
    recipe.phase_offset = torch.zeros(count)
    recipe.timers = {"twist": torch.zeros(count)}
    recipe.common_step_counter = 24001
    recipe.step_dt = 0.02
    recipe.observations = recipe.sensors = SimpleNamespace(reset=lambda ids: None)
    recipe.swing_height = None
    recipe.bridge_path = None
    recipe.command_progress = CommandProgress(count, recipe.device)
    from examples.microduck.velocity_commands import VelocityCommandSampler

    recipe.velocity_commands = VelocityCommandSampler(recipe.spec["commands"]["twist"], count, recipe.device)
    return recipe


def test_all_six_directions_have_low_medium_high_samples_and_keep_mixed_commands():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        recipe = make_recipe(12000)
        recipe.reset(torch.arange(len(recipe.standing)))
    cmd = recipe.env.commands[:, :3]
    zero = (cmd == 0).all(-1)
    bins = [((0.05, 0.15), (0.15, 0.25), (0.25, 0.4)),
            ((0.03, 0.1), (0.1, 0.2), (0.2, 0.3)),
            ((0.05, 0.2), (0.2, 0.5), (0.5, 1.0))]
    single_axis = (cmd != 0).sum(-1) == 1
    for axis, ranges in enumerate(bins):
        for sign in (-1, 1):
            speeds = sign * cmd[single_axis & (sign * cmd[:, axis] > 0), axis]
            for lo, hi in ranges:
                assert ((speeds >= lo) & (speeds < hi)).sum() > 50, (axis, sign, lo, hi)
    turns = (cmd[:, :2] == 0).all(-1) & (cmd[:, 2] != 0)
    assert turns.sum() > 300
    assert ((cmd != 0).sum(-1) == 3).float().mean() > 0.15
    assert (cmd.abs() <= torch.tensor([0.4, 0.3, 1.0])).all()
    assert torch.equal(zero, recipe.standing)
    assert zero[recipe.velocity_commands.active].all()
    ordinary = ~recipe.velocity_commands.active
    assert abs(zero[ordinary].float().mean().item() - 0.2 * 0.85) < 0.02
    assert abs(turns[ordinary].float().mean().item() - 0.15) < 0.02


def test_repeated_stop_start_has_real_dwell_and_keeps_the_same_speed():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        recipe = make_recipe(1024)
        recipe.reset(torch.arange(1024))
        ids = recipe.velocity_commands.active.nonzero().flatten()
        assert len(ids) > 0
        initial = recipe.env.commands.clone()
        for _ in range(50):
            recipe.update_commands()
        torch.testing.assert_close(recipe.env.commands, initial)
        moving_command = None
        for moving in (True, False, True, False):
            # Only this cohort's command expires; other environments and head/body slots must remain untouched.
            before = recipe.env.commands.clone()
            recipe.timers["twist"].fill_(10.0)
            recipe.timers["twist"][ids] = 0
            recipe.update_commands()
            torch.testing.assert_close(recipe.env.commands[~recipe.velocity_commands.active],
                                       before[~recipe.velocity_commands.active])
            torch.testing.assert_close(recipe.env.commands[:, 3:], before[:, 3:])
            if moving:
                command = recipe.env.commands[ids, :3]
                assert ((command != 0).sum(-1) == 1).all()
                for axis in range(3):
                    assert (command[:, axis] > 0).any() and (command[:, axis] < 0).any()
                if moving_command is None:
                    moving_command = command.clone()
                else:
                    torch.testing.assert_close(command, moving_command)
                assert not recipe.standing[ids].any()
                assert ((recipe.timers["twist"][ids] >= 3) & (recipe.timers["twist"][ids] <= 4)).all()
            else:
                assert (recipe.env.commands[ids, :3] == 0).all()
                assert recipe.standing[ids].all()
                assert ((recipe.timers["twist"][ids] >= 2) & (recipe.timers["twist"][ids] <= 3)).all()


def test_partial_reset_restarts_sequence_without_touching_other_environments():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(8)
        recipe = make_recipe(64)
        recipe.reset(torch.arange(64))
        recipe.timers["twist"].zero_()
        recipe.update_commands()
        commands = recipe.env.commands.clone()
        timer = recipe.timers["twist"].clone()
        moving = recipe.velocity_commands.moving.clone()
        active = recipe.velocity_commands.active.clone()
        motion_command = recipe.velocity_commands.motion_command.clone()
        recipe.command_progress.age.fill_(1.0)
        recipe.command_progress.previous.copy_(recipe.env.commands[:, :2])
        previous_command = recipe.command_progress.previous.clone()
        recipe.reset(torch.tensor([0]))
    torch.testing.assert_close(recipe.env.commands[1:], commands[1:])
    torch.testing.assert_close(recipe.timers["twist"][1:], timer[1:])
    torch.testing.assert_close(recipe.velocity_commands.moving[1:], moving[1:])
    torch.testing.assert_close(recipe.velocity_commands.active[1:], active[1:])
    torch.testing.assert_close(recipe.velocity_commands.motion_command[1:], motion_command[1:])
    assert recipe.command_progress.age[0] == 0
    assert (recipe.command_progress.age[1:] == 1).all()
    assert not recipe.command_progress.previous[0].any()
    torch.testing.assert_close(recipe.command_progress.previous[1:], previous_command[1:])
    if recipe.velocity_commands.active[0]:
        assert recipe.standing[0] and (recipe.env.commands[0, :3] == 0).all()


def test_local_sampling_only_enabled_for_velocity():
    tasks = json.loads(SNAPSHOT.read_text())["tasks"]
    assert tasks["velocity"]["commands"]["twist"]["local"] is True
    assert all("local_sampling" not in spec["commands"]["twist"] for task, spec in tasks.items() if task != "velocity")


@pytest.mark.parametrize("axis,limit", [("lin_vel_x", 0.4), ("lin_vel_y", 0.3), ("ang_vel_z", 1.0)])
def test_signed_sampling_rejects_bins_outside_either_direction(axis, limit):
    from examples.microduck.velocity_commands import VelocityCommandSampler

    config = deepcopy(json.loads(SNAPSHOT.read_text())["tasks"]["velocity"]["commands"]["twist"])
    config["ranges"][axis] = [-limit / 2, limit]
    with pytest.raises(ValueError, match=axis):
        VelocityCommandSampler(config, 4, "cpu")


def test_empty_and_partial_expiry_keep_other_commands_unchanged():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(9)
        recipe = make_recipe(64)
        recipe.reset(torch.arange(64))
        before = recipe.env.commands.clone()
        recipe.update_commands(torch.empty(0, dtype=torch.long))
        torch.testing.assert_close(recipe.env.commands, before)
        ids = recipe.velocity_commands.active.nonzero().flatten()[:1]
        assert len(ids) == 1
        recipe.timers["twist"].fill_(10)
        recipe.timers["twist"][ids] = 0
        recipe.update_commands()
        untouched = torch.ones(64, dtype=torch.bool)
        untouched[ids] = False
        torch.testing.assert_close(recipe.env.commands[untouched], before[untouched])
        assert (recipe.env.commands[ids, :3] != 0).sum() == 1
