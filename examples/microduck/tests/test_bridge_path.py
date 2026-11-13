"""Path-frame feedback, independent resets, and physically supported goal completion."""

import math
from dataclasses import asdict
from types import SimpleNamespace

import torch

from examples.microduck.bridge_path import BridgePath
from examples.microduck.scenes import SCENE_CONFIG
from examples.microduck.task_profiles import load_task_spec


def fixture(count=3):
    pose = torch.zeros(count, 7)
    pose[:, 6] = 1
    commands = torch.zeros(count, 13)
    env = SimpleNamespace(num_envs=count, device="cpu", step_dt=.02, commands=commands, _views=lambda: (None, None, pose),
                          terrain=SimpleNamespace(cfg={"config": asdict(SCENE_CONFIG)}))
    recipe = SimpleNamespace(env=env, standing=torch.zeros(count, dtype=torch.bool))
    config, _, _ = load_task_spec("velocity", "elastic_bridge")
    path = BridgePath(recipe, config["commands"]["twist"]["path_following"])
    path.reset(torch.arange(count))
    return path, env, pose


def test_correction_sign_frame_bounds_and_partial_reset():
    path, env, pose = fixture()
    pose[:, 1] = torch.tensor([.08, -.08, 0])
    path.update()
    assert env.commands[0, 1] < 0 and env.commands[0, 2] < 0
    assert env.commands[1, 1] > 0 and env.commands[1, 2] > 0
    assert env.commands[2, 1:3].eq(0).all()
    pose[2, 5:7] = torch.tensor([math.sin(math.pi / 4), math.cos(math.pi / 4)])
    path.update()
    assert env.commands[2, 2] < 0
    torch.testing.assert_close(env.commands[2, 0], torch.tensor(.1))
    assert env.commands[:, 2].abs().max() <= path.config["yaw_limit"]
    before = env.commands.clone()
    path.hold.fill_(.2)
    path.completed.fill_(True)
    path.reset(torch.tensor([0]))
    path.update(torch.tensor([0]))
    torch.testing.assert_close(env.commands[1:], before[1:])
    torch.testing.assert_close(path.hold, torch.tensor([0., .2, .2]))
    assert path.completed.tolist() == [False, True, True]


def test_success_requires_both_feet_contact_stillness_and_no_failure():
    path, env, pose = fixture()
    pose[:, 0] = path.goal_x
    feet = torch.zeros(3, 3, 3)
    feet[:, 1:3, 0] = path.goal_x
    found = torch.ones(3, 2)
    path.recipe.robot = SimpleNamespace(data=SimpleNamespace(site_pos_w=feet))
    path.recipe.sensors = SimpleNamespace(feet=SimpleNamespace(data=SimpleNamespace(found=found)))
    gravity = torch.tensor([[0., 0., -1.]] * 3)
    velocity = torch.zeros(3, 3)
    failure = torch.tensor([False, False, True])
    found[1, 1] = 0
    path.update()
    assert env.commands[:, :3].eq(0).all()
    successes = torch.zeros(3, dtype=torch.int)
    for _ in range(30):
        _, success, _ = path.evaluate(pose, velocity, gravity, failure)
        successes += success
    assert successes.tolist() == [1, 0, 0]
    path.reset(torch.arange(3))
    found.fill_(1)
    velocity[:, 0] = .2
    for _ in range(30):
        _, success, _ = path.evaluate(pose, velocity, gravity, torch.zeros(3, dtype=torch.bool))
        assert not success.any()


def test_progress_is_signed():
    path, _env, pose = fixture(1)
    # Exercise signed progress for saved recipes even though new training disables this reward.
    path.config["progress_weight"] = 5.0
    path.recipe.robot = SimpleNamespace(data=SimpleNamespace(site_pos_w=torch.zeros(1, 3, 3)))
    path.recipe.sensors = SimpleNamespace(feet=SimpleNamespace(data=SimpleNamespace(found=torch.ones(1, 2))))
    args = (torch.zeros(1, 3), torch.tensor([[0., 0., -1.]]), torch.tensor([False]))
    pose[:, 0] = .01
    forward, _, _ = path.evaluate(pose, *args)
    pose[:, 0] = 0
    backward, _, _ = path.evaluate(pose, *args)
    torch.testing.assert_close(forward + backward, torch.zeros(1))


def test_no_goal_distance_slowdown_and_body_forward_floor():
    path, env, pose = fixture(4)
    path.speed.fill_(.3)
    pose[:, 0] = torch.tensor([0., path.goal_x - .06, path.goal_x + .08, path.goal_x])
    path.update()
    torch.testing.assert_close(env.commands[:, 0], torch.tensor([.3, .3, .3, 0.]))
    # Even after a large yaw and lateral-frame rotation, moving commands obey the requested floor.
    pose[:, 0] = 0
    yaw = torch.tensor([0., math.pi / 2, math.pi, -math.pi / 2])
    pose[:, 5] = (yaw / 2).sin()
    pose[:, 6] = (yaw / 2).cos()
    pose[:, 1] = .1
    path.update()
    assert (env.commands[:, 0] >= .1).all()
