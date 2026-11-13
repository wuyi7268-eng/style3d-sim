"""Running source fidelity, task isolation, curricula and real CPU PPO lifecycle."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from examples.microduck import task_profiles
from examples.microduck.common import setup
from examples.microduck.env import MicroDuckEnv
from examples.microduck.play import update_playback_observation
from examples.microduck.ppo import apply_official_ppo, create_runner, ppo_config
from examples.microduck.recipe import OfficialRecipe, recipe_fingerprint
from examples.microduck.running import (
    command_ranges,
    running_forward_progress,
    running_planar_drift_cost,
)
from examples.microduck.task_registry import OFFICIAL_TASKS, base_task


def test_running_config_and_ui_keep_existing_tasks():
    from examples.microduck.ui import configuration

    spec = task_profiles.load_task_spec("running")[0]
    cfg = apply_official_ppo(ppo_config(1), spec, 1)
    assert base_task("running") == "velocity"
    assert spec["episode_length_s"] == 12
    assert spec["commands"]["twist"]["rel_standing_envs"] == .03
    assert spec["rewards"]["forward_progress"]["weight"] == 5
    assert spec["rewards"]["forward_progress"]["params"]["speed_cap"] == 2.4
    assert spec["rewards"]["foot_clearance"]["weight"] == -2
    assert spec["rewards"]["air_time"]["weight"] == 0
    assert not any(name.startswith(("cloth_", "gait_")) for name in spec["rewards"])
    assert cfg["num_steps_per_env"] == 24
    assert cfg["algorithm"]["entropy_coef"] == .02
    assert "symmetry_cfg" not in cfg["algorithm"]
    snapshot = json.loads(task_profiles.SNAPSHOT.read_text())["tasks"]
    ui = configuration()
    for task in OFFICIAL_TASKS:
        assert task_profiles.load_task_spec(task)[0] == snapshot[task]
        assert ui["ppo"][task] == apply_official_ppo(ppo_config(1024), snapshot[task], 1024)
    assert ui["ppo"]["running"]["algorithm"]["entropy_coef"] == .02
    assert setup("train", ["--task", "running"])[0].policy_variant == "standard"
    with pytest.raises(SystemExit):
        setup("train", ["--task", "running", "--solver", "synreal", "--terrain", "elastic_bridge"])
    with pytest.raises(ValueError, match="flat ground"):
        task_profiles.load_task_spec("running", "cloth")


def test_running_recipe_and_code_edits_are_isolated(tmp_path, monkeypatch):
    path = tmp_path / "running.json"
    path.write_bytes(task_profiles.RUNNING_RECIPE.read_bytes())
    monkeypatch.setattr(task_profiles, "RUNNING_RECIPE", path)
    selections = [("velocity", "default"), ("velocity", "elastic_bridge"), ("standup", "default"), ("running", "default")]
    before = [recipe_fingerprint(*pair) for pair in selections]
    data = json.loads(path.read_text())
    data["spec"]["rewards"]["forward_progress"]["weight"] = 7
    path.write_text(json.dumps(data))
    after = [recipe_fingerprint(*pair) for pair in selections]
    assert before[:-1] == after[:-1]
    assert before[-1] != after[-1]
    original = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda p: original(p) + (b"\n# changed" if p.name == "running.py" else b""))
    changed = [recipe_fingerprint(*pair) for pair in selections]
    assert changed[:-1] == after[:-1]
    assert changed[-1] != after[-1]


def test_forward_reward_and_drift_match_upstream_equations():
    velocity = torch.tensor([[-1., 0., 0.], [0., .1, 0.], [1.2, -.2, 0.], [3., .3, 0.]])
    angular = torch.zeros_like(velocity)
    angular[:, 2] = .1
    command = torch.zeros(4, 3)
    recipe = SimpleNamespace(robot=SimpleNamespace(data=SimpleNamespace(
        root_link_lin_vel_b=velocity, root_link_ang_vel_b=angular)), get_command=lambda _: command, extras={"log": {}})
    torch.testing.assert_close(running_forward_progress(recipe, 2.4), torch.tensor([0., 0., .5, 1.]))
    torch.testing.assert_close(running_planar_drift_cost(recipe), .01 + 4 * velocity[:, 1].square())
    with pytest.raises(ValueError):
        running_forward_progress(recipe, 0)


def test_speed_stages_and_partial_command_reset():
    spec = task_profiles.load_task_spec("running")[0]
    count = 4096
    recipe = SimpleNamespace(
        spec=spec, common_step_counter=0, device="cpu", step_dt=.02,
        env=SimpleNamespace(task="running", family="velocity", commands=torch.zeros(count, 13)),
        standing=torch.zeros(count, dtype=torch.bool), timers={k: torch.zeros(count) for k in spec["commands"]},
        velocity_commands=None, bridge_path=None,
    )
    recipe.get_command = lambda name: OfficialRecipe.get_command(recipe, name)
    stages = spec["curricula"]["running_speed_range"]["params"]["speed_stages"]
    for index, stage in enumerate(stages):
        recipe.common_step_counter = stage["step"]
        assert command_ranges(recipe)["lin_vel_x"] == [stage["min_speed"], stage["max_speed"]]
        if index:
            recipe.common_step_counter -= 1
            assert command_ranges(recipe)["lin_vel_x"][1] == stages[index - 1]["max_speed"]
    recipe.common_step_counter = stages[-1]["step"]
    torch.manual_seed(9)
    OfficialRecipe.update_commands(recipe, torch.arange(count))
    commands = recipe.env.commands
    assert .02 < recipe.standing.float().mean() < .04
    assert torch.all(commands[recipe.standing, :3] == 0)
    assert torch.all((commands[~recipe.standing, 0] >= 1.65) & (commands[~recipe.standing, 0] <= 2.2))
    before = commands.clone()
    OfficialRecipe.update_commands(recipe, torch.tensor([1, 3]))
    torch.testing.assert_close(commands[[0, 2, 4]], before[[0, 2, 4]])
    assert torch.all(recipe.timers["twist"] == 12)


def test_heading_reference_resets_only_the_new_episode():
    from examples.microduck.reward_functions import heading_hold_reward

    data = SimpleNamespace(root_link_quat_w=torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.]]))
    recipe = SimpleNamespace(scene={"robot": SimpleNamespace(data=data)}, num_envs=2,
                             episode_length_buf=torch.zeros(2, dtype=torch.long))
    torch.testing.assert_close(heading_hold_reward(recipe), torch.ones(2))
    data.root_link_quat_w[:] = torch.tensor([math.cos(.2), 0., 0., math.sin(.2)])
    recipe.episode_length_buf[:] = torch.tensor([1, 10])
    torch.testing.assert_close(heading_hold_reward(recipe), torch.tensor([1., math.exp(-1)]))


def test_cpu_training_checkpoint_and_playback(tmp_path):
    env = MicroDuckEnv(task="running", device="cpu", num_envs=1)
    replay_env = None
    try:
        assert (env.num_obs, env.num_privileged_obs, env.max_episode_length) == (61, 70, 600)
        assert env.official.omitted == {}
        cfg = apply_official_ppo(ppo_config(1), env.official.spec, 1)
        runner = create_runner(env, cfg, log_dir=str(tmp_path), device="cpu")
        # Exercise PPO/checkpoints without copying unrelated workspace diffs into test artifacts.
        runner.git_status_repos = []
        before = runner.alg.policy.actor[0].weight.detach().clone()
        runner.learn(2)
        assert not torch.equal(before, runner.alg.policy.actor[0].weight)
        path = tmp_path / "model_1.pt"
        runner.save(str(path))
        # Rebuild the environment because the first runner's normalized initial
        # observation is an inference tensor owned by its runner.
        replay_env = MicroDuckEnv(task="running", device="cpu", num_envs=1)
        restored = create_runner(replay_env, cfg, device="cpu")
        restored.load(str(path))
        obs, _ = replay_env.reset()
        replay_env.commands[:, :3] = torch.tensor([1., 0., 0.])
        update_playback_observation(replay_env, obs)
        assert torch.all(obs[:, 51:] == 0)
        with torch.inference_mode():
            actions = restored.get_inference_policy()(obs)
            torch.testing.assert_close(actions, runner.get_inference_policy()(obs))
            reset_count = replay_env.total_resets
            replay_env.episode_length_buf[:] = 601
            played = replay_env.step_playback(actions)
        assert replay_env.total_resets == reset_count
        assert torch.isfinite(played).all()
        assert all(torch.isfinite(p).all() for p in restored.alg.policy.parameters())
        metadata = {**env.cfg, "ppo": cfg}
        path.with_name("config.json").write_text(json.dumps(metadata))
        assert setup("play", ["--checkpoint", str(path), "--device", "cpu"])[0].task == "running"
        assert setup("train", ["--resume", str(path), "--device", "cpu"])[0].task == "running"
        with pytest.raises(SystemExit):
            setup("train", ["--resume", str(path), "--task", "velocity"])
    finally:
        if replay_env is not None:
            replay_env.close()
        env.close()
