"""Bridge objectives, optimizer defaults and resume guards must not leak into flat Velocity."""

import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from examples.microduck import bridge_rewards, task_profiles
from examples.microduck.common import setup
from examples.microduck.env import environment_config
from examples.microduck.ppo import apply_official_ppo, ppo_config
from examples.microduck.recipe import OfficialRecipe, recipe_fingerprint
from examples.microduck.scenes import SCENE_CONFIG
from examples.microduck.tests.test_cloth_learning import cloth_fixture


@pytest.fixture
def isolated_configs(tmp_path, monkeypatch):
    for name in ("SNAPSHOT", "BRIDGE_RECIPE", "BRIDGE_TRAINING"):
        original = getattr(task_profiles, name)
        target = tmp_path / original.name
        target.write_bytes(original.read_bytes())
        monkeypatch.setattr(task_profiles, name, target)
    return task_profiles


def edit_json(path, mutate):
    value = json.loads(path.read_text())
    mutate(value)
    path.write_text(json.dumps(value))


def test_bridge_recipe_edits_leave_flat_spec_and_fingerprint_unchanged(isolated_configs):
    flat = load("default")
    bridge = load("elastic_bridge")
    flat_hash, bridge_hash = fingerprints()

    # Find the curriculum by its target, so the test is independent of display names.
    document = json.loads(isolated_configs.BRIDGE_RECIPE.read_text())
    key = next(k for k, v in document["spec"]["curricula"].items()
               if v["params"].get("reward_name") == "action_rate_l2")
    change_spec = deepcopy(document)
    change_spec["spec"]["rewards"]["air_time"]["weight"] = .125
    change_spec["spec"]["curricula"][key]["params"]["weight_stages"][0]["weight"] = -9
    change_spec["spec"]["commands"]["twist"]["path_following"]["speed_range"] = [.04, .08]
    isolated_configs.BRIDGE_RECIPE.write_text(json.dumps(change_spec))
    assert load("default") == flat
    assert load("elastic_bridge") != bridge
    assert fingerprints()[0] == flat_hash
    assert fingerprints()[1] != bridge_hash


def load(terrain):
    return task_profiles.load_task_spec("velocity", terrain)[0]


def fingerprints():
    return recipe_fingerprint(), recipe_fingerprint("velocity", "elastic_bridge")


def test_flat_configuration_edits_do_not_change_bridge(isolated_configs):
    bridge, bridge_hash = load("elastic_bridge"), fingerprints()[1]
    edit_json(isolated_configs.SNAPSHOT,
              lambda d: d["tasks"]["velocity"]["rewards"]["air_time"].update(weight=99))
    assert load("default")["rewards"]["air_time"]["weight"] == 99
    assert load("elastic_bridge") == bridge
    assert fingerprints()[1] == bridge_hash


def test_bridge_ppo_settings_are_independent(isolated_configs):
    flat = apply_official_ppo(ppo_config(2), load("default"), 2)
    bridge = apply_official_ppo(ppo_config(2), load("elastic_bridge"), 2)
    assert (bridge["num_steps_per_env"], bridge["save_interval"]) == (24, 50)
    assert bridge["algorithm"]["learning_rate"] == 1e-4
    assert bridge["algorithm"]["entropy_coef"] == .005
    assert bridge["algorithm"]["schedule"] == "adaptive"
    flat_hash = fingerprints()[0]
    edit_json(isolated_configs.BRIDGE_TRAINING, lambda d: d["rl"]["algorithm"].update(learning_rate=2e-4))
    assert apply_official_ppo(ppo_config(2), load("default"), 2) == flat
    assert apply_official_ppo(ppo_config(2), load("elastic_bridge"), 2)["algorithm"]["learning_rate"] == 2e-4
    assert fingerprints()[0] == flat_hash


@pytest.mark.parametrize("filename", ["bridge_rewards.py", "bridge_path.py"])
def test_bridge_code_fingerprint_does_not_invalidate_flat(monkeypatch, filename):
    before = fingerprints()
    read_bytes = Path.read_bytes

    def changed_bytes(path):
        contents = read_bytes(path)
        return contents + b"\n# Bridge-only change\n" if path.name == filename else contents

    monkeypatch.setattr(Path, "read_bytes", changed_bytes)
    after = fingerprints()
    assert after[0] == before[0]
    assert after[1] != before[1]


def test_bridge_dispatch_and_reward_parity():
    from examples.microduck import reward_functions

    recipe, _, data, _, _ = cloth_fixture(2)
    recipe.env.terrain.kind = "elastic_bridge"
    recipe.reward_overrides = bridge_rewards.REWARDS
    data.root_link_lin_vel_b[:, 0] = torch.tensor([0., .1])
    for name, term in load("elastic_bridge")["rewards"].items():
        function = term["func"].rsplit(".", 1)[-1]
        if function not in bridge_rewards.REWARDS:
            continue
        expected = getattr(reward_functions, function)(recipe, **term["params"])
        actual = OfficialRecipe.base_reward(recipe, function, term["params"])
        torch.testing.assert_close(actual, expected, msg=name)
    recipe.reward_overrides = {"cloth_command_progress": lambda *_args, **_kwargs: torch.full((2,), 7.)}
    torch.testing.assert_close(OfficialRecipe.base_reward(recipe, "cloth_command_progress", {}), torch.full((2,), 7.))


def test_cli_bridge_training_defaults_and_scope():
    flat, _, _ = setup("train", ["--solver", "synreal"])
    bridge, _, _ = setup("train", ["--solver", "synreal", "--terrain", "elastic_bridge"])
    assert (flat.iterations, flat.policy_variant) == (5, "standard")
    assert (bridge.iterations, bridge.policy_variant, bridge.run_name) == (10000, "cloth_history", "elastic_bridge_velocity")
    explicit, _, _ = setup("train", ["--solver", "synreal", "--terrain", "elastic_bridge", "--iterations", "2"])
    assert explicit.iterations == 2
    with pytest.raises(SystemExit):
        setup("train", ["--solver", "synreal", "--training-config", str(task_profiles.BRIDGE_TRAINING)])


def test_short_entry_rejects_other_task_or_terrain():
    from examples.microduck.elastic_bridge import main

    for options in (["--terrain", "default"], ["--task=velstand"], ["--solver", "mujoco"]):
        with pytest.raises(SystemExit):
            main(["train", *options])


def test_existing_velocity_checkpoint_can_initialize_bridge():
    from examples.microduck.checkpoint import transfer_metadata

    checkpoint = Path("/home/ubuntu/models/velocity_24300/model_24300.pt")
    if not checkpoint.is_file():
        pytest.skip("Optional local transfer checkpoint is unavailable")
    metadata = transfer_metadata(checkpoint)
    args, _, _ = setup("train", ["--solver", "synreal", "--terrain", "elastic_bridge", "--init-from", str(checkpoint)])
    assert metadata["policy_variant"] == "history"
    assert args.policy_variant == "cloth_history"
    assert args.resume is None


@pytest.mark.parametrize("terrain", ["default", "elastic_bridge"])
def test_bridge_edits_only_block_bridge_resume(tmp_path, isolated_configs, terrain):
    checkpoint = tmp_path / "model_0.pt"
    checkpoint.touch()
    metadata = {
        "solver": "synreal", "task": "velocity", "recipe": "official_adapted_v1",
        "observation_contract": "official_asymmetric_pd_v2",
        "official_alignment": {"recipe_fingerprint": recipe_fingerprint("velocity", terrain)},
        "physics": asdict(environment_config("elastic_bridge" if terrain == "elastic_bridge" else "official").physics),
        "terrain": {"kind": terrain, "config": asdict(SCENE_CONFIG)},
        "ppo": {"policy": {"class_name": "ActorCritic"}},
    }
    checkpoint.with_name("config.json").write_text(json.dumps(metadata))
    options = ["--solver", "synreal", "--resume", str(checkpoint), "--use-checkpoint-environment"]
    setup("train", options)
    edit_json(isolated_configs.BRIDGE_RECIPE, lambda d: d["spec"]["rewards"]["air_time"].update(weight=.1))
    if terrain == "elastic_bridge":
        with pytest.raises(SystemExit):
            setup("train", options)
    else:
        setup("train", options)
