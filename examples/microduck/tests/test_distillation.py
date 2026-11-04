"""Frozen history teachers, raw-input routing, actor-only BC, and portable resume."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from examples.microduck.distillation import (
    FrozenTeacher,
    teacher_config,
    teacher_masks,
    validate_environment,
    warm_start,
)
from examples.microduck.env import PhysicsConfig
from examples.microduck.observations import CONTRACT, HISTORY_SPEC
from examples.microduck.policy import HistoryActorCritic, HistoryNormalization
from examples.microduck.ppo import create_runner, ppo_config
from examples.microduck.tests.test_history_policy import HistoryEnv


def source(tmp_path, task):
    directory = tmp_path / task
    directory.mkdir()
    cfg = ppo_config(4, "history")
    cfg["policy"].update(actor_hidden_dims=[16], critic_hidden_dims=[16])
    options = dict(cfg["policy"])
    options.pop("class_name")
    critic_dim = 70 if task == "velocity" else 68
    policy = HistoryActorCritic(541, critic_dim + 480, 14, **options)
    norm = HistoryNormalization(61)
    critic_norm = HistoryNormalization(critic_dim, norm)
    norm(torch.randn(32, 541) * 2 + (3 if task == "velocity" else 8))
    critic_norm(torch.randn(32, critic_dim + 480))
    optimizer = torch.optim.Adam(policy.parameters(), lr=7e-5)
    sum(param.square().sum() for param in policy.parameters()).backward()
    optimizer.step()
    metadata = {
        "solver": "synreal", "task": task, "actuator": "position_pd", "self_collisions": False,
        "observation_contract": CONTRACT, "policy_variant": "history", "history": HISTORY_SPEC,
        "recipe": "official_adapted_v1",
        "kp": 2.0, "kd": 0.053, "effort_limit": 0.96, "policy_dt": 0.02,
        "physics": vars(PhysicsConfig()), "ppo": cfg,
        "official_alignment": {"actor_dim": 61, "critic_dim": critic_dim, "recipe_fingerprint": "source"},
    }
    (directory / "config.json").write_text(json.dumps(metadata))
    path = directory / "model_12.pt"
    torch.save({
        "model_state_dict": policy.state_dict(), "obs_norm_state_dict": norm.state_dict(),
        "privileged_obs_norm_state_dict": critic_norm.state_dict(), "iter": 12,
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)
    return path


@pytest.fixture
def config(tmp_path):
    bc = teacher_config(source(tmp_path, "velocity"), source(tmp_path, "standup"))
    bc.update(epochs=1, min_samples=1, min_mini_batch=2, mini_batches=1)
    cfg = ppo_config(4, "history")
    cfg["policy"] = bc["walk"]["policy"]
    cfg["teacher_bc"] = bc
    cfg["num_steps_per_env"] = 2
    cfg["algorithm"].update(num_learning_epochs=1, num_mini_batches=1)
    return cfg


def test_teacher_mapping_independent_normalization_and_freezing(config):
    desc = config["teacher_bc"]["stand"]
    teacher = FrozenTeacher(desc, "cpu")
    teacher.load_source(desc)
    before = copy.deepcopy(teacher.state_dict())
    raw = torch.randn(5, 541)
    original = raw.clone()
    expected = raw.clone()
    expected[:, 48:51] = 0
    actual = teacher(raw, standing=True)
    torch.testing.assert_close(actual, teacher.policy.act_inference(teacher.normalizer(expected)))
    torch.testing.assert_close(raw, original)
    for key, value in teacher.state_dict().items():
        torch.testing.assert_close(value, before[key])
    assert not actual.requires_grad
    assert not any(p.requires_grad for p in teacher.parameters())
    # Mapping command slots must not discard the student's last actions or historical frames.
    torch.testing.assert_close(expected[:, 61:], raw[:, 61:])
    torch.testing.assert_close(expected[:, 34:48], raw[:, 34:48])


def test_state_gates_use_raw_gravity_and_body_routing(config):
    raw = torch.zeros(6, 541)
    angle = torch.deg2rad(torch.tensor([0., 30., 60., 0., 0., 60.]))
    raw[:, 3] = angle.sin()
    raw[:, 5] = -angle.cos()
    raw[3, 58] = 0.1  # standing tilt: StandUp teacher
    raw[4, 57] = -0.02  # standing height: PPO only
    walk, stand = teacher_masks(raw, config["teacher_bc"])
    assert walk.tolist() == [True, False, False, False, False, False]
    assert stand.tolist() == [False, False, True, True, False, True]
    assert not (walk & stand).any()


def test_warm_start_loads_both_encoders_and_normalizers_without_progress(config):
    runner = create_runner(HistoryEnv(), config)
    info = warm_start(runner, config["teacher_bc"])
    checkpoint = torch.load(config["teacher_bc"]["walk"]["path"], weights_only=True)
    for key, value in runner.alg.policy.state_dict().items():
        torch.testing.assert_close(value, checkpoint["model_state_dict"][key])
    for key, value in runner.obs_normalizer.state_dict().items():
        torch.testing.assert_close(value, checkpoint["obs_norm_state_dict"][key])
    for key, value in runner.privileged_obs_normalizer.state_dict().items():
        torch.testing.assert_close(value, checkpoint["privileged_obs_norm_state_dict"][key])
    assert runner.current_learning_iteration == 0
    assert runner.alg.optimizer.state and not runner.alg.bc_optimizer.state
    assert info["optimizer_restored"] is True
    assert runner.alg.learning_rate == runner.alg.optimizer.param_groups[0]["lr"] == 7e-5
    for key, entry in checkpoint["optimizer_state_dict"]["state"].items():
        for field, value in entry.items():
            torch.testing.assert_close(value, runner.alg.optimizer.state_dict()["state"][key][field])


def test_bc_updates_actor_encoder_only_and_keeps_teachers_frozen(config):
    runner = create_runner(HistoryEnv(), config)
    warm_start(runner, config["teacher_bc"])
    alg = runner.alg
    raw = torch.randn(8, 541)
    targets = alg.teachers["stand"](raw, standing=True)
    masks = torch.tensor([[False, True]] * 8)
    before = copy.deepcopy(alg.policy.state_dict())
    teachers = {n: copy.deepcopy(t.state_dict()) for n, t in alg.teachers.items()}
    stats = alg.bc_update(raw, targets, masks)
    after = alg.policy.state_dict()
    assert stats["bc_updates"] > 0
    assert not torch.equal(before["actor_history_encoder.0.weight"], after["actor_history_encoder.0.weight"])
    assert not torch.equal(before["actor.0.weight"], after["actor.0.weight"])
    for key in before:
        if key.startswith("critic") or key == "std":
            torch.testing.assert_close(before[key], after[key])
    for name, teacher in alg.teachers.items():
        for key, value in teacher.state_dict().items():
            torch.testing.assert_close(value, teachers[name][key])
    alg.bc_config["min_samples"] = 100
    assert alg.bc_update(raw, targets, masks)["bc_updates"] == 0


class RoutedEnv(HistoryEnv):
    def __init__(self):
        super().__init__()
        self.set_pose()

    def set_pose(self):
        self.current[:, 3:6] = torch.tensor([0., 0., -1.])
        self.current[2:, 3:6] = torch.tensor([1., 0., 0.])
        self.current[:, 48:] = 0
        self.current[:, 48] = 0.2

    def step(self, actions):
        self.history.advance()
        self.current = torch.randn(4, 61)
        self.set_pose()
        obs, extras = self.get_observations()
        return obs, actions.square().sum(-1), torch.zeros(4, dtype=torch.long), extras


def test_rollout_raw_alignment_and_self_contained_resume_playback(config, tmp_path):
    env = RoutedEnv()
    runner = create_runner(env, config, str(tmp_path / "run"))
    runner.git_status_repos = []
    warm_start(runner, config["teacher_bc"])
    wrapper = runner.alg.observation_env
    normalized, extras = runner.env.get_observations()
    raw = wrapper.raw.clone()
    with torch.inference_mode():
        runner.alg.act(normalized, extras["observations"]["critic"])
    torch.testing.assert_close(runner.alg.bc_raw[0], raw)
    torch.testing.assert_close(runner.alg.bc_targets[0, :2], runner.alg.teachers["walk"](raw[:2]))
    torch.testing.assert_close(runner.alg.bc_targets[0, 2:], runner.alg.teachers["stand"](raw[2:], standing=True))
    runner.learn(1)
    assert runner.alg.bc_optimizer.state
    path = tmp_path / "run/model_0.pt"
    for name in ("walk", "stand"):
        Path(config["teacher_bc"][name]["path"]).unlink()
    restored = create_runner(RoutedEnv(), config, str(tmp_path / "resumed"))
    restored.git_status_repos = []
    restored.load(str(path))
    assert restored.alg.bc_optimizer.state
    assert restored.alg.optimizer.state
    assert restored.alg.learning_rate == runner.alg.learning_rate
    assert restored.alg.learning_rate == restored.alg.optimizer.param_groups[0]["lr"]
    for name in runner.alg.teachers:
        for key, value in runner.alg.teachers[name].state_dict().items():
            torch.testing.assert_close(value, restored.alg.teachers[name].state_dict()[key])
    for key, value in runner.alg.bc_optimizer.state_dict()["state"].items():
        for field, tensor in value.items():
            torch.testing.assert_close(tensor, restored.alg.bc_optimizer.state_dict()["state"][key][field])
    restored.current_learning_iteration += 1
    restored.learn(1)
    # Inference constructs no teachers and works even after their source files disappear.
    playback = create_runner(RoutedEnv(), config, training=False)
    playback.load(str(path), load_optimizer=False)
    assert not hasattr(playback.alg, "teachers")
    torch.testing.assert_close(playback.get_inference_policy()(raw), runner.get_inference_policy()(raw))


def test_rejects_teacher_task_hash_and_environment_mismatch(config):
    with pytest.raises(ValueError, match="requires task"):
        teacher_config(config["teacher_bc"]["stand"]["path"], config["teacher_bc"]["walk"]["path"])
    desc = copy.deepcopy(config["teacher_bc"]["stand"])
    desc["sha256"] = "changed"
    with pytest.raises(ValueError, match="changed"):
        FrozenTeacher(desc, "cpu").load_source(desc)
    env = SimpleNamespace(task="velstand", backend="synreal", mixed_terrain=False,
                          cfg={"self_collisions": False}, physics=PhysicsConfig(linear_iterations=36))
    with pytest.raises(ValueError, match="physics differs"):
        validate_environment(config["teacher_bc"], env)


def test_teacher_allows_only_control_period_preserving_subdivision(config):
    from dataclasses import replace

    source_physics = PhysicsConfig(**config["teacher_bc"]["walk"]["physics"])
    env = SimpleNamespace(task="velstand", backend="synreal", mixed_terrain=False,
                          cfg={"self_collisions": False}, physics=source_physics.with_substeps(8))
    validate_environment(config["teacher_bc"], env)
    for physics in (source_physics.with_substeps(2), source_physics.with_substeps(6),
                    replace(source_physics, physics_dt=0.0025),
                    replace(source_physics.with_substeps(8), linear_iterations=70)):
        env.physics = physics
        with pytest.raises(ValueError, match="physics differs"):
            validate_environment(config["teacher_bc"], env)


def test_cli_physics_subdivision_preserves_policy_and_history_timing(tmp_path):
    from examples.microduck.common import setup

    path = tmp_path / "training.json"
    path.write_text(json.dumps({"training": {"task": "velstand", "physics_substeps": 8}}))
    args, _, _ = setup("train", ["--training-config", str(path)])
    assert args.physics.decimation == 8
    assert args.physics.physics_dt == pytest.approx(0.0025)
    assert args.physics.physics_dt * args.physics.decimation == pytest.approx(0.02)
    override, _, _ = setup("train", ["--training-config", str(path), "--physics-substeps", "16"])
    assert override.physics.physics_dt == pytest.approx(0.00125)


def test_teacher_allows_more_joint_solve_iterations_without_changing_dynamics(config):
    from dataclasses import replace

    physics = PhysicsConfig(**config["teacher_bc"]["walk"]["physics"]).with_substeps(8)
    env = SimpleNamespace(task="velstand", backend="synreal", mixed_terrain=False,
                          cfg={"self_collisions": False}, physics=replace(physics, joint_constraint_primal_iterations=4))
    validate_environment(config["teacher_bc"], env)
    source = copy.deepcopy(config["teacher_bc"])
    source["walk"]["physics"]["joint_constraint_primal_iterations"] = 4
    env.physics = physics
    with pytest.raises(ValueError, match="physics differs"):
        validate_environment(source, env)
    env.physics = replace(physics, joint_constraint_primal_iterations=4, linear_iterations=140)
    with pytest.raises(ValueError, match="physics differs"):
        validate_environment(config["teacher_bc"], env)


def test_cli_joint_solve_refinement_is_explicit_and_preserves_other_physics(tmp_path):
    from examples.microduck.common import setup

    path = tmp_path / "training.json"
    path.write_text(json.dumps({"training": {"task": "velstand", "physics_substeps": 8,
                                             "joint_constraint_primal_iterations": 4}}))
    args, _, _ = setup("train", ["--training-config", str(path)])
    from dataclasses import replace

    assert args.physics == replace(PhysicsConfig().with_substeps(8), joint_constraint_primal_iterations=4)
    override, _, _ = setup("train", ["--training-config", str(path), "--joint-constraint-primal-iterations", "3"])
    assert override.physics.joint_constraint_primal_iterations == 3
    with pytest.raises(SystemExit):
        setup("train", ["--joint-constraint-primal-iterations", "0"])


def test_cli_config_can_select_new_teachers_and_rejects_partial_pair(config, tmp_path):
    from examples.microduck.common import setup

    bc = config['teacher_bc']
    path = tmp_path / 'training.json'
    training = {
        'solver': 'synreal', 'task': 'velstand', 'policy_variant': 'history',
        'velocity_teacher': bc['walk']['path'], 'standup_teacher': bc['stand']['path'],
    }
    path.write_text(json.dumps({'training': training}))
    args, _, _ = setup('train', ['--training-config', str(path)])
    assert args.teacher_bc['stand']['sha256'] == bc['stand']['sha256']
    replacement = Path(bc['stand']['path']).with_name('model_250.pt')
    checkpoint = torch.load(bc['stand']['path'], weights_only=True)
    checkpoint['model_state_dict']['actor.0.bias'] += 0.01
    torch.save(checkpoint, replacement)
    args, _, _ = setup('train', ['--training-config', str(path), '--standup-teacher', str(replacement)])
    assert args.teacher_bc['stand']['path'] == str(replacement)
    assert args.teacher_bc['stand']['sha256'] != bc['stand']['sha256']
    # A directory is resolved once, to a pinned numbered file, when a new run starts.
    args, _, _ = setup('train', ['--training-config', str(path), '--standup-teacher', str(replacement.parent)])
    assert args.teacher_bc['stand']['path'] == str(replacement)
    with pytest.raises(SystemExit):
        setup('train', ['--task', 'velstand', '--solver', 'synreal', '--policy-variant', 'history',
                        '--velocity-teacher', bc['walk']['path']])


def test_warm_start_can_reset_optimizer(config):
    config["teacher_bc"]["warm_start_optimizer"] = "reset"
    runner = create_runner(HistoryEnv(), config)
    info = warm_start(runner, config["teacher_bc"])
    assert not runner.alg.optimizer.state
    assert not info["optimizer_restored"]
    assert runner.alg.learning_rate == config["algorithm"]["learning_rate"]
    assert runner.current_learning_iteration == 0


@pytest.mark.parametrize("min_samples,selected_stand,expected_loss", [(1, 3, 7.25), (4, 0, 2.0)])
def test_bc_mixed_weighted_mean_and_small_walk_group(config, min_samples, selected_stand, expected_loss):
    runner = create_runner(HistoryEnv(), config)
    warm_start(runner, config["teacher_bc"])
    alg = runner.alg
    alg.bc_config.update(min_samples=min_samples, min_mini_batch=100, walk_coef=2.0, stand_coef=3.0)
    raw = torch.randn(5, 541)
    masks = torch.tensor([[True, False]] + [[False, True]] * 3 + [[False, False]])
    # The rollout has advanced statistics since sampling these raw frames.
    runner.obs_normalizer(torch.randn(40, 541) + 12)
    runner.obs_normalizer.eval()
    with torch.no_grad():
        expected_input = runner.obs_normalizer(raw)
        target = alg.policy.act_inference(expected_input).clone()
        target[0] += 1
        target[1:4] += 3 ** 0.5
    runner.obs_normalizer.train()
    before = copy.deepcopy(runner.obs_normalizer.state_dict())
    seen = []
    hook = alg.policy.actor_history_encoder.register_forward_pre_hook(lambda module, inputs: seen.append(inputs[0].clone()))
    stats = alg.bc_update(raw, target, masks)
    hook.remove()
    assert stats["teacher_bc"] == pytest.approx(expected_loss, rel=1e-5)
    assert stats["bc_walk_selected"] == 1
    assert stats["bc_stand_samples"] == 3
    assert stats["bc_stand_selected"] == selected_stand
    assert stats["bc_walk_mse"] == pytest.approx(1, rel=1e-5)
    assert stats["bc_updates"] == 1
    assert runner.obs_normalizer.training
    for key, value in before.items():
        torch.testing.assert_close(value, runner.obs_normalizer.state_dict()[key])
    expected_history = expected_input[masks[:, 0] | (masks[:, 1] & (min_samples == 1)), 61:]
    torch.testing.assert_close(seen[0][:, 0].sort().values, expected_history[:, 0].sort().values)


def test_rejects_incompatible_optimizer_before_training(config):
    path = Path(config["teacher_bc"]["walk"]["path"])
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["optimizer_state_dict"]["state"][0]["exp_avg"] = torch.zeros(123)
    torch.save(checkpoint, path)
    config["teacher_bc"] = teacher_config(path, config["teacher_bc"]["stand"]["path"])
    runner = create_runner(HistoryEnv(), config)
    with pytest.raises(ValueError, match="incompatible exp_avg"):
        warm_start(runner, config["teacher_bc"])
