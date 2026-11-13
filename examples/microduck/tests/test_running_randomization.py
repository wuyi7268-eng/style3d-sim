"""Running robustness boundaries, non-accumulating model edits and PPO continuation."""

from copy import deepcopy
from types import SimpleNamespace

import newton
import pytest
import torch
import warp as wp

from examples.microduck.env import MicroDuckEnv
from examples.microduck.ppo import apply_official_ppo, create_runner, ppo_config
from examples.microduck.running_randomization import (
    RunningRandomization,
    robustness_stage,
    train_stages,
)
from examples.microduck.task_profiles import load_task_spec


@pytest.mark.parametrize("iteration,stage", [(0, 0), (11748, 0), (11749, 1), (11848, 1), (11849, 2),
                                           (11998, 2), (11999, 3), (12198, 3), (20000, 3)])
def test_released_stage_boundaries(iteration, stage):
    spec = load_task_spec("running")[0]
    assert robustness_stage(spec, iteration * 24)[0] == stage
    assert sum(s["updates"] for s in spec["robustness"]["stages"]) == 450


def model_fixture(count=4):
    model = SimpleNamespace(
        body_mass=wp.array([2., 1., .5] * count, dtype=wp.float32, device="cpu"),
        body_inv_mass=wp.array([.5, 1., 2.] * count, dtype=wp.float32, device="cpu"),
        body_inertia=wp.array([wp.mat33(.2, 0., 0., 0., .3, 0., 0., 0., .4)] * count * 3,
                             dtype=wp.mat33, device="cpu"),
        body_inv_inertia=wp.array([wp.mat33(5., 0., 0., 0., 1 / .3, 0., 0., 0., 2.5)] * count * 3,
                                 dtype=wp.mat33, device="cpu"),
        body_com=wp.array([[.01, .02, .03]] * count * 3, dtype=wp.vec3, device="cpu"),
        joint_armature=wp.array([.0018] * count * 20, dtype=wp.float32, device="cpu"),
        body_label=["robot/trunk_base", "robot/jaw_soft", "robot/leg"] * count,
    )
    calls = []
    env = SimpleNamespace(model=model, num_envs=count, device="cpu", reset_on_fall=True, cfg={},
                          qdi=torch.arange(6, 20), _sync=lambda: None, backend="mujoco",
                          solver=SimpleNamespace(notify_model_changed=calls.append))
    return env, calls


def test_randomization_is_bounded_correlated_and_partial_reset_safe():
    env, calls = model_fixture()
    spec = load_task_spec("running")[0]
    spec["running_randomization_enabled"] = True
    randomizer = RunningRandomization(env, spec)
    scale = randomizer.mass[:, 0] / randomizer.nominal_mass[:, 0]
    assert torch.all((scale >= .95) & (scale <= 1.05))
    torch.testing.assert_close(randomizer.inertia[:, 0], randomizer.nominal_inertia[:, 0] * scale[:, None, None])
    torch.testing.assert_close(wp.to_torch(env.model.body_inv_mass).reshape(4, 3)[:, 0], 1 / randomizer.mass[:, 0])
    torch.testing.assert_close(wp.to_torch(env.model.body_inv_inertia).reshape(4, 3, 3, 3)[:, 0],
                               torch.linalg.inv(randomizer.inertia[:, 0]))
    torch.testing.assert_close(randomizer.mass[:, 1:], randomizer.nominal_mass[:, 1:])
    recipe = SimpleNamespace(common_step_counter=11999 * 24)
    q = torch.zeros(4, 21)
    q[:, 6] = 1
    original_com = randomizer.com.clone()
    original_armature = randomizer.armature.clone()
    ids = torch.tensor([1, 3])
    for _ in range(50):
        randomizer.reset(ids, recipe, q)
        delta = randomizer.com - randomizer.nominal_com
        assert delta[:, 0].abs().max() <= .008001
        assert delta[:, 1].abs().max() <= .006001
        assert not delta[:, 2].any()
        armature_scale = randomizer.armature[ids[:, None], env.qdi] / .0018
        assert torch.all((armature_scale >= .9) & (armature_scale <= 1.1))
    torch.testing.assert_close(randomizer.com[[0, 2]], original_com[[0, 2]])
    torch.testing.assert_close(randomizer.armature[[0, 2]], original_armature[[0, 2]])
    torch.testing.assert_close(randomizer.armature[:, :6], original_armature[:, :6])
    torch.testing.assert_close(q[ids, 3:7].norm(dim=-1), torch.ones(2))
    assert q[ids, 3:5].abs().max() < .018
    assert all(c == newton.ModelFlags.BODY_INERTIAL_PROPERTIES | newton.ModelFlags.JOINT_DOF_PROPERTIES for c in calls)
    assert randomizer.push_range(recipe) == {"x": (-.1, .1), "y": (-.1, .1)}
    recipe.common_step_counter = 0
    assert randomizer.push_range(recipe) == {"x": (0., 0.), "y": (0., 0.)}


@pytest.mark.parametrize("training", [False, True])
def test_disabled_randomization_does_not_change_model_or_spawn(training):
    env, calls = model_fixture()
    env.reset_on_fall = training
    spec = load_task_spec("running")[0]
    spec["running_disturbances_enabled"] = False
    randomizer = RunningRandomization(env, spec)
    q = torch.randn(4, 21)
    old = q.clone()
    recipe = SimpleNamespace(common_step_counter=11999 * 24)
    randomizer.reset(torch.arange(4), recipe, q)
    assert calls == []
    torch.testing.assert_close(q, old)
    torch.testing.assert_close(randomizer.mass, randomizer.nominal_mass)
    torch.testing.assert_close(randomizer.inertia, randomizer.nominal_inertia)
    torch.testing.assert_close(randomizer.com, randomizer.nominal_com)
    torch.testing.assert_close(randomizer.armature, randomizer.nominal_armature)
    assert randomizer.push_range(recipe)["x"] == (0., 0.)


@pytest.mark.parametrize("training", [False, True])
def test_disturbances_preserve_nominal_inertial_properties(training):
    env, calls = model_fixture()
    env.reset_on_fall = training
    randomizer = RunningRandomization(env, load_task_spec("running")[0])
    q = torch.zeros(4, 21)
    q[:, 6] = 1
    before = q.clone()
    recipe = SimpleNamespace(common_step_counter=11999 * 24)
    randomizer.reset(torch.tensor([1, 3]), recipe, q)
    assert calls == []
    for name in ("mass", "inertia", "com", "armature"):
        torch.testing.assert_close(getattr(randomizer, name), getattr(randomizer, "nominal_" + name))
    torch.testing.assert_close(q[[0, 2]], before[[0, 2]])
    if training:
        assert not torch.equal(q[[1, 3]], before[[1, 3]])
        torch.testing.assert_close(q[:, 3:7].norm(dim=-1), torch.ones(4))
        assert q[:, 3:5].abs().max() < .018
        assert randomizer.push_range(recipe)["x"] == (-.1, .1)
    else:
        torch.testing.assert_close(q, before)
        assert randomizer.push_range(recipe)["x"] == (0., 0.)


def test_velocity_observations_use_each_worlds_randomized_com(monkeypatch):
    # This checks frame conversion only; CPU MuJoCo does not batch worlds.
    def solver_factory(*args, **kwargs):
        return SimpleNamespace(notify_model_changed=lambda flags: None)

    solver_factory.register_custom_attributes = newton.solvers.SolverMuJoCo.register_custom_attributes
    monkeypatch.setattr(newton.solvers, "SolverMuJoCo", solver_factory)
    env = MicroDuckEnv(task="running", device="cpu", num_envs=2)
    try:
        com = wp.to_torch(env.model.body_com).reshape(2, -1, 3)
        com[:, 0] = torch.tensor([[.001, 0., 0.], [.008, 0., 0.]])
        _, _, pose, velocity = env._views()
        pose[:, 3:7] = torch.tensor([0., 0., 0., 1.])
        velocity[:] = torch.tensor([0., 0., 0., 0., 0., 1.])
        values = env._kinematics()
        expected = torch.tensor([[0., -.001, 0.], [0., -.008, 0.]])
        torch.testing.assert_close(values[3], expected)
        q, qd, pose, linear, angular, gravity = values
        env.official.refresh(q, qd, pose, linear, angular, gravity, env.actions, env.actions)
        torch.testing.assert_close(env.official.robot.data.root_link_lin_vel_w, expected)
    finally:
        env.close()


def test_push_events_apply_current_stage_to_only_expired_worlds():
    from examples.microduck.events import PushEvents

    env, _ = model_fixture()
    spec = load_task_spec("running")[0]
    spec["running_randomization_enabled"] = True
    randomizer = RunningRandomization(env, spec)
    velocity = torch.zeros(4, 20)
    synchronized = []
    env.task = "running"
    env.step_dt = .02
    env._views = lambda: (None, velocity, None, None)
    env.sync_selected_state = lambda state, ids: synchronized.append(ids.clone())
    env.states = [None]
    env.running_randomization = randomizer
    recipe = SimpleNamespace(env=env, spec=spec, num_envs=4, device="cpu", common_step_counter=0)
    events = PushEvents(recipe)
    for stage in [0, 11749, 11849, 11999]:
        recipe.common_step_counter = stage * 24
        velocity.zero_()
        synchronized.clear()
        events.timers["push_robot"][:] = torch.tensor([0., 2., 0., 2.])
        events.apply()
        limit = randomizer.push_range(recipe)["x"][1]
        assert velocity.abs().max() <= limit
        assert not velocity[[1, 3]].any()
        assert not velocity[:, 2:].any()
        if limit:
            assert velocity[[0, 2], :2].abs().sum() > 0
            torch.testing.assert_close(synchronized[0], torch.tensor([0, 2]))
        else:
            assert synchronized == []
        assert torch.all((events.timers["push_robot"][[0, 2]] >= 3) & (events.timers["push_robot"][[0, 2]] <= 6))


def test_cpu_ppo_crosses_phases_and_preserves_optimizer(tmp_path):
    env = MicroDuckEnv(task="running", device="cpu", num_envs=1)
    try:
        # Compress only the phase schedule for a real, short PPO lifecycle test.
        spec = env.official.spec
        spec["robustness"] = deepcopy(spec["robustness"])
        spec["robustness"]["start_iteration"] = 1
        for stage in spec["robustness"]["stages"]:
            stage["updates"] = 1
        env.running_randomization.spec = spec
        cfg = apply_official_ppo(ppo_config(1), spec, 1)
        runner = create_runner(env, cfg, str(tmp_path), "cpu")
        runner.git_status_repos = []
        optimizer = runner.alg.optimizer
        train_stages(runner, env, 4)
        assert runner.current_learning_iteration == 3
        assert env.official.common_step_counter == 4 * 24
        assert runner.alg.optimizer is optimizer
        assert all(torch.isfinite(p).all() for p in runner.alg.policy.parameters())
        assert all((tmp_path / f"model_{i}.pt").exists() for i in range(4))
        assert {int(state["step"]) for state in optimizer.state.values()} == {20}
        # Match a new CLI invocation: neither PPO normalizers nor observation
        # delay caches from the old process survive checkpoint loading.
        env.close()
        env = MicroDuckEnv(task="running", device="cpu", num_envs=1)
        env.official.spec["robustness"] = deepcopy(spec["robustness"])
        env.running_randomization.spec = env.official.spec
        runner = create_runner(env, cfg, str(tmp_path), "cpu")
        runner.git_status_repos = []
        runner.load(str(tmp_path / "model_3.pt"))
        optimizer = runner.alg.optimizer
        runner.current_learning_iteration += 1
        train_stages(runner, env, 1)
        assert runner.current_learning_iteration == 4
        assert env.official.common_step_counter == 5 * 24
        assert {int(state["step"]) for state in optimizer.state.values()} == {25}
        runner.num_steps_per_env = 48
        with pytest.raises(ValueError, match="24 rollout"):
            train_stages(runner, env, 1)
    finally:
        env.close()
