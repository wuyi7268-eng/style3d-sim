"""External state edits must leave every unselected world's state and history intact."""

import os
from types import SimpleNamespace

import newton
import pytest
import torch
import warp as wp

from examples.microduck.env import MicroDuckEnv
from examples.microduck.events import PushEvents


def kinematic_env(*, articulations=1, shared_body=False):
    source = newton.ModelBuilder()
    for _ in range(articulations):
        body = source.add_link(mass=1.0, inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        joint = source.add_joint_free(child=body)
        source.add_articulation([joint])
    builder = newton.ModelBuilder()
    builder.replicate(source, 4)
    if shared_body:
        body = builder.add_link(mass=1.0, inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        joint = builder.add_joint_free(child=body)
        builder.add_articulation([joint])
    env = MicroDuckEnv.__new__(MicroDuckEnv)
    env.model = builder.finalize(device="cpu")
    env.device, env.backend, env.num_envs = torch.device("cpu"), "mujoco", 4
    env.states = [env.model.state(), env.model.state()]
    env.step_dt = 0.02
    env.commands = torch.zeros(4, 13)
    for state in env.states:
        newton.eval_fk(env.model, state.joint_q, state.joint_qd, state)
        # A body-space integrator need not lie exactly on its reconstructed joint pose.
        wp.to_torch(state.body_q)[:, 0] += 0.125
        wp.to_torch(state.body_qd)[:, 1] += 0.25
    return env


@pytest.mark.parametrize("ids", [[0], [3, 1], [0, 1, 2, 3], []])
def test_selected_fk_preserves_other_articulations_and_shared_bodies(ids):
    env = kinematic_env(articulations=2, shared_body=True)
    worlds = wp.to_torch(env.model.body_world)
    assert worlds.tolist() == [0, 0, 1, 1, 2, 2, 3, 3, -1]
    selected = torch.isin(worlds, torch.tensor(ids, dtype=torch.long))
    for state in env.states:
        before = {name: wp.to_torch(getattr(state, name)).clone() for name in ("body_q", "body_qd")}
        reference = env.model.state()
        newton.eval_fk(env.model, state.joint_q, state.joint_qd, reference)
        env.sync_selected_state(state, torch.tensor(ids, dtype=torch.long))
        for name, previous in before.items():
            actual = wp.to_torch(getattr(state, name))
            torch.testing.assert_close(actual[~selected], previous[~selected], atol=0, rtol=0)
            torch.testing.assert_close(actual[selected], wp.to_torch(getattr(reference, name))[selected], atol=0, rtol=0)


def event_config(function, params):
    return {"func": f"mdp.{function}", "params": params, "mode": "interval", "interval_range_s": [1.0, 1.0]}


def configure_events(env, kind):
    events = {}
    if kind in ("push", "mixed"):
        events["test_push"] = event_config("push_by_setting_velocity", {"velocity_range": {"x": [0.2, 0.2]}})
    if kind in ("tip", "mixed"):
        events["test_tip"] = event_config(
            "seated_backward_tip", {"prob": 1.0, "ang_vel_range": [0.3, 0.3], "lateral_frac": 0.0}
        )
    if kind in ("command", "mixed"):
        events["test_command"] = event_config("seated_head_command_jerk", {"prob": 1.0, "flip_frac": 0.0})
    recipe = SimpleNamespace(
        env=env, num_envs=env.num_envs, device=env.device, common_step_counter=0,
        posture_alpha=torch.ones(env.num_envs, device=env.device),
        spec={"events": events, "curricula": {}, "commands": {"head_pose": {"ranges": [[0.1, 0.2]] * 4}}},
    )
    pushes = PushEvents(recipe)
    for name, timer in pushes.timers.items():
        timer.fill_(1.0)
        if name == "test_tip":
            timer[:3] = 0.0
        else:
            timer[3 if name == "test_command" else 0] = 0.0
    # Only world 0 is eligible for the tip, although three timers have expired.
    env.commands[:, 0] = 1.0
    env.commands[1, 0] = 0.0
    recipe.posture_alpha[2] = 0.0
    return pushes


@pytest.mark.parametrize("kind", ["push", "tip", "command", "mixed"])
def test_events_only_project_worlds_whose_physics_changed(kind):
    env = kinematic_env()
    pushes = configure_events(env, kind)
    state = env.states[0]
    before = {name: wp.to_torch(getattr(state, name)).clone() for name in ("body_q", "body_qd", "joint_qd")}
    pushes.apply()
    untouched = slice(None) if kind == "command" else slice(1, None)
    for name, previous in before.items():
        actual = wp.to_torch(getattr(state, name))
        if name == "joint_qd":
            actual, previous = actual.reshape(4, -1), previous.reshape(4, -1)
        torch.testing.assert_close(actual[untouched], previous[untouched], atol=0, rtol=0)
    velocity = env._views()[1]
    if kind in ("push", "mixed"):
        assert velocity[0, 0].item() == pytest.approx(0.2)
    if kind in ("tip", "mixed"):
        assert velocity[0, 4].item() == pytest.approx(-0.3)
    if kind in ("command", "mixed"):
        assert ((env.commands[3, 3:7] >= 0.1) & (env.commands[3, 3:7] <= 0.2)).all()
    assert torch.all(pushes.timers.get("test_tip", torch.ones(3))[:3] > 0)


def physics_snapshot(env, *, survivors):
    env._sync()
    result = {}
    selected_world = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    selected_world[survivors] = True
    body_world = wp.to_torch(env.model.body_world).long()
    body_mask = (body_world >= 0) & selected_world[body_world.clamp_min(0)]
    particle_world = wp.to_torch(env.model.particle_world).long()
    particle_mask = (particle_world >= 0) & selected_world[particle_world.clamp_min(0)]
    for index, state in enumerate(env.states):
        for name in ("body_q", "body_qd"):
            result[f"buffer{index}/{name}"] = wp.to_torch(getattr(state, name))[body_mask].clone()
        for name in ("particle_q", "particle_qd"):
            values = getattr(state, name)
            if values is not None:
                result[f"buffer{index}/{name}"] = wp.to_torch(values)[particle_mask].clone()
        for name in ("joint_q", "joint_qd"):
            result[f"buffer{index}/{name}"] = wp.to_torch(getattr(state, name)).reshape(env.num_envs, -1)[survivors].clone()
    if env.backend == "synreal":
        drive_bodies = wp.to_torch(env.solver._joint_drive_body_indices).long()
        parent_world = body_world[drive_bodies[:, 0].clamp_min(0)]
        child_world = body_world[drive_bodies[:, 1]]
        drive_mask = (child_world >= 0) & selected_world[child_world.clamp_min(0)]
        has_parent = drive_bodies[:, 0] >= 0
        drive_mask |= has_parent & (parent_world >= 0) & selected_world[parent_world.clamp_min(0)]
        for name in ("_joint_drive_feedback_angle_initialized", "_joint_drive_feedback_previous_angles"):
            result[name] = wp.to_torch(getattr(env.solver, name))[drive_mask].clone()
    return result


@pytest.mark.skipif(
    os.environ.get("MICRODUCK_GPU_TESTS") != "1" or not torch.cuda.is_available(),
    reason="Set MICRODUCK_GPU_TESTS=1 on a CUDA host; SynReal also requires login",
)
@pytest.mark.parametrize("backend", ["mujoco", "synreal"])
@pytest.mark.parametrize("operation", ["reset", "push", "tip"])
@pytest.mark.parametrize("variant", ["standard", "history"])
def test_partial_events_preserve_integrated_state_and_drive_history(backend, operation, variant):
    from examples.microduck.common import create_env, setup

    args, _, _ = setup("train", ["--solver", backend, "--num-envs", "4", "--device", "cuda:0",
                               "--task", "velocity", "--policy-variant", variant])
    env = create_env(args)
    try:
        env.official.pushes.events = {}
        for _ in range(20):
            env.step(torch.zeros_like(env.actions))
        # Freeze at an event boundary; only the explicit edit below may change state.
        env._sync()
        newton.eval_ik(env.model, env.states[0], env.states[0].joint_q, env.states[0].joint_qd)
        before = physics_snapshot(env, survivors=slice(1, None))
        history = env.official.observations.history
        history_before = history.frames[1:].clone() if history is not None else None
        current_before = history.current[1:].clone() if history is not None else None
        selected_before = physics_snapshot(env, survivors=slice(0, 1))
        if backend == "synreal":
            assert torch.all(before["_joint_drive_feedback_angle_initialized"] == 1)
        if operation == "reset":
            env._reset(torch.tensor([0], device=env.device))
        else:
            configure_events(env, operation).apply()
        after = physics_snapshot(env, survivors=slice(1, None))
        selected_after = physics_snapshot(env, survivors=slice(0, 1))
        changed = [name for name in before if not torch.equal(before[name], after[name])]
        assert not changed, f"{operation} changed unselected worlds: {changed}"
        if history is not None:
            torch.testing.assert_close(history.frames[1:], history_before, atol=0, rtol=0)
            torch.testing.assert_close(history.current[1:], current_before, atol=0, rtol=0)
        assert any(not torch.equal(value, selected_after[name]) for name, value in selected_before.items())
        if backend == "synreal":
            assert torch.all(selected_after["_joint_drive_feedback_angle_initialized"] == 0)
        observations, reward, _, _ = env.step(torch.zeros_like(env.actions))
        assert torch.isfinite(observations).all() and torch.isfinite(reward).all()
    finally:
        env.close()
