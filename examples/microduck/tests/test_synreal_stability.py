"""Opt-in GPU regression against a real VelStand pre-explosion state."""

import json
import os
from pathlib import Path

import pytest


@pytest.mark.skipif(os.environ.get("MICRODUCK_GPU_TESTS") != "1", reason="Requires SynReal login and CUDA")
def test_fallen_robot_does_not_amplify_angular_velocity():
    import newton
    import torch
    import warp as wp

    from examples.microduck.common import create_env, setup
    from examples.microduck.policy_spec import EFFORT, KD, KP
    from examples.microduck.stability import unstable_worlds

    args, _, _ = setup("train", ["--solver", "synreal", "--task", "velstand", "--num-envs", "1"])
    env = create_env(args)
    data = json.loads((Path(__file__).parent / "fixtures/velstand_pre_explosion.json").read_text())
    try:
        # Native warm-start/proxy caches were not captured. Reconstruct both buffers
        # from rigid state; the original Cayley path still diverges within two steps.
        for state in env.states:
            for target, source in (("body_q", "previous_body_q"), ("body_qd", "previous_body_qd")):
                wp.to_torch(getattr(state, target)).copy_(torch.tensor(data[source][0], device=env.device))
            env._sync()
            newton.eval_ik(env.model, state, state.joint_q, state.joint_qd)
            env.solver.sync_affine_body_state(state)
        desired = env.home + torch.tensor(data["actions"], device=env.device)
        for step in range(100):
            env._sync()
            newton.eval_ik(env.model, env.states[0], env.states[0].joint_q, env.states[0].joint_qd)
            env._sync()
            q, qd, _, _ = env._views()
            torque = (KP * (desired - q[:, env.qi]) - KD * qd[:, env.qdi]).clamp(-EFFORT, EFFORT)
            targets = wp.to_torch(env.control.joint_target_q).reshape(1, -1)
            targets[:, env.qdi] = q[:, env.qi] + (torque + KD * qd[:, env.qdi]) / KP
            env._sync()
            env.states[0].clear_forces()
            env.collision.collide(env.states[0], env.contacts)
            env.solver.step(*env.states, env.control, env.contacts, env.physics_dt)
            env.states.reverse()
            env._sync()
            pose = wp.to_torch(env.states[0].body_q).reshape(1, -1, 7)
            velocity = wp.to_torch(env.states[0].body_qd).reshape(1, -1, 6)
            assert not unstable_worlds(pose, velocity).any(), f"Physics diverged at substep {step}"
        assert velocity[..., 3:].abs().max() < 100.0
    finally:
        env.close()
