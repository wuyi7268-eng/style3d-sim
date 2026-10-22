"""Official ONNX replay using the original example's model, controller and captured physics."""

import time

import warp as wp

from examples.example_robot_microduck import Example


def run_walk(args):
    from newton.viewer import ViewerGL, ViewerNull

    wp.set_device(args.device)
    viewer = ViewerGL() if args.viewer == "gl" else ViewerNull(num_frames=args.num_frames)
    # Use the original constructor and step verbatim: no PD, timestep or solver changes for these weights.
    original_args = Example.create_parser().parse_args([])
    for name in ("solver", "device", "force_login", "disable_cuda_graph", "command"):
        setattr(original_args, name, getattr(args, name))
    original_args.incremental_command = True
    frames = 0
    warmup = min(20, args.num_frames // 4)
    try:
        example = Example(viewer, original_args)
        print(f"Official walk.onnx: solver={args.solver}, graph={example.graph is not None}, physics=original example")
        print("Press I/K: forward/back; J/L: lateral; U/O: turn; P: reset pose. Release keeps the velocity command.")
        previous_command = None
        measured_start = time.monotonic()
        for _ in range(args.num_frames):
            if not viewer.is_running():
                break
            if viewer.is_paused():
                example.render()
                continue
            if frames == warmup:
                measured_start = time.monotonic()
            example.step()
            # The policy observation contains the exact command used for this step, including viewer keys.
            command = tuple(float(x) for x in example.policy.observation[0, 48:51])
            if command != previous_command:
                print(f"Velocity command: vx={command[0]:.2f}, vy={command[1]:.2f} m/s, yaw={command[2]:.2f} rad/s")
                previous_command = command
            example.render()
            frames += 1
        elapsed = time.monotonic() - measured_start
        measured_frames = max(0, frames - warmup)
        if measured_frames:
            fps = measured_frames / elapsed
            print(f"Playback: {fps:.1f} FPS, simulation/wall={fps * example.frame_dt:.2f}x")
        print(f"Official policy playback complete: frames={frames}")
    finally:
        viewer.close()
    return 0
