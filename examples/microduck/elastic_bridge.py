"""Elastic-bridge Velocity entry point with an independent recipe and training configuration."""

import argparse
import sys
import time
from pathlib import Path


def preview(argv):
    import warp as wp
    from newton.viewer import ViewerGL, ViewerNull

    from .env import MicroDuckEnv

    parser = argparse.ArgumentParser(description="Static elastic bridge preview; no login or policy required")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--viewer", choices=("gl", "null"), default="gl")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--num-frames", type=int, default=30000)
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args(argv)
    if args.num_frames < 1:
        parser.error("--num-frames must be positive")
    if args.screenshot and args.viewer != "gl":
        parser.error("--screenshot requires --viewer gl")
    wp.set_device(args.device)
    viewer = ViewerGL(width=1440, height=900, headless=args.headless) if args.viewer == "gl" else ViewerNull(args.num_frames)
    env, frame = None, None
    try:
        env = MicroDuckEnv(
            solver="synreal", num_envs=1, device=args.device, viewer=viewer, terrain="elastic_bridge",
            interactive=True, controller="home", preview=True, randomize_foot_friction=False,
        )
        viewer.set_model(env.model)
        env.terrain.configure_viewer(viewer)
        print("Elastic bridge: inlet platform spawn, fixed short ends, free sides. "
              "STATIC PREVIEW (no physics or training).", flush=True)
        for _ in range(args.num_frames):
            if not viewer.is_running():
                break
            env.render()
            if args.screenshot:
                frame = viewer.get_frame(target_image=frame, render_ui=False)
            if args.viewer == "gl" and not args.headless:
                time.sleep(1 / 60)
        if frame is not None:
            from PIL import Image

            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(frame.numpy()).save(args.screenshot)
    finally:
        if env is not None:
            env.close()
        else:
            viewer.close()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preview", "train", "play"))
    args = parser.parse_args(argv[:1])
    if args.mode == "preview":
        return preview(argv[1:])
    # The shared entry points own checkpoint validation, PPO, PD and manual replay resets.
    options = argv[1:]
    defaults = {"--solver": "synreal", "--task": "velocity", "--terrain": "elastic_bridge"}
    for option, expected in defaults.items():
        for index, item in enumerate(options):
            if item.split("=")[0] == option:
                value = item.split("=", 1)[1] if "=" in item else options[index + 1] if index + 1 < len(options) else None
                if value != expected:
                    parser.error(f"elastic_bridge requires {option} {expected}; use the shared entry point for other tasks")
    if args.mode == "train":
        defaults["--num-envs"] = "32"
    for option, value in defaults.items():
        if not any(item.split("=")[0] == option for item in options):
            options.extend((option, value))
    if args.mode == "train":
        from .train import main as run
    else:
        from .play import main as run
    return run(options)


if __name__ == "__main__":
    main()
