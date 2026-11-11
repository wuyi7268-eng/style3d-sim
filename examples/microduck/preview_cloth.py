"""Show the initial four-corner cloth layout without starting physics or training."""

import argparse

import warp as wp


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    wp.set_device(args.device)
    from newton.viewer import ViewerGL

    from .env import MicroDuckEnv

    viewer = ViewerGL(width=1440, height=900)
    env = None
    try:
        env = MicroDuckEnv(
            solver="synreal",
            num_envs=1,
            device=args.device,
            viewer=viewer,
            terrain="cloth",
            interactive=True,
            controller="home",
            preview=True,
            randomize_foot_friction=False,
        )
        viewer.set_model(env.model)
        viewer.set_camera(pos=wp.vec3(1.3, -1.6, 1.35), pitch=-28.0, yaw=132.0)
        viewer.show_ground = True
        print("Static cloth preview: physics and training are disabled. Close the window to exit.", flush=True)
        while viewer.is_running():
            env.render()
    finally:
        if env is not None:
            env.close()
        else:
            viewer.close()


if __name__ == "__main__":
    main()
