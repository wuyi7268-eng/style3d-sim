# MicroDuck soft-pad playback rendering

Record a single-environment checkpoint replay from the repository root. The recording captures the same
post-step body poses and deformable pad vertices used by playback; it does not rerun the physics in Blender.

```bash
.venv/bin/python -m examples.microduck.play --solver synreal \
  --checkpoint logs/rsl_rl/microduck_synreal/velocity_soft_4800/model_4800.pt \
  --num-envs 1 --viewer gl --command 0.35 0 0 --num-frames 1000 \
  --allow-recipe-mismatch --no-fall-reset --use-checkpoint-environment \
  --record-dir /path/to/new/recording --record-until-pad-edge

blender -b --python-exit-code 1 --python demos/microduck_soft_pad/build_render_scene.py -- \
  --record-dir /path/to/new/recording
blender -b /path/to/new/recording/scene.blend --python-exit-code 1 \
  --python demos/utils/validate_import.py -- --usd /path/to/new/recording/simulation.usdc
blender -b /path/to/new/recording/scene.blend --python-exit-code 1 \
  --python demos/utils/render_frames.py -- --animation --device OPTIX
.venv/bin/python demos/utils/encode_video.py --output-dir /path/to/new/recording \
  --output video.mp4 --fps 50 --end LAST_FRAME --contact-sheet
```

Replace `LAST_FRAME` with the inclusive last frame in `run_manifest.json`. The directory must not already exist.
`--record-until-pad-edge` stops when the root position reaches either edge of the pad, or at `--num-frames`,
whichever comes first. Manual resets and keyboard commands remain active during GL playback and are captured.
The output includes the recorded USD, trajectory, environment, checkpoint and source fingerprints, Blender project,
frame provenance, and verified video. Rendering adds materials, lights and a close tracking camera only. A 5 cm grid follows the pad surface through
rest-space UV coordinates, making compression and recovery visible without modifying the geometry. A checkpoint
with `--allow-recipe-mismatch` is a diagnostic replay with the current recipe, not a strict historical reproduction.

For current `env.py` pad settings, omit `--use-checkpoint-environment`. For a 15-second capture at 50 Hz,
use `--num-frames 750`. Optional `--heading-hold` steers a single Velocity checkpoint along its initial forward
line using simulated root pose feedback. It changes only policy yaw commands, not weights or external forces.
`--heading-max-yaw` defaults to 1 rad/s; larger limits exceed the saved training command range and require
behavioral checks. The 2026-11-06 soft45 demo used forward 0.46 and a 2 rad/s limit, with yaw commands recorded
per frame. This assisted playback is not evidence that the unassisted policy tracks a straight line.

`--heading-hold --heading-max-yaw 2 --adaptive-forward` additionally regulates measured forward progress
toward 0.10 m/s and damps measured yaw rate. The forward command starts at `--command VX`, varies within
`[VX - 0.06, VX + 0.09]`, and changes at most 0.10 m/s per second. This is a command-space assistance loop,
not a guarantee of speed tracking; current validation covers nominal VX=0.46 on the soft45 pad only.
That run advanced 0.94 m in 11.1 seconds and stopped at the pad edge, versus 0.53 m in 15 seconds for the
previous yaw-only controller. Both use ground-truth pose and exceed the original training command range.
Use `--num-frames 750 --record-until-pad-edge` to capture up to 15 seconds without continuing past the edge.

Add `--relief-lighting` when building the Blender scene to emphasize real pad indentation: a lower close camera,
grazing side light, reduced fill, stronger 5 cm grid lines, and short-range contact occlusion make surface slopes
and recovery easier to see. This changes only camera, lighting and materials; cached geometry is never displaced
or smoothed. Preview a loaded-foot frame before rendering the full animation.

Use `--mechanical-normals` to preserve robot creases above 35 degrees and apply face-area/angle weighted
split normals. This removes broad-panel shading artifacts caused by smoothing across mechanical corners;
vertex positions, topology and recorded transforms remain unchanged. Verify representative frames with
`validate_import.py` and compare panel highlights before rendering the full sequence.

Circular trampoline playback also supports the same Blender pipeline. The recorder exports the moving cloth,
robot meshes, and capsule frame at the actual policy rate; no physics runs in Blender. For example:

```bash
.venv/bin/python -m examples.microduck.jump play --checkpoint /path/to/model.pt \
  --allow-source-mismatch --num-envs 1 --viewer gl --num-frames 750 --record-dir /path/to/new/take
blender -b --python-exit-code 1 --python demos/microduck_soft_pad/build_render_scene.py -- \
  --record-dir /path/to/new/take --mechanical-normals
```

Continue with the import validation, frame rendering, and encoding commands above (`LAST_FRAME=749`, `--fps 50`).
The trampoline camera follows the duck at a fixed viewing angle. Source mismatch playback uses the current
scene and the saved policy; it does not reproduce the historical square trampoline. Recording preserves manual
reset behavior, and does not apply automatic resets or external assistance.
