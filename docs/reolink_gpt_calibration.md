# Reolink GPT Calibration & Control Notes

This document captures the current process we use to align the Reolink E1 camera, the GPT‑5‑mini pose estimates, and the Cutebot controller. It describes the calibration assets, the math behind the corrections, and the next steps required to make the loop fully reliable.

## Overview

- **Camera & GPT**: We grab a single frame from the Reolink E1 RTSP stream and send it to GPT‑5‑mini, which returns nose coordinates, corner pixels, and heading.
- **Cutebot firmware**: Streams magnetometer headings and accepts signed wheel PWM via BLE.
- **Dashboard**: A Rich-based console (four panes) shows live magnetometer headings, GPT nose positions, controller commands, and logs.

Accurate navigation depends on:
1. Mapping GPT’s output to true board inches.
2. Ensuring magnetometer headings match the laminated board markers.

## Required Calibration Artifacts

Capture five reference poses with the Cutebot nose at known board coordinates. Each pose produces a JPEG and JSON via `perception/reolink_gpt_snapshot.py`:

```
calibration/reolink_gpt/calib_00.{jpg,json}    # (0", 0")
calibration/reolink_gpt/calib_3000.{jpg,json} # (30", 0")
calibration/reolink_gpt/calib_0024.{jpg,json} # (0", 24")
calibration/reolink_gpt/calib_3024.{jpg,json} # (30", 24")
calibration/reolink_gpt/calib_1512.{jpg,json} # (15", 12")
```

Each JSON includes:
- `cutebot_nose_inches_projected`: GPT’s raw inch estimate (distorted when the board is rotated).
- `board_corners_pixels`: pixel coordinates of the board corners.
- `cutebot_nose_pixels`: the pixel location of the nose.
- `arrow_heading_degrees`: GPT’s heading estimate from the green arrow.

## Theory

### GPT Inch Corrections

When the Reolink camera or board moves, GPT’s projected inches no longer match our board axes—typically an axis flip plus translation. We fit an affine map:

```
[real_x, real_y]^T = A * [gpt_x, gpt_y]^T + b
```

Using the five calibration JSON pairs we solve for `A` and `b` via least squares and store them in `calibration/reolink_gpt/transform.json`.

### Magnetometer vs heading cards

The Cutebot’s firmware now clamps PWM to −100…100. It streams headings at roughly 5 Hz. If a calibration is required, the `read_heading.py` helper collects samples so we can align raw compass readings to the laminated markers (0°, 90°, 180°, 270°).

### Control loop

1. Capture a snapshot → JSON via GPT.
2. Convert GPT’s inches with `transform.json` to true board coordinates.
3. Read the micro:bit heading (`CutebotUARTSession.request_heading`).
4. Compute `(dx, dy)` to the target pose and rotate/drive accordingly.
5. After the position tolerance is satisfied, pivot until the magnetometer heading matches the target heading within tolerance.

## Current Steps

1. **Collect snapshots**: Position the Cutebot at the five reference coordinates listed above and run `perception/reolink_gpt_snapshot.py` for each.
2. **Generate `transform.json`** (prefer the helper script):
   ```bash
   # Run inside the project venv so numpy is available.
   PYTHONPATH=. python scripts/reolink_gpt_fit_transform.py \
     --output calibration/reolink_gpt/transform.json
   ```
   The script auto-detects the five `calib_XXXX.json` files, solves the least-squares fit, and prints per-sample errors plus the overall RMSE. If you collect additional poses, provide them with repeated `--calibration X,Y:/path/to/sample.json` flags before re-running.

3. **Run the vector driver (with dashboard)**:
   ```bash
   PYTHONPATH=. python scripts/reolink_gpt_vector_drive.py \
     --target-x 27 --target-y 12 --target-heading 180 \
     --heading-calibration none \
     --transform calibration/reolink_gpt/transform.json \
     --iterations 4 --forward-speed 25 --pivot-speed 20 \
     --drive-duration-ms 420 --pivot-duration-ms 160 \
     --position-tolerance 0.8 --heading-tolerance 10 \
     --settle-sec 0.7 --pose-delay 1.0 --pose-retries 4 \
     --min-confidence 0.55 --verbose
   ```
   The first logged pose should now be close to the actual start; the controller will nudge forward and pivot only to align the requested heading.

## Next Steps

1. **Keep `transform.json` fresh**—rerun the fitter whenever the camera or board shifts, and watch the RMSE the script reports.
2. **Verify accuracy** across a few more snapshots—errors should stay within ~1″.
3. **Run the vector driver** without `--lock-axis`: once the mapping is consistent, `dx` and `dy` will reflect the real vector to the target.
4. Optionally **reject GPT headings** when they disagree with the magnetometer by >30°; fall back to vision only if both sensors align.
5. Once reliable, **plug the Reolink tracker into `CutebotFeedbackLoop`**, allowing the existing CSV/YOLO or the GPT tracker to be swapped via command-line flag.
