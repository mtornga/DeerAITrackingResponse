# Deer Tracking Pipeline Guide

This document explains how to reproduce the two-deer tracking demo we just built and how to adapt it for future clips. The workflow is modular, so you can re-run individual pieces (detection, cropping, tracking) as needed.

---

## 1. Environment & Dependencies

1. Start from the repo root.
2. (One time) create/activate the Python 3.12 virtualenv mentioned in the repo README and install the base stack:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --no-cache-dir --force-reinstall -r constraints.txt
   ```
3. Install the higher-level tooling (YOLO, HuggingFace, timm, etc.):
   ```bash
   pip install --no-cache-dir -r requirements.txt
   ```
4. GPU is optional. The pipeline runs on CPU, albeit slower for the descriptor and YOLO steps.

---

## 2. Stage: Detection (MegaDetector v5)

We use the upstream `process_video.py` script out of the MegaDetector repo (cloned under `tmp/MegaDetector/`) with the MD v5a weights (`models/md_v5a.0.0.pt`).

```bash
source .venv/bin/activate
PYTHONPATH=tmp/MegaDetector:tmp/ai4eutils:tmp/yolov5 \
python tmp/MegaDetector/detection/process_video.py \
  models/md_v5a.0.0.pt outdoor/Backyard-00-064458-064505.mp4 \
  --output_json_file runs/megadetector/Backyard-00-064458-064505.json \
  --render_output_video \
  --output_video_file runs/megadetector/Backyard-00-064458-064505_annotated.mp4 \
  --json_confidence_threshold 0.2 \
  --rendering_confidence_threshold 0.5 \
  --frame_folder tmp/megadetector_frames_Backyard-00-064458-064505 \
  --keep_extracted_frames --reuse_frames_if_available
```

Outputs:
- `runs/megadetector/<video>.json` – detections per frame (YOLO-normalized bbox, confidence, class).
- `runs/megadetector/<video>_annotated.mp4` – optional overlay video.
- `tmp/megadetector_frames_<video>/` – extracted frames (used by later stages).

You can re-run this command on any other clip by substituting the input/output paths.

---

## 3. Stage: (Optional) Cropping

If you need crops for classification or other downstream experiments:

```bash
source .venv/bin/activate
PYTHONPATH=tmp/MegaDetector:tmp/ai4eutils \
python tmp/MegaDetector/classification/crop_detections.py \
  runs/megadetector/Backyard-00-064458-064505.json \
  tmp/Nest2025-11-05_21-23-30_crops \
  --images-dir tmp/megadetector_frames_Backyard-00-064458-064505 \
  --threshold 0.2 --square-crops --logdir tmp/Nest2025-11-05_21-23-30_crops
```

This step is not required for tracking but is handy for species classification or manual QA.

---

## 4. Stage: Appearance Descriptor (MegaDescriptor T‑CNN‑288)

We use the BVRA MegaDescriptor to turn each detection into a 1,536‑D appearance embedding.

1. Download the weights from Hugging Face (one time):
   ```bash
   source .venv/bin/activate
   python - <<'PY'
   from huggingface_hub import snapshot_download
   snapshot_download(
       repo_id="BVRA/MegaDescriptor-T-CNN-288",
       local_dir="models/MegaDescriptor-T-CNN-288",
       local_dir_use_symlinks=False
   )
   PY
   ```
2. Load the EfficientNet-B3 backbone with timm and feed 288×288 crops:
   ```python
   import timm, torch
   from torchvision import transforms

   state = torch.load("models/MegaDescriptor-T-CNN-288/pytorch_model.bin", map_location="cpu")["model"]
   descriptor = timm.create_model("efficientnet_b3", pretrained=False, num_classes=0)
   descriptor.load_state_dict(state)
   descriptor.eval()

   preproc = transforms.Compose([
       transforms.Resize((288, 288)),
       transforms.ToTensor(),
       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
   ])
   ```

Given a `PIL.Image` crop, apply the transform, run `descriptor(tensor.unsqueeze(0))`, and L2 normalize the result. This vector is what we compare between frames.

**Tip:** A second descriptor (`BVRA/MegaDescriptor-L-384`, Swin-L) is also available. It provides better discrimination at the cost of speed and a 384 px crop requirement. Swap it in only when you have a GPU and need maximum re-identification accuracy.

---

## 5. Stage: Tracking with Kalman + Embedding Matching

We keep two tracks (`deer1`, `deer2`). Each track maintains:
- A state vector `[cx, cy, w, h, vx, vy, vw, vh]` using a constant-velocity Kalman filter.
- The last appearance embedding (for cosine matching).

Per frame:
1. Predict each track forward with the Kalman filter.
2. Take the top-N (we use 2) MegaDetector boxes, embed them, and compute a cost matrix:
   - `cost = α * (1 – cosine_similarity) + (1 – α) * (1 – IoU)` with α = 0.7.
3. Filter by gating distance (Mahalanobis) so wildly inconsistent boxes don’t match.
4. Solve the assignment with the Hungarian algorithm (scipy).
5. Update matched tracks (Kalman + new embedding); mark unmatched tracks as “missed”; initialize tracks if they were uninitialized and a deer detection was left over.
6. Cap `missed` to prevent label churn (we drop/reinit after ~8 missed frames).

Finally, render the tracks on top of the raw frames:

There is now a reusable CLI under `scripts/track_megadetector_kf.py`:

```bash
source .venv/bin/activate
python scripts/track_megadetector_kf.py \
  --frames-dir tmp/megadetector_frames_segment_092731 \
  --detections-json runs/megadetector/segment_092731.json \
  --class-map 1=deer \
  --conf-threshold 0.2 \
  --max-detections-per-frame 4 \
  --max-tracks-per-class 4 \
  --max-missed 30 \
  --output-json runs/megadetector/segment_092731_tracks_kf.json \
  --output-video runs/megadetector/segment_092731_tracked_kf.mp4
```

Key flags:
- `--class-map`: map MegaDetector `category_id` → label prefix (repeat per class).
- `--max-detections-per-frame` / `--max-tracks-per-class`: optional caps if you expect only a few animals at once.
- `--descriptor-weights`: swap in the Swin-L weights if needed.

Outputs:
- `runs/megadetector/<video>_tracks_kf.json`: frame→[{label, bbox, conf}] mapping.
- `runs/megadetector/<video>_tracked_kf.mp4`: annotated video with consistent IDs.

---

## 6. Stage: CameraTrapDetectoR (Optional Verification)

We also experimented with CameraTrapDetectoR’s YOLOv8-based classifier (weights in `models/v3_cl_deploy_12122024/`). To run it on a folder of frames:

```bash
source .venv/bin/activate
python models/v3_cl_deploy_12122024/deploy_model_v3.py \
  models/v3_cl_deploy_12122024/weights.pt \
  tmp/ctd_v3_frames_Backyard-00-212715-212730 \
  --output_dir runs/ctd_v3_Backyard-00-212715-212730 \
  --checkpoint_frequency 1000
```

This writes `CTDv3_predictions_raw_final.csv` (boxes + labels per detection) and `CTDv3_predictions_formatted_final.csv` (aggregated counts). Use it when you need coarse species labels, but note it can confuse similar species (e.g., fox vs. mink) depending on the scene.

---

## 7. Directory Map

| Path | Description |
|------|-------------|
| `models/md_v5a.0.0.pt` | MegaDetector weights. |
| `tmp/MegaDetector/` | CameraTraps repo clone (detection scripts). |
| `tmp/ai4eutils`, `tmp/yolov5` | Helper repos required by MegaDetector scripts. |
| `tmp/megadetector_frames_<video>/` | Frame extractions (reused by later stages). |
| `runs/megadetector/*.json` | MegaDetector detections. |
| `runs/megadetector/*_annotated.mp4` | Detection overlay videos. |
| `models/MegaDescriptor-T-CNN-288/` | Appearance descriptor weights. |
| `runs/megadetector/*_tracks_kf.*` | Kalman-smoothed tracking outputs. |
| `models/v3_cl_deploy_12122024/` | CameraTrapDetectoR inference script + weights. |
| `runs/ctd_v3_*/` | YOLOv8 classifier outputs. |

---

## 8. Stage: AprilTag-Based Outdoor Calibration

We now have four AprilTags (IDs 0–3) mounted in view of the Reolink 3 POE camera. Use them to recover a metric transform between pixels and your yard map so each tracked animal gets a 2D property coordinate.

### Assets to start with
- `outdoor/calibration/apriltags_outdoor_calibration.pdf` – drawing with measured tag distances/angles from the camera origin.
- `outdoor/calibration/DayTime4AprilTags.jpg` – clean daylight still with all tags visible.
- `outdoor/calibration/BackyardMe*` JPGs – human poses at known distances for validation.
- Any additional clips where the tags stay inside the frame (capture a short burst every time you move the camera).
- `calibration/outdoor/apriltag_pixels.json` – auto-extracted pixel centers/corners for each detected AprilTag.
- `calibration/outdoor/apriltag_homography.json` (`.npy`) – the latest solved image→yard transform.
- `outdoor/calibration/BackyardMe*_foot.jpg` – helper annotations that mark the automatically sampled “hoof” pixel we used when validating the homography against the taped distances.

**Measured tag reference (from the PDF + field notes)**

| Tag ID | Range from camera | Vertical pitch (down) | Tag height above lawn | Notes |
|--------|-------------------|-----------------------|-----------------------|-------|
| `0` | 50 ft | ~5° | 31 in (2.6 ft) | Slightly below camera line, aligns with fern bed |
| `1` | 11 ft | 30° | ~4 ft | On the raised bed “ground” (bed surface is 4 ft above lawn) |
| `2` | 11 ft (to the right) | 20° | 36 in (3 ft) | Mounted on right fence rail |
| `3` | 48 ft | 30° | 5 ft | Near the back tree line |

Treat these as the authoritative distances until you remeasure. Azimuth (left/right) offsets are recorded in the PDF; Tag 2 is the only one with a pronounced horizontal offset, so be sure to transcribe its angle relative to the camera boresight before fitting the homography.
When tags were reprinted (IDs 2–5 from the official `tag36h11` family) we recorded the following usable bearings: Tag 2 ≈ 10° right, Tag 3 ≈ 80° right, Tag 4 ≈ 70° left, Tag 5 ≈ 60° left. Those are the numbers baked into the current homography.

### 8.1 Detect the AprilTags and persist pixel corners
1. Extract a representative frame (or reuse `DayTime4AprilTags.jpg`).
2. Install an AprilTag detector (the `pupil-apriltags` Python wheel works well in the existing virtualenv).
3. Run a short helper script to record each tag’s pixel-space corners:
   ```bash
   source .venv/bin/activate
   pip install --no-cache-dir pupil-apriltags opencv-python
   python - <<'PY'
   import json, cv2, pupil_apriltags as apriltag
   from pathlib import Path

   img_path = Path("outdoor/calibration/DayTime4AprilTags.jpg")
   img = cv2.imread(str(img_path))
   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   detector = apriltag.Detector(families="tag36h11")
   detections = detector.detect(gray, estimate_tag_pose=False)

   corners = {
       int(d.tag_id): {
           "center_px": d.center.tolist(),
           "corners_px": d.corners.tolist()
       } for d in detections
   }

   payload = {
       "image": str(img_path),
       "image_shape": img.shape[:2],
       "tag_pixels": corners
   }
   out_path = Path("calibration/outdoor/apriltag_pixels.json")
   out_path.parent.mkdir(parents=True, exist_ok=True)
   out_path.write_text(json.dumps(payload, indent=2))
   print(f"Wrote {out_path}")
   PY
   ```
   This file anchors every tag ID to precise pixel coordinates for the snapshot you will calibrate against.

### 8.2 Solve the homography (or full extrinsics)
1. Choose a world coordinate system for the yard. A practical choice is `(0, 0)` at the camera tripod, +X pointing to the right fence, +Y pointing “forward” into the yard.
2. Using the distances/angles recorded in `apriltags_outdoor_calibration.pdf`, convert each tag into `(x, y, z)` coordinates (in feet or meters; be consistent later).
   - Horizontal range = `range_ft * cos(elevation_rad)`, height difference = `range_ft * sin(elevation_rad)`.
   - Split horizontal range into X/Y with the measured azimuth: `x = horiz * sin(azimuth)`, `y = horiz * cos(azimuth)`.
   - Absolute tag height above lawn comes from the table above (e.g., Tag 1 sits 4 ft higher because of the raised bed); adjust `z` accordingly before calling OpenCV.
3. Match those 2D/3D world points with the pixel centers collected above. You can now either:
   - Solve a planar homography if the tags roughly share the same ground plane:
     ```python
     import numpy as np, cv2, json
     px = np.float32([tag_pixels[i]["center_px"] for i in tag_ids])
     world = np.float32([[tag_world[i]["x"], tag_world[i]["y"]] for i in tag_ids])
     H, mask = cv2.findHomography(px, world, method=cv2.RANSAC)
     ```
   - Or compute full camera extrinsics (rotation/translation) with `cv2.solvePnP` if you also model the tags’ elevation. This gives you a richer projection for sloped yards.
4. Save the resulting transform (homography matrix or R|t vectors, plus metadata) to `calibration/outdoor/apriltag_homography.json`.

**Reference implementation (already in repos):**
- `calibration/outdoor/apriltag_pixels.json` is produced by running the detector over `DayTime4AprilTags.jpg` with `pupil-apriltags` (quad_decimate=0.6, decode_sharpening=0.25). Each entry stores pixel centers + corner quads.
- `calibration/outdoor/apriltag_homography.json` stores the 3×3 matrix solved by `cv2.findHomography` using tag IDs {2,3,4,5} and world feet derived from the measured ranges/azimuths above.
- Raw matrix (for posterity):
  ```
  H = [[-0.0066586, -0.0059348, 32.327876],
       [ 0.0000474,  0.0024777, -9.024377],
       [-0.0001558, -0.0011813, 1.0]]
  ```
  Load it with `np.load('calibration/outdoor/apriltag_homography.npy')` or parse the JSON to keep downstream scripts language-agnostic.

### 8.3 Validate with the human-distance stills
1. Load each `BackyardMe*.jpg`, detect the AprilTags again, and project the annotated person’s foot position into world space using the homography.
2. Compare the projected distance against the measured distance recorded in the filename. You should stay within ~1–2 feet; otherwise revisit the tag coordinates or ensure you picked the bottom of the bounding box when projecting.
3. Document the residuals in the calibration JSON so future clips know the expected error.
   - We bootstrap the “hoof” pixel by HSV-thresholding the bright yellow tape and taking the lowest blob pixel (see `outdoor/calibration/BackyardMe*_foot.jpg`). It is good enough for quick QA, but you may switch to manual clicks if the heuristics drift under different lighting.

### 8.4 Keep the calibration current
- Re-run the above workflow any time you bump the camera or move the tags.
- Drop the new JSON next to the old one and add a `calibration_id` field (e.g., `apriltag_outdoor_2025-01-05`). Each downstream run should take that ID as an argument so you can reproduce historical projections.

## 9. Stage: Map Tracks to Property Coordinates

With a valid homography/extrinsics file, add one more post-processing pass that attaches world coordinates to every Kalman track.

### 9.1 Pick a canonical image point per detection
- Use the bottom-center of each detection box (`(x_min + x_max)/2, y_max`) because it best approximates hoof contact with the ground plane.
- If detections are angled, optionally run a lightweight pose/segmentation model to find the lowest visible pixel; store that logic in `scripts/world_projection.py` so it can evolve independently.

### 9.2 Project into the yard map
```python
import cv2, json, numpy as np

H = np.asarray(json.load(open("calibration/outdoor/apriltag_homography.json"))["homography"])
pixel_pts = np.float32(footpoints).reshape(-1, 1, 2)
world_pts = cv2.perspectiveTransform(pixel_pts, H).reshape(-1, 2)
```
- If you solved for full extrinsics, use `cv2.projectPoints`/`cv2.solvePnP` inverses instead.
- Append the resulting `(world_x, world_y)` to each detection row along with derived metrics like `range_ft = hypot(world_x, world_y)` or bearing relative to the camera axis.

### 9.3 Write the enriched dataset
- Extend `detections_world.csv` (or create `runs/world/<video>_world.csv`) with fields: `video`, `frame`, `track_id`, `species`, `world_x_ft`, `world_y_ft`, `range_ft`, `bearing_deg`, `confidence`, `calibration_id`.
- Gate out projections whose confidence is low or whose world point falls outside the convex hull of the four tags; log them for QA instead of polluting the dataset.

### 9.4 QA overlays
- Render a top-down scatter plot of `world_x/world_y` over a sketch of the yard (Matplotlib or Plotly) for each processed clip.
- Optionally back-project the world points into the original frame to verify they align with hooves—`cv2.perspectiveTransform` with the inverse homography matrix makes this round-trip check easy.
- For regression tests, we project the `BackyardMe16/32/48` foot points with the current homography and confirm they land near 16 ft, 32 ft, and 48 ft along the yard axis. Keep those JPGs in sync with future camera moves so new developers can immediately sanity-check their calibration.

## 10. Tips & Extensions

- Want more than two deer? Increase the number of detections you keep per frame and extend the label list (`deer3`, `deer4`, …). The tracker already handles more entries.
- Need stronger ID stability? Swap in the Swin-L descriptor (`BVRA/MegaDescriptor-L-384`), but expect a slower runtime and ensure crops are at least 384×384.
- Tracking other species: adjust the MegaDetector filter (`category == '1'` is “animal”). For multi-species tracking, keep additional lists keyed by detector class.
- Batch mode: the tracker script is stateless; run it per video. If you want to process multiple videos in one go, wrap it in a loop and change the output prefixes.

With these steps documented, a new developer should be able to rebuild the entire pipeline end-to-end: detection → appearance embedding → Kalman/assignment → world projection → annotated video and CSV. Let us know if you need reusable scripts extracted from the notebook cells — we can promote them into `scripts/` for easier reuse. !*** End Patch
