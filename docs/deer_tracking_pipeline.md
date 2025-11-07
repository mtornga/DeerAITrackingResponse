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

```bash
source .venv/bin/activate
python scripts/track_deer_kf.py \
  --frames tmp/megadetector_frames_Backyard-00-064458-064505 \
  --detections runs/megadetector/Backyard-00-064458-064505.json \
  --descriptor models/MegaDescriptor-T-CNN-288/pytorch_model.bin \
  --output-json runs/megadetector/Backyard-00-064458-064505_tracks_kf.json \
  --output-video runs/megadetector/Backyard-00-064458-064505_tracked_kf.mp4
```

(*`scripts/track_deer_kf.py` contains the exact logic described above — feel free to adapt it for more animals or for swapping in the L‑384 descriptor.*)

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

## 8. Tips & Extensions

- Want more than two deer? Increase the number of detections you keep per frame and extend the label list (`deer3`, `deer4`, …). The tracker already handles more entries.
- Need stronger ID stability? Swap in the Swin-L descriptor (`BVRA/MegaDescriptor-L-384`), but expect a slower runtime and ensure crops are at least 384×384.
- Tracking other species: adjust the MegaDetector filter (`category == '1'` is “animal”). For multi-species tracking, keep additional lists keyed by detector class.
- Batch mode: the tracker script is stateless; run it per video. If you want to process multiple videos in one go, wrap it in a loop and change the output prefixes.

With these steps documented, a new developer should be able to rebuild the entire pipeline (detection → appearance embedding → Kalman/assignment → annotated video) for any new clip. Let us know if you need reusable scripts extracted from the notebook cells — we can promote them into `scripts/` for easier reuse. !*** End Patch
