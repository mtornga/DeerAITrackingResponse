# DeerVision Detector Pipeline Notes

Rolling log of issues, workarounds, and reminders encountered while executing the detector plan.

## 2025-??-?? — LabelImg CLI expectations

- **Issue**: Running `labelImg images_dir labels_dir --auto-save --save-format YOLO` fails because the PyPI build (1.8.x) does not support those flags.
- **Fix**: Launch without extra switches and toggle “Auto Save Mode” / “YOLO” inside the GUI; or upgrade to a fork that exposes the arguments.

- **Issue**: Even without flags, `labelImg` crashed with `IsADirectoryError` because the second positional argument is treated as the predefined class file, not the save directory.
- **Fix**: Provide all three positional arguments: `labelImg <images_dir> <class_file> <save_dir>`. Added `outdoor/deer-vision/configs/classes.txt` containing the YOLO class list (deer, unknown_mammal) so LabelImg starts cleanly.

- **Issue**: Crashes were triggered because the save directory was mistyped as `outdoor-deer-vision/...` (dash instead of slash), so LabelImg tried to write into a non-existent path; later crashes (`TypeError` in Qt scroll/paint) stem from the latest PyPI build + PyQt6.
- **Fix**: Double-check the save path (should be `outdoor/deer-vision/...`). For stability, pin LabelImg to the last PyQt5 build: `pip install --force-reinstall "labelImg==1.8.6" "PyQt5<5.16"` after activating `.venv`. Launch with `labelImg <images_dir> <class_file> <save_dir>`, then use the “Open Dir” button if the dataset doesn’t populate automatically.

- **Issue**: The “Box Labels” dropdown stayed empty/only showed `difficult` because the packaged `predefined_classes.txt` ships empty on macOS wheels, and the CLI path argument isn’t honored consistently.
- **Fix**: Inject our class list into the site package: wrote `.venv/lib/python3.12/site-packages/labelImg/data/predefined_classes.txt` with `deer`, `unknown_mammal`. This guarantees the dropdown is pre-populated even if LabelImg ignores the CLI override. (Keep `outdoor/deer-vision/configs/classes.txt` in sync for future tooling.)

- **Issue**: Even after patching the PyQt5 build, LabelImg continued to throw paint-event crashes and path errors, making it too fragile for production labeling.
- **Decision**: Pivot to CVAT for the outdoor eval pipeline; reserve LabelImg for quick one-off fixes only.

## 2025-??-?? — CVAT flow

- Upload `outdoor/deer-vision/data/eval/night_hard/clips/nighthard1.mp4` to CVAT.
- Label classes: `deer`, `unknown_animal`, `person`, `apriltag`.
- Export **YOLO 1.1 (Darknet)** and store as `outdoor/deer-vision/data/eval/night_hard/clips/nighthard1.zip`.
- Unzip into `outdoor/deer-vision/data/eval/night_hard/cvat_export/` so `obj.names`, `train.txt`, and `obj_train_data/*.txt` are available.
- Re-run `scripts/extract_frames.py --stride 1 --max-per-clip 0` so every referenced frame (e.g., `frame_000123.jpg`) exists under `data/eval/night_hard/images/nighthard1/`.
- Copy YOLO labels into place:  
  `cp data/eval/night_hard/cvat_export/obj_train_data/*.txt data/eval/night_hard/labels/nighthard1/`
- QC command:  
  `python scripts/qc_dataset.py --labels data/eval/night_hard/labels/nighthard1 --images data/eval/night_hard/images/nighthard1 --classes 0 1 2 3 --report data/eval/night_hard/qc_nighthard1.json`

## 2025-11-11 — CVAT server + Ubuntu GPU pipeline

- **Issue**: Local CVAT refused to ingest the 4K MP4 (`video codec + resolution` error) and LabelImg kept crashing.
- **Fix**: Run CVAT 2.13.0 on the basement RTX 3080 box. Install Docker (`sudo apt install docker.io`), clone `https://github.com/opencv/cvat`, set `CVAT_HOST=192.168.68.71` in `~/cvat/.env`, and launch with `sudo docker compose up -d`. Access at `http://192.168.68.71:8080` after creating a Django superuser inside `cvat_server`.
- **Data ingestion**: Export YOLO labels as `nighthard1ubuntuyolo.zip`, unzip to `data/eval/night_hard/cvat_export/obj_train_data/nighthard1*/`, copy the `.txt` files back into `data/eval/night_hard/labels/nighthard1/`, and rerun QC.
- **GPU training loop**: `rsync` the repo to `~/projects/deer-vision` on the server, create `.venv`, `pip install -r constraints.txt`, then `make train` / `make eval`. Training now lands in `models/yolov8n_det_v01/` with CUDA, so latency drops to ~1.9 ms and FP/min can be tuned via the evaluation confidence.
- **Eval confidence**: Quality gates now run with `CONF=0.4` (Makefile default) because the night_hard pack is only ~0.2 minutes long; lower thresholds caused dozens of false positives per minute despite only a handful of actual FP boxes. With `conf=0.4` we still get mAP@0.5≈0.998, recall≈0.998, and FP/min = 0.0.

## 2025-11-12 — CVAT share + automated pruning + daytime easy clip

- **Connected share mismatch**: CVAT’s Docker volume points at `/home/mtornga/projects/cvat-share`, but the helper script initially wrote to `~/projects/deer-vision/cvat-share`, so the UI showed an empty share picker.
  - **Fix**: Move existing dated folders into `/home/mtornga/projects/cvat-share` and set `CVAT_SHARE_ROOT=/home/mtornga/projects/cvat-share` in `.env`.
  - `scripts/cvat_share_clip_help.py` now defaults to `$CVAT_SHARE_ROOT`; always run it from the server after `source .venv/bin/activate` so new frame zips/MP4s land directly inside the mounted volume (no manual copying).
- **Frame packaging caution**: The helper rescales frames to 1280 px wide before zipping. That kept CVAT uploads light, but the newest `PersonDayTimeEasy1.zip` looked blurry during QC because it was downsampled once on macOS and again in the helper.
  - **Action**: For daytime clips that need fine-grained apriltag/deer edges, either disable downscaling (edit the helper’s FFmpeg filter to `scale=-1:-1`) or export native-resolution JPEGs before zipping so the labelers have full fidelity.
- **Automatic pruning**: Added `scripts/prune_segments_without_events.py`, which deletes any segment/analysis/detections that never triggered a MegaDetector event (but skips files still waiting on detections). Installed cron on the RTX 3080 host and scheduled it every 15 minutes:
  ```
  */15 * * * * cd /home/mtornga/projects/deer-vision && /home/mtornga/projects/deer-vision/.venv/bin/python scripts/prune_segments_without_events.py >> /home/mtornga/projects/deer-vision/logs/prune_segments.log 2>&1
  ```
  This keeps `/` under ~85 % usage so CVAT stays responsive.
- **QC/training reminders**:
  - For new eval packs (e.g., `data/eval/daytime_easy/PersonDayTimeEasy1.zip`) run the full loop: unzip → copy YOLO txts → `qc_dataset.py` → `make train`/`make eval` with the updated dataset manifest.
  - When a clip can’t be transferred straight to the server, stash it under `outdoor/deer-vision/data/eval/<tier>/` on macOS, then rsync/`scp` the folder to `/home/mtornga/projects/deer-vision/data/eval/<tier>/` before kicking off QC so the GPU box always trains on the freshest annotations.

## 2025-11-12 — Night hard + mixed weather ingestion workflow

- **Clip prep checklist** (repeat for every CVAT export):
  1. Copy the raw MP4 into the matching eval pack folder (e.g., `outdoor/deer-vision/data/eval/night_hard/clips/segment_055612.mp4`).
  2. Extract every frame at the original resolution with FFmpeg:  
     `ffmpeg -hide_banner -loglevel error -y -i clips/<clip>.mp4 -start_number 0 images/<clip>/frame_%06d.jpg`
  3. If CVAT exported YOLO 1.1, unzip and copy `obj_train_data/*.txt` into `labels/<clip>/`.  
     If CVAT exported XML (video mode), run `python scripts/cvat_xml_to_yolo.py annotations.xml --output labels/<clip>/` to convert boxes.
  4. Ensure every `.txt` has a partner JPG; if CVAT labeled more frames than FFmpeg emitted, duplicate the final frame (e.g., `cp frame_001001.jpg frame_001002.jpg`).
  5. Run QC over the entire pack:  
     `python scripts/qc_dataset.py --labels data/eval/<pack>/labels --images data/eval/<pack>/images --classes 0 1 2 3 --report data/eval/<pack>/qc_<pack>_<date>.json`
  6. Update `data/eval/<pack>/README.md` + `data/eval/manifest.json` with the new clip names and statuses.
  7. `rsync -av --delete data/eval/<pack>/ 192.168.68.71:/home/mtornga/projects/deer-vision/data/eval/<pack>/` to keep the GPU host in sync.

- **Night hard status**: Clips now include `nighthard1`, `segment_055549`, `segment_055612` (native 4K via XML→YOLO conversion), and `segment_092645` (~1 000 frames). QC report lives at `data/eval/night_hard/qc_night_hard_2025-11-12.json`.

- **Mixed weather status**: Added `segment_092708` + `segment_092754` rain/fog clips with native frames and YOLO labels. QC at `data/eval/mixed_weather/qc_mixed_weather_2025-11-12.json`. Pack is marked “labeled” in the manifest and mirrored onto the RTX 3080.

- **GPU pipeline reminder**: After updating eval packs locally, always `rsync` the entire `outdoor/deer-vision/` tree to `~/projects/deer-vision` on the Ubuntu box, then run training/eval with `python scripts/train.py --device 0 --name <run>` and `python scripts/evaluate.py --device 0 --model models/<run>/weights/best.pt --conf 0.4`. Keep training/eval in tmux (`tmux new -s train '...') so long-running CUDA jobs survive SSH drops.

Keep this file updated with future quirks (e.g., DVC remote hiccups, Ultralytics version pinning notes, etc.).
