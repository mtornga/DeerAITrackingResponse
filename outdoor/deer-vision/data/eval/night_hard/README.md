## Night Hard Eval Pack

- **Clip(s)**:
  - `clips/nighthard1.mp4` (copied from `outdoor/eval_clips/` on 2025-??). Keep this clip immutable; additional hard night clips go into the same folder with a short description appended to this file.
  - `clips/segment_055549.mp4` (copied from `outdoor/eval_clips/` on 2025-11-12). Frames extracted at 1280 px width live in `images/segment_055549/` with YOLO labels under `labels/segment_055549/`.
  - `clips/segment_055612.mp4` (copied from `outdoor/eval_clips/` on 2025-11-12). Frames preserved at native 4096×1152 resolution in `images/segment_055612/`; labels were converted from the CVAT XML export via `scripts/cvat_xml_to_yolo.py` into `labels/segment_055612/`.
  - `clips/segment_092645.mp4` (copied from `outdoor/eval_clips/` on 2025-11-12). High-resolution clip with ~1 000 frames stored in `images/segment_092645/`; YOLO labels under `labels/segment_092645/`.
- **Status**: Frames + YOLO labels now available for `nighthard1`, `segment_055549`, `segment_055612`, and `segment_092645`. Once labeled, freeze the `images/` + `labels/` contents and never edit in place—clone to a new pack if changes are required.
- **Notes**: Expect very low ambient light and small targets; consider tiling (`scripts/tile_images.py`) before training to improve recall.
