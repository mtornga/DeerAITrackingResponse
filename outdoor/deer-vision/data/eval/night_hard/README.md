## Night Hard Eval Pack

- **Clip(s)**: `clips/nighthard1.mp4` (copied from `outdoor/eval_clips/` on 2025-??). Keep this clip immutable; additional hard night clips go into the same folder with a short description appended to this file.
- **Status**: Frames + YOLO labels TBD. Extract approximately 150–300 frames focusing on IR sequences with partial occlusions. Once labeled, freeze the `images/` + `labels/` contents and never edit in place—clone to a new pack if changes are required.
- **Notes**: Expect very low ambient light and small targets; consider tiling (`scripts/tile_images.py`) before training to improve recall.
