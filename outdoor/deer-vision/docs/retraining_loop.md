## Weekly Retraining Loop

1. **Collect Clips**  
   - Drop new MP4 files into `data/raw/clips/` (back them up to S3 at `s3://wildlife-ugv/raw/`).  
   - Run `make frames` to extract fixed-count frames per clip.

2. **Auto-Label & Correct**  
   - Execute `make autolabel` to generate low-threshold YOLO drafts.  
   - Open LabelImg pointing at `data/interim/autolabel/images` and `.../labels` to correct boxes.

3. **QC & Dataset Versioning**  
   - `make qc` to ensure label sanity.  
   - `make split` (or bump `--dst` to `deer_det_v0X`) and run `dvc add data/datasets/deer_det_v0X && dvc push`.

4. **Boost Small Objects**  
   - Optional: `make tiles` and `make negatives` to stage night-friendly crops + backgrounds.

5. **Train & Evaluate**  
   - `make train` (update `V` env var when incrementing versions).  
   - `make eval` which runs `scripts/evaluate.py` over `data/eval/*` and enforces pytest gates.

6. **Export & Review**  
   - `make export` to produce ONNX/TorchScript artifacts.  
   - Launch `make ui` to eyeball metrics + error visualizations before shipping.

7. **Release Checklist**  
   - Commit `configs/`, `dvc.lock`, and new metrics.  
   - Attach qualitative notes (night recall delta, FP/min) to the PR.  
   - Upload models to `s3://wildlife-ugv/models/` and update operators if new credentials/assets are required.
