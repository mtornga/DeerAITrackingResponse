import json
import os
from pathlib import Path
from typing import Dict, List

import streamlit as st


def resolve_review_root() -> Path:
    # Allow override via env var (works on mac/ubuntu)
    env = os.environ.get("REVIEW_ROOT")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p
    # Common mounts: mac SMB, Ubuntu USB
    candidates = [
        Path("~/DeerShare/runs/review").expanduser(),
        Path("/Volumes/deer-share/runs/review"),
        Path("/srv/deer-share/runs/review"),
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fallback to current working dir
    return Path("runs/review").resolve()


def load_index(root: Path, date: str | None) -> Dict:
    if not root.exists():
        return {}
    dates = sorted([p.name for p in root.iterdir() if p.is_dir()])
    if not dates:
        return {}
    selected = date or dates[-1]
    index_path = root / selected / "index.json"
    if index_path.exists():
        try:
            data = json.loads(index_path.read_text())
            if data:
                return data
        except Exception:
            pass
    # Fallback: synthesize a minimal index from the directory structure
    return build_index_from_dir(root / selected)


def build_index_from_dir(day_root: Path) -> Dict:
    data: Dict[str, Dict[str, Dict]] = {}
    if not day_root.exists():
        return data
    # Expect structure: <day>/<segment>/<variant>/*.jpg
    for seg_dir in sorted([p for p in day_root.iterdir() if p.is_dir()]):
        variants: Dict[str, Dict] = {}
        for var_dir in sorted([p for p in seg_dir.iterdir() if p.is_dir()]):
            metrics = {"frames": 0, "tp": 0, "fp": 0, "fn": 0}
            summary = var_dir / "summary.json"
            if summary.exists():
                try:
                    m = json.loads(summary.read_text())
                    for k in ("frames", "tp", "fp", "fn"):
                        if k in m:
                            metrics[k] = int(m[k])
                except Exception:
                    pass
            variants[var_dir.name] = metrics
        if variants:
            data[seg_dir.name] = variants
    return data


def main() -> None:
    st.set_page_config(layout="wide")
    st.title("Deer-Vision A/B Review")
    review_root = resolve_review_root()
    votes_path = review_root / "review_votes.json"
    votes = {}
    if votes_path.exists():
        try:
            votes = json.loads(votes_path.read_text())
        except Exception:
            votes = {}

    st.caption(f"Review root: {review_root}")
    # Date selector
    dates = sorted([p.name for p in review_root.iterdir() if p.is_dir()]) if review_root.exists() else []
    if not dates:
        st.info("No review folders found. Set REVIEW_ROOT or mount the share (e.g., ~/DeerShare).")
        return
    date_sel = st.selectbox("Date", dates, index=len(dates) - 1)
    data = load_index(review_root, date_sel)
    if not data:
        st.info(f"Index is empty for {review_root}/{date_sel} (generator still running). Showing directory view if available.")
        data = build_index_from_dir(review_root / date_sel)
        if not data:
            return

    seg = st.selectbox("Segment", sorted(data.keys()))
    variants = data.get(seg, {})

    st.divider()
    mode = st.selectbox("Frame selection", ["Disagreements", "Detections only", "All frames"], index=0)
    show_n = st.slider("Frames per page", min_value=6, max_value=60, value=24, step=6)
    page = st.number_input("Page (0-based)", min_value=0, value=0, step=1)

    def has_overlay_color(img_path: Path) -> bool:
        try:
            import cv2
            im = cv2.imread(str(img_path))
            if im is None:
                return False
            # Downscale for speed
            if im.shape[1] > 640:
                new_w = 640
                new_h = int(im.shape[0] * new_w / im.shape[1])
                im = cv2.resize(im, (new_w, new_h))
            b, g, r = cv2.split(im)
            magenta = (r > 200) & (b > 200) & (g < 60)
            green = (g > 150) & (r < 80) & (b < 80)
            red = (r > 150) & (g < 80) & (b < 80)
            return bool(magenta.any() or green.any() or red.any())
        except Exception:
            return False

    # Load per-variant frames.json if present
    frames_by_variant: Dict[str, Dict[str, Dict]] = {}
    for vname in variants.keys():
        fj = review_root / date_sel / seg / vname / "frames.json"
        try:
            frames_by_variant[vname] = json.loads(fj.read_text()) if fj.exists() else {}
        except Exception:
            frames_by_variant[vname] = {}

    # Build candidate frame keys per mode
    def union_keys() -> List[str]:
        keys = set()
        for d in frames_by_variant.values():
            keys.update(d.keys())
        return sorted(keys)

    selected_keys: List[str] = []
    if mode == "Disagreements":
        for k in union_keys():
            preds = [int(frames_by_variant[v].get(k, {}).get("pred", 0)) for v in frames_by_variant]
            if any(p > 0 for p in preds) and not all(p > 0 for p in preds):
                selected_keys.append(k)
    elif mode == "Detections only":
        for k in union_keys():
            preds = [int(frames_by_variant[v].get(k, {}).get("pred", 0)) for v in frames_by_variant]
            if any(p > 0 for p in preds):
                selected_keys.append(k)
    else:
        # fall back to chronological order by first variant folder listing
        first_v = next(iter(variants.keys())) if variants else None
        if first_v:
            folder = review_root / date_sel / seg / first_v
            selected_keys = [p.name for p in sorted(folder.iterdir()) if p.suffix.lower() == ".jpg"]

    # Paginate
    start = page * show_n
    end = start + show_n
    page_keys = selected_keys[start:end]
    if not page_keys:
        st.caption("No frames selected for current filters.")

    # Render rows: one row per frame across variants
    for fk in page_keys:
        cols = st.columns(len(variants) or 1)
        for (vname, metrics), col in zip(variants.items(), cols):
            with col:
                folder = review_root / date_sel / seg / vname
                img = folder / fk
                if img.exists():
                    st.image(str(img), caption=f"{vname} · {fk}")
                else:
                    st.caption(f"missing: {vname}/{fk}")

    # Voting (per variant)
    st.divider()
    for vname in variants.keys():
        key = f"{seg}:{vname}"
        c1, c2, c3 = st.columns(3)
        if c1.button(f"👍 {vname}", key=f"up_{key}"):
            votes[key] = "up"
        if c2.button(f"👎 {vname}", key=f"down_{key}"):
            votes[key] = "down"
        if c3.button(f"Clear {vname}", key=f"clear_{key}"):
            votes.pop(key, None)

    (votes_path).write_text(json.dumps(votes, indent=2))


if __name__ == "__main__":
    main()
