#!/usr/bin/env python3
"""Streamlit UI to review golden clips with day/night classification.

Shows a sample frame alongside lighting metadata so you can sanity-check
the auto-split. Defaults to 20 clips; filter by category and lighting.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st

try:
    import cv2
except ImportError:  # pragma: no cover - UI fallback
    cv2 = None

SEGMENT_EXTENSIONS: Tuple[str, ...] = (".mkv", ".mp4")


@dataclass
class ClipEntry:
    id: str
    category: str
    lighting: Optional[str]
    lighting_confidence: Optional[float]
    path_rel: str
    dest_dir: str
    tags: List[str]
    notes: str


def resolve_share_root(explicit: Optional[str] = None) -> Path:
    if explicit:
        return Path(explicit).expanduser()

    candidates: List[Path] = []
    env_server = os.environ.get("DEER_SHARE_SERVER_PATH")
    env_local = os.environ.get("DEER_SHARE_LOCAL_MOUNT")

    if env_server:
        candidates.append(Path(env_server).expanduser())
    if env_local:
        candidates.append(Path(env_local).expanduser())

    candidates.append(Path("/srv/deer-share"))
    candidates.append(Path.home() / "DeerShare")

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    raise RuntimeError(
        f"Unable to locate shared storage root; checked: {', '.join(str(c) for c in candidates)}"
    )


@st.cache_data(show_spinner=False)
def load_manifest(manifest_path: Path) -> List[ClipEntry]:
    raw = json.loads(manifest_path.read_text())
    clips_raw = raw.get("clips", [])
    clips: List[ClipEntry] = []
    for entry in clips_raw:
        clips.append(
            ClipEntry(
                id=entry.get("id") or entry.get("segment_id") or entry.get("path", ""),
                category=entry.get("category", "unknown"),
                lighting=entry.get("lighting"),
                lighting_confidence=entry.get("lighting_confidence"),
                path_rel=entry.get("path") or "",
                dest_dir=entry.get("dest_dir") or "",
                tags=[str(t) for t in (entry.get("tags") or [])],
                notes=entry.get("notes", ""),
            )
        )
    return clips


def pick_video_file(clip_dir: Path) -> Optional[Path]:
    for ext in SEGMENT_EXTENSIONS:
        for vid in sorted(clip_dir.glob(f"*{ext}")):
            return vid
    return None


def resolve_clip_dir(golden_root: Path, clip: ClipEntry) -> Optional[Path]:
    """Resolve a clip directory from manifest fields, handling absolute and relative paths."""
    candidates: List[Path] = []

    if clip.path_rel:
        p = Path(clip.path_rel)
        candidates.append(p if p.is_absolute() else golden_root / p)

    if clip.dest_dir:
        d = Path(clip.dest_dir)
        if d.is_absolute():
            candidates.append(d)
        else:
            candidates.append(golden_root / d)

    # Deduplicate while preserving order
    seen = set()
    unique_candidates = []
    for c in candidates:
        key = c.resolve() if c.exists() else c
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(c)

    for c in unique_candidates:
        if c.exists():
            return c
    return None


@st.cache_data(show_spinner=False)
def sample_frame_bytes(video_path: Path, frame_ratio: float) -> Optional[bytes]:
    if cv2 is None:
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target_idx = max(0, min(frame_count - 1, int(frame_count * frame_ratio)))
    if target_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ok, buf = cv2.imencode(".jpg", frame_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return buf.tobytes() if ok else None


def main() -> None:
    st.set_page_config(page_title="Golden Lighting Review", layout="wide")
    st.title("Golden Clips Lighting Review")

    share_root = resolve_share_root()
    golden_root = share_root / "golden_clips"
    manifest_path = golden_root / "manifest.json"
    if not manifest_path.exists():
        st.error(f"Manifest not found at {manifest_path}")
        return

    clips = load_manifest(manifest_path)
    categories = sorted({c.category for c in clips})
    lighting_options = ["all", "day", "night"]

    with st.sidebar:
        st.header("Filters")
        category = st.selectbox("Category", options=["all"] + categories, index=0)
        lighting = st.radio("Lighting", options=lighting_options, index=0)
        max_items = st.slider("Max clips", min_value=5, max_value=50, value=20, step=5)
        frame_ratio = st.slider("Frame position", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
        show_low_conf = st.checkbox("Show low-confidence only", value=False)
        low_conf_thresh = st.slider("Low-confidence threshold", min_value=0.0, max_value=1.0, value=0.15, step=0.01)

    filtered = []
    for clip in clips:
        if category != "all" and clip.category != category:
            continue
        if lighting != "all" and clip.lighting and clip.lighting != lighting:
            continue
        if show_low_conf:
            if clip.lighting_confidence is None:
                continue
            if clip.lighting_confidence > low_conf_thresh:
                continue
        filtered.append(clip)

    st.caption(f"Showing {min(len(filtered), max_items)} of {len(filtered)} clips (filter applied).")

    if not filtered:
        st.info("No clips match the selected filters.")
        return

    for clip in filtered[:max_items]:
        clip_dir = resolve_clip_dir(golden_root, clip)

        video_path = pick_video_file(clip_dir) if clip_dir else None

        cols = st.columns([1, 2])
        meta_col, img_col = cols[0], cols[1]

        with meta_col:
            st.markdown(f"**{clip.id}**")
            st.write(f"Category: `{clip.category}`")
            st.write(f"Lighting: `{clip.lighting or 'unknown'}` (conf: {clip.lighting_confidence})")
            st.write(f"Dir: `{clip.path_rel or clip.dest_dir}`")
            if clip.tags:
                st.write(f"Tags: {', '.join(clip.tags)}")
            if clip.notes:
                st.write(f"Notes: {clip.notes}")

        with img_col:
            if video_path and video_path.exists():
                frame_bytes = sample_frame_bytes(video_path, frame_ratio)
                if frame_bytes:
                    st.image(frame_bytes, caption=video_path.name, use_column_width=True)
                else:
                    st.warning("Could not read frame (missing OpenCV?).")
            else:
                st.warning("No video file found for this clip.")

        st.divider()


if __name__ == "__main__":
    main()
