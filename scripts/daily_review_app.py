#!/usr/bin/env python3
"""Streamlit UI for the Deer Vision daily review workflow."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import streamlit as st


UTC = timezone.utc
SEGMENT_EXTENSIONS: Tuple[str, ...] = (".mp4", ".mkv")
ReviewStatus = Literal["pending", "in_progress", "done"]


@dataclass
class ClipEntry:
    clip_id: str
    path: str
    first_seen: str
    capture_mtime: float
    review_status: ReviewStatus = "pending"
    detector_model: Optional[str] = None
    max_conf: Optional[float] = None
    tags: List[str] = None
    notes: str = ""


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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


def discover_segments(segments_root: Path) -> List[Path]:
    if not segments_root.exists():
        return []
    return [
        p
        for p in segments_root.rglob("*")
        if p.is_file() and p.suffix.lower() in SEGMENT_EXTENSIONS
    ]


def load_index(index_path: Path) -> Dict[str, ClipEntry]:
    if not index_path.exists():
        return {}
    try:
        raw = json.loads(index_path.read_text())
    except json.JSONDecodeError:
        return {}

    clips: Dict[str, ClipEntry] = {}
    for clip_id, data in (raw.get("clips") or {}).items():
        entry = ClipEntry(
            clip_id=clip_id,
            path=data.get("path", clip_id),
            first_seen=data.get("first_seen", _now_iso()),
            capture_mtime=data.get("capture_mtime", 0.0),
            review_status=data.get("review_status", "pending"),
            detector_model=data.get("detector_model"),
            max_conf=data.get("max_conf"),
            tags=data.get("tags") or [],
            notes=data.get("notes", ""),
        )
        clips[clip_id] = entry
    return clips


def augment_with_event_meta(share_root: Path, entry: ClipEntry) -> None:
    try:
        rel_path = Path(entry.path)
        date = rel_path.parent.name
        stem = rel_path.stem
    except Exception:
        return

    meta_path = share_root / "runs" / "live" / "events" / date / stem / "meta.json"
    if not meta_path.exists():
        return

    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return

    max_conf = meta.get("max_confidence")
    if isinstance(max_conf, (int, float)):
        entry.max_conf = float(max_conf)

    counts = meta.get("counts") or {}
    tags = set(entry.tags or [])
    for label, count in counts.items():
        if count:
            tags.add(label)
    entry.tags = sorted(tags)
    entry.detector_model = entry.detector_model or "mdv5_ultralytics"


def build_index(share_root: Path, segments_root: Path, index_path: Path) -> Dict[str, ClipEntry]:
    existing = load_index(index_path)

    for segment_path in discover_segments(segments_root):
        try:
            rel = segment_path.relative_to(share_root)
        except ValueError:
            rel = segment_path.relative_to(segments_root)

        clip_id = str(rel)
        stat = segment_path.stat()
        if clip_id in existing:
            entry = existing[clip_id]
            if stat.st_mtime != entry.capture_mtime:
                entry.capture_mtime = stat.st_mtime
        else:
            entry = ClipEntry(
                clip_id=clip_id,
                path=str(rel),
                first_seen=_now_iso(),
                capture_mtime=stat.st_mtime,
                review_status="pending",
                tags=[],
            )
            existing[clip_id] = entry

        augment_with_event_meta(share_root, entry)

    # Do not write back here; the CLI can own serialization.
    return existing


def clip_bucket_local(entry: ClipEntry) -> datetime:
    dt = datetime.fromtimestamp(entry.capture_mtime).astimezone()
    return dt.replace(minute=0, second=0, microsecond=0)


def main() -> None:
    st.set_page_config(page_title="Deer Vision Daily Review", layout="wide")
    st.title("Deer Vision — Daily Review")
    st.markdown(
        "Morning review inbox for segments captured on the yard cameras. "
        "Filtered to clips where the detector saw something interesting."
    )

    # Shared storage resolution
    share_root_str = st.sidebar.text_input(
        "Shared root override",
        value="",
        help="Optional: override the detected share root (e.g., /srv/deer-share or ~/DeerShare).",
    )
    try:
        share_root = resolve_share_root(share_root_str or None)
    except RuntimeError as exc:
        st.error(str(exc))
        return

    segments_root = share_root / "runs" / "live" / "analysis"
    index_path = share_root / "index" / "daily_review_index.json"

    if not segments_root.exists():
        st.warning(f"No segment directory found at {segments_root}")
        return

    # Filters
    status: ReviewStatus = st.sidebar.selectbox(
        "Status", options=["pending", "in_progress", "done"], index=0
    )
    events_only = st.sidebar.checkbox("Events only", value=True)
    min_conf = st.sidebar.slider(
        "Minimum max confidence", min_value=0.0, max_value=1.0, value=0.3, step=0.05
    )
    limit = st.sidebar.number_input("Max clips to show", min_value=1, max_value=200, value=40)

    clips = build_index(share_root, segments_root, index_path)
    filtered: List[ClipEntry] = []
    for entry in clips.values():
        if entry.review_status != status:
            continue
        if events_only and (entry.max_conf is None or entry.max_conf <= 0.0):
            continue
        if min_conf > 0.0 and (entry.max_conf is None or entry.max_conf < min_conf):
            continue
        filtered.append(entry)

    filtered.sort(key=lambda e: e.capture_mtime, reverse=True)

    st.caption(
        f"{len(clips)} clip(s) tracked; showing {min(len(filtered), limit)} "
        f"with status={status}, events_only={events_only}, min_conf>={min_conf:.2f}"
    )

    if not filtered:
        st.info("No clips match the current filters.")
        return

    # Layout: left column = list, right column = selected clip details.
    list_col, detail_col = st.columns([2, 3])

    with list_col:
        st.subheader("Clips")
        selected_id = st.session_state.get("selected_clip_id", filtered[0].clip_id)

        prev_bucket: Optional[datetime] = None
        for entry in filtered[: int(limit)]:
            bucket = clip_bucket_local(entry)
            if prev_bucket != bucket:
                st.markdown(f"**{bucket.strftime('%Y-%m-%d %H:00 %Z')}**")
                prev_bucket = bucket

            label = Path(entry.path).name
            tags_str = ", ".join(entry.tags or [])
            conf_str = "-" if entry.max_conf is None else f"{entry.max_conf:.2f}"

            if st.button(
                f"{label}  •  max={conf_str}  •  {tags_str or 'untagged'}",
                key=f"clip-btn-{entry.clip_id}",
            ):
                selected_id = entry.clip_id
                st.session_state["selected_clip_id"] = selected_id

    with detail_col:
        st.subheader("Details")
        selected = next((e for e in filtered if e.clip_id == selected_id), filtered[0])

        st.markdown(f"**Clip:** `{selected.path}`")
        st.markdown(
            f"- **Status:** {selected.review_status}\n"
            f"- **Max confidence:** {selected.max_conf if selected.max_conf is not None else '-'}\n"
            f"- **Tags:** {', '.join(selected.tags or []) or '-'}\n"
            f"- **First seen:** {selected.first_seen}\n"
        )

        video_path = share_root / selected.path
        if video_path.exists():
            st.video(str(video_path))
        else:
            st.warning(f"Video file not found at {video_path}")

        events_rel = Path(selected.path).parent.name, Path(selected.path).stem
        events_dir = share_root / "runs" / "live" / "events" / events_rel[0] / events_rel[1]
        meta_path = events_dir / "meta.json"
        detections_path = events_dir / "detections.json"

        with st.expander("Event metadata (read-only)", expanded=False):
            if meta_path.exists():
                st.json(json.loads(meta_path.read_text()))
            else:
                st.write("No meta.json found for this clip.")

        with st.expander("Raw detections (read-only)", expanded=False):
            if detections_path.exists():
                st.json(json.loads(detections_path.read_text()))
            else:
                st.write("No detections.json found for this clip.")


if __name__ == "__main__":
    main()

