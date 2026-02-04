#!/usr/bin/env python3
"""Streamlit UI to review golden_frames_v8 images and labels."""

from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import streamlit as st

try:
    import cv2
except ImportError:  # pragma: no cover - UI fallback
    cv2 = None


UTC = timezone.utc
IMAGE_EXTS = (".jpg", ".jpeg", ".png")
CLASS_NAMES = {0: "deer", 1: "person"}
REVIEW_STATUSES = ["ok", "bad_label", "uncertain"]


def _ensure_repo_root_on_path() -> Path:
    script_path = Path(__file__).resolve()
    for parent in script_path.parents:
        if (parent / ".env").exists():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return parent
    fallback = script_path.parents[1]
    if str(fallback) not in sys.path:
        sys.path.insert(0, str(fallback))
    return fallback


REPO_ROOT = _ensure_repo_root_on_path()
if str(REPO_ROOT / "agents") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "agents"))

from training_loop_utils import resolve_share_root  # noqa: E402


def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_training_config() -> Dict[str, object]:
    config_path = REPO_ROOT / "configs" / "training_loop.json"
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text())
    except json.JSONDecodeError:
        return {}


def resolve_dataset_root(share_root: Path, config: Dict[str, object]) -> Path:
    root = config.get("golden_frames_root", "golden_frames_v8")
    root_path = Path(str(root))
    if root_path.is_absolute():
        return root_path
    return share_root / root_path


@dataclass
class ImageEntry:
    image_path: str
    label_path: str
    lighting: str
    split: str
    label_count: int
    class_counts: Dict[str, int]
    has_labels: bool


def parse_label_file(label_path: Path) -> Tuple[List[Tuple[int, float, float, float, float]], Dict[str, int]]:
    labels: List[Tuple[int, float, float, float, float]] = []
    counts: Dict[str, int] = {}
    if not label_path.exists():
        return labels, counts
    for raw in label_path.read_text().splitlines():
        parts = raw.strip().split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError:
            continue
        labels.append((class_id, x_center, y_center, width, height))
        name = CLASS_NAMES.get(class_id, f"class_{class_id}")
        counts[name] = counts.get(name, 0) + 1
    return labels, counts


@st.cache_data(show_spinner=False)
def discover_entries(dataset_root: Path) -> List[ImageEntry]:
    entries: List[ImageEntry] = []
    for lighting in ("day", "night"):
        for split in ("train", "val"):
            images_dir = dataset_root / lighting / "images" / split
            labels_dir = dataset_root / lighting / "labels" / split
            if not images_dir.exists():
                continue
            for ext in IMAGE_EXTS:
                for image_path in sorted(images_dir.rglob(f"*{ext}")):
                    label_path = labels_dir / f"{image_path.stem}.txt"
                    labels, counts = parse_label_file(label_path)
                    has_labels = bool(labels)
                    entries.append(
                        ImageEntry(
                            image_path=str(image_path),
                            label_path=str(label_path),
                            lighting=lighting,
                            split=split,
                            label_count=len(labels),
                            class_counts=counts,
                            has_labels=has_labels,
                        )
                    )
    return entries


def filter_entries(
    entries: Iterable[ImageEntry],
    lighting: str,
    split: str,
    class_filter: str,
    label_filter: str,
) -> List[ImageEntry]:
    filtered: List[ImageEntry] = []
    for entry in entries:
        if lighting != "all" and entry.lighting != lighting:
            continue
        if split != "all" and entry.split != split:
            continue
        if label_filter == "with_labels" and not entry.has_labels:
            continue
        if label_filter == "background_only" and entry.has_labels:
            continue
        if class_filter != "any":
            if entry.class_counts.get(class_filter, 0) <= 0:
                continue
        filtered.append(entry)
    return filtered


def load_image_with_boxes(image_path: Path, labels: List[Tuple[int, float, float, float, float]]) -> Optional[object]:
    if cv2 is None:
        return None
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    height, width = image.shape[:2]
    for class_id, x_center, y_center, box_w, box_h in labels:
        x1 = int((x_center - box_w / 2) * width)
        y1 = int((y_center - box_h / 2) * height)
        x2 = int((x_center + box_w / 2) * width)
        y2 = int((y_center + box_h / 2) * height)
        color = (0, 165, 255) if class_id == 0 else (255, 100, 100)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = CLASS_NAMES.get(class_id, f"class_{class_id}")
        cv2.putText(image, label, (x1, max(y1 - 6, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def review_log_path(dataset_root: Path) -> Path:
    return dataset_root / "review_notes.jsonl"


def append_review(log_path: Path, entry: ImageEntry, status: str, note: str) -> None:
    payload = {
        "timestamp": now_iso(),
        "status": status,
        "note": note,
        "image_path": entry.image_path,
        "label_path": entry.label_path,
        "lighting": entry.lighting,
        "split": entry.split,
        "label_count": entry.label_count,
        "class_counts": entry.class_counts,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def trigger_rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
        return
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
        return


def load_review_stats(log_path: Path) -> Dict[str, int]:
    stats = {status: 0 for status in REVIEW_STATUSES}
    if not log_path.exists():
        return stats
    for line in log_path.read_text().splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        status = payload.get("status")
        if status in stats:
            stats[status] += 1
    return stats


def pick_entry(entries: List[ImageEntry]) -> Optional[ImageEntry]:
    if not entries:
        return None
    if "entry_index" not in st.session_state:
        st.session_state.entry_index = 0
    if st.session_state.entry_index >= len(entries):
        st.session_state.entry_index = 0
    return entries[st.session_state.entry_index]


def main() -> None:
    st.set_page_config(page_title="Golden Frames Review", layout="wide")
    st.title("Golden Frames Review (v8)")

    share_root = resolve_share_root()
    config = load_training_config()
    dataset_root = resolve_dataset_root(share_root, config)
    if not dataset_root.exists():
        st.error(f"Dataset root not found: {dataset_root}")
        return

    entries = discover_entries(dataset_root)
    if not entries:
        st.error("No images found in dataset.")
        return

    with st.sidebar:
        st.header("Filters")
        lighting = st.selectbox("Lighting", options=["all", "day", "night"], index=0)
        split = st.selectbox("Split", options=["all", "train", "val"], index=0)
        class_filter = st.selectbox("Class", options=["any", "deer", "person"], index=0)
        label_filter = st.selectbox(
            "Labels",
            options=["any", "with_labels", "background_only"],
            index=0,
        )
        shuffle = st.checkbox("Shuffle", value=True)
        max_items = st.slider("Max items", min_value=50, max_value=5000, value=500, step=50)

    filtered = filter_entries(entries, lighting, split, class_filter, label_filter)
    if shuffle:
        random.shuffle(filtered)
    filtered = filtered[:max_items]

    st.caption(f"Showing {len(filtered)} images (filtered from {len(entries)} total).")

    if not filtered:
        st.info("No entries match the current filters.")
        return

    controls = st.columns([1, 1, 1, 2])
    if controls[0].button("Prev"):
        st.session_state.entry_index = max(st.session_state.entry_index - 1, 0)
    if controls[1].button("Next"):
        st.session_state.entry_index = min(st.session_state.entry_index + 1, len(filtered) - 1)
    if controls[2].button("Random"):
        st.session_state.entry_index = random.randint(0, len(filtered) - 1)

    current = pick_entry(filtered)
    if current is None:
        return

    image_path = Path(current.image_path)
    label_path = Path(current.label_path)
    labels, counts = parse_label_file(label_path)

    left, right = st.columns([2, 1])
    with left:
        if cv2 is None:
            st.warning("OpenCV not available; showing raw image without boxes.")
            st.image(str(image_path), caption=str(image_path))
        else:
            annotated = load_image_with_boxes(image_path, labels)
            if annotated is None:
                st.error(f"Failed to load image: {image_path}")
            else:
                st.image(annotated, caption=str(image_path))

    with right:
        st.markdown("**Entry**")
        st.write(f"Lighting: `{current.lighting}`")
        st.write(f"Split: `{current.split}`")
        st.write(f"Labels: `{current.label_count}`")
        st.write(f"Label file: `{label_path}`")
        if counts:
            st.write("Class counts:")
            st.json(counts)
        else:
            st.write("No labels detected (background).")

        st.markdown("**Review**")
        note = st.text_area("Note", value="", height=80)
        action_cols = st.columns(3)
        log_path = review_log_path(dataset_root)
        if action_cols[0].button("Mark OK"):
            append_review(log_path, current, "ok", note)
            st.cache_data.clear()
            trigger_rerun()
        if action_cols[1].button("Flag Bad"):
            append_review(log_path, current, "bad_label", note)
            st.cache_data.clear()
            trigger_rerun()
        if action_cols[2].button("Uncertain"):
            append_review(log_path, current, "uncertain", note)
            st.cache_data.clear()
            trigger_rerun()

        stats = load_review_stats(log_path)
        st.markdown("**Review stats**")
        st.json(stats)

        if log_path.exists():
            st.download_button(
                "Download review log",
                data=log_path.read_text(),
                file_name=log_path.name,
                mime="application/json",
            )


if __name__ == "__main__":
    main()
