#!/usr/bin/env python3
"""
Mobile-friendly review server for DeerVision pipeline.

Provides a lightweight web interface for reviewing detection events
and submitting feedback from a mobile device.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from flask import Flask, render_template, request, jsonify, send_file, Response

# Set up path
def _ensure_repo_root_on_path() -> Path:
    script_path = Path(__file__).resolve()
    for parent in (script_path.parent, *script_path.parents):
        candidate = parent / ".env"
        if candidate.exists():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return parent
    root = script_path.parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


REPO_ROOT = _ensure_repo_root_on_path()

try:
    from env_loader import load_env_file
    load_env_file()
except ImportError:
    pass

# Configuration
SHARE_ROOT = Path(os.environ.get("DEER_SHARE_SERVER_PATH", "/srv/deer-share"))
EVENTS_DIR = SHARE_ROOT / "runs/live/events"

app = Flask(__name__, template_folder="templates")
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "deervision-review-2025")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def get_event_dirs(limit: int = 50) -> List[Dict[str, Any]]:
    """Get list of event directories, newest first."""
    events = []

    if not EVENTS_DIR.exists():
        return events

    # Walk through date directories
    date_dirs = sorted(EVENTS_DIR.iterdir(), reverse=True)

    for date_dir in date_dirs:
        if not date_dir.is_dir():
            continue

        # Get event subdirectories
        event_subdirs = sorted(date_dir.iterdir(), reverse=True)

        for event_dir in event_subdirs:
            if not event_dir.is_dir():
                continue

            meta_path = event_dir / "meta.json"
            feedback_path = event_dir / "feedback.json"

            event_info = {
                "id": f"{date_dir.name}/{event_dir.name}",
                "date": date_dir.name,
                "segment": event_dir.name,
                "path": str(event_dir),
                "has_feedback": feedback_path.exists(),
                "meta": None,
                "feedback": None,
            }

            # Load meta.json if exists
            if meta_path.exists():
                try:
                    event_info["meta"] = json.loads(meta_path.read_text())
                except Exception as e:
                    logger.warning(f"Error reading meta.json: {e}")

            # Load feedback.json if exists
            if feedback_path.exists():
                try:
                    event_info["feedback"] = json.loads(feedback_path.read_text())
                except Exception as e:
                    logger.warning(f"Error reading feedback.json: {e}")

            events.append(event_info)

            if len(events) >= limit:
                return events

    return events


def get_event_by_id(event_id: str) -> Optional[Dict[str, Any]]:
    """Get a specific event by ID."""
    event_dir = EVENTS_DIR / event_id

    if not event_dir.is_dir():
        return None

    parts = event_id.split("/")
    if len(parts) != 2:
        return None

    date_str, segment = parts

    meta_path = event_dir / "meta.json"
    feedback_path = event_dir / "feedback.json"

    event_info = {
        "id": event_id,
        "date": date_str,
        "segment": segment,
        "path": str(event_dir),
        "has_feedback": feedback_path.exists(),
        "meta": None,
        "feedback": None,
        "video_files": [],
    }

    # Find video files
    for ext in ["mkv", "mp4"]:
        event_info["video_files"].extend([f.name for f in event_dir.glob(f"*.{ext}")])

    # Load meta.json
    if meta_path.exists():
        try:
            event_info["meta"] = json.loads(meta_path.read_text())
        except Exception as e:
            logger.warning(f"Error reading meta.json: {e}")

    # Load feedback
    if feedback_path.exists():
        try:
            event_info["feedback"] = json.loads(feedback_path.read_text())
        except Exception as e:
            logger.warning(f"Error reading feedback.json: {e}")

    return event_info


def save_feedback(event_id: str, feedback: Dict[str, Any]) -> bool:
    """Save feedback for an event."""
    event_dir = EVENTS_DIR / event_id

    if not event_dir.is_dir():
        return False

    feedback_path = event_dir / "feedback.json"

    # Load existing meta to get original prediction
    meta_path = event_dir / "meta.json"
    original_prediction = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            original_prediction = {
                "max_confidence": meta.get("max_confidence"),
                "counts": meta.get("counts"),
                "promoted_by": meta.get("promoted_by"),
            }
        except Exception:
            pass

    # Build feedback record
    feedback_record = {
        "reviewed_at": datetime.utcnow().isoformat() + "Z",
        "reviewer": "mobile",
        "category": feedback.get("category", "unknown"),
        "quality": feedback.get("quality", "unknown"),
        "notes": feedback.get("notes", ""),
        "original_prediction": original_prediction,
    }

    try:
        feedback_path.write_text(json.dumps(feedback_record, indent=2))
        logger.info(f"Saved feedback for {event_id}: {feedback_record['category']}/{feedback_record['quality']}")
        return True
    except Exception as e:
        logger.error(f"Failed to save feedback: {e}")
        return False


@app.route("/")
def index():
    """List pending reviews."""
    events = get_event_dirs(limit=50)

    # Split into pending and reviewed
    pending = [e for e in events if not e["has_feedback"]]
    reviewed = [e for e in events if e["has_feedback"]]

    return render_template(
        "index.html",
        pending=pending,
        reviewed=reviewed[:10],  # Show last 10 reviewed
        total_pending=len(pending),
        total_reviewed=len(reviewed),
    )


@app.route("/review/<path:event_id>")
def review(event_id: str):
    """Review a specific event."""
    event = get_event_by_id(event_id)

    if event is None:
        return "Event not found", 404

    # Get next/prev events for navigation
    all_events = get_event_dirs(limit=100)
    pending = [e for e in all_events if not e["has_feedback"]]

    current_idx = None
    for i, e in enumerate(pending):
        if e["id"] == event_id:
            current_idx = i
            break

    next_event = pending[current_idx + 1]["id"] if current_idx is not None and current_idx + 1 < len(pending) else None
    prev_event = pending[current_idx - 1]["id"] if current_idx is not None and current_idx > 0 else None

    return render_template(
        "review.html",
        event=event,
        next_event=next_event,
        prev_event=prev_event,
        remaining=len(pending),
    )


@app.route("/review/<path:event_id>/feedback", methods=["POST"])
def submit_feedback(event_id: str):
    """Submit feedback for an event."""
    data = request.get_json() or request.form.to_dict()

    category = data.get("category")
    quality = data.get("quality")
    notes = data.get("notes", "")

    if not category or not quality:
        return jsonify({"error": "category and quality are required"}), 400

    success = save_feedback(event_id, {
        "category": category,
        "quality": quality,
        "notes": notes,
    })

    if success:
        # Find next pending event
        all_events = get_event_dirs(limit=100)
        pending = [e for e in all_events if not e["has_feedback"]]
        next_event = pending[0]["id"] if pending else None

        return jsonify({
            "success": True,
            "next_event": next_event,
            "remaining": len(pending),
        })
    else:
        return jsonify({"error": "Failed to save feedback"}), 500


@app.route("/clip/<path:event_id>/video/<filename>")
def serve_video(event_id: str, filename: str):
    """Stream video file."""
    event_dir = EVENTS_DIR / event_id
    video_path = event_dir / filename

    if not video_path.exists():
        return "Video not found", 404

    # Determine MIME type
    mime_type = "video/mp4" if filename.endswith(".mp4") else "video/x-matroska"

    return send_file(video_path, mimetype=mime_type)


@app.route("/clip/<path:event_id>/thumbnail")
def serve_thumbnail(event_id: str):
    """Generate and serve thumbnail."""
    import cv2
    import tempfile

    event = get_event_by_id(event_id)
    if event is None or not event["video_files"]:
        return "Event not found", 404

    video_path = Path(event["path"]) / event["video_files"][0]

    try:
        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        target_frame = int(frame_count * 0.3)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return "Could not read frame", 500

        # Resize
        h, w = frame.shape[:2]
        max_size = 400
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        # Encode to JPEG
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

        return Response(buffer.tobytes(), mimetype="image/jpeg")

    except Exception as e:
        logger.error(f"Thumbnail generation failed: {e}")
        return "Thumbnail generation failed", 500


@app.route("/status")
def status():
    """Pipeline status endpoint."""
    events = get_event_dirs(limit=100)
    pending = [e for e in events if not e["has_feedback"]]
    reviewed = [e for e in events if e["has_feedback"]]

    # Get latest event time
    latest_event = events[0] if events else None
    latest_time = None
    if latest_event and latest_event["meta"]:
        latest_time = latest_event["meta"].get("created_at")

    return jsonify({
        "status": "running",
        "events_today": len([e for e in events if e["date"] == datetime.now().strftime("%Y-%m-%d")]),
        "pending_reviews": len(pending),
        "reviewed_today": len([e for e in reviewed if e["date"] == datetime.now().strftime("%Y-%m-%d")]),
        "latest_event": latest_event["id"] if latest_event else None,
        "latest_time": latest_time,
    })


@app.route("/api/events")
def api_events():
    """API endpoint for events list."""
    limit = request.args.get("limit", 50, type=int)
    pending_only = request.args.get("pending", "false").lower() == "true"

    events = get_event_dirs(limit=limit)

    if pending_only:
        events = [e for e in events if not e["has_feedback"]]

    return jsonify(events)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Mobile review server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8501, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    # Override from env
    host = os.environ.get("REVIEW_SERVER_HOST", args.host)
    port = int(os.environ.get("REVIEW_SERVER_PORT", args.port))

    logger.info(f"Starting mobile review server on {host}:{port}")
    logger.info(f"Events directory: {EVENTS_DIR}")

    app.run(host=host, port=port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
