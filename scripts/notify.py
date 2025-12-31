#!/usr/bin/env python3
"""
Pushover notification module for DeerVision pipeline.

Sends push notifications to phone when wildlife events are detected.
Supports image attachments, deep-link URLs, and rate limiting.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
import tempfile

import cv2
import requests

# Set up logging
logger = logging.getLogger(__name__)


def _ensure_repo_root_on_path() -> Path:
    """Add repo root to path for imports."""
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


@dataclass
class PushoverConfig:
    """Configuration for Pushover notifications."""
    user_key: str
    api_token: str
    enabled: bool = True
    priority: int = 0  # -2 to 2
    sound: str = "pushover"  # notification sound

    @classmethod
    def from_env(cls) -> "PushoverConfig":
        """Load configuration from environment variables."""
        return cls(
            user_key=os.environ.get("PUSHOVER_USER_KEY", ""),
            api_token=os.environ.get("PUSHOVER_API_TOKEN", ""),
            enabled=os.environ.get("PUSHOVER_ENABLED", "true").lower() == "true",
            priority=int(os.environ.get("PUSHOVER_PRIORITY", "0")),
        )

    def is_valid(self) -> bool:
        """Check if configuration is complete."""
        return bool(self.user_key and self.api_token and self.enabled)


class NotificationRateLimiter:
    """Rate limiter to prevent notification spam."""

    def __init__(self, min_interval_seconds: float = 60.0):
        self.min_interval = min_interval_seconds
        self._last_notification: dict[str, float] = {}

    def should_send(self, category: str) -> bool:
        """Check if enough time has passed since last notification for category."""
        now = time.time()
        last = self._last_notification.get(category, 0)
        return (now - last) >= self.min_interval

    def record_send(self, category: str) -> None:
        """Record that a notification was sent."""
        self._last_notification[category] = time.time()

    def time_until_next(self, category: str) -> float:
        """Return seconds until next notification is allowed."""
        now = time.time()
        last = self._last_notification.get(category, 0)
        remaining = self.min_interval - (now - last)
        return max(0, remaining)


# Global rate limiter instance
_rate_limiter = NotificationRateLimiter(min_interval_seconds=60.0)


def extract_thumbnail(
    video_path: Path,
    output_path: Optional[Path] = None,
    frame_ratio: float = 0.3,
    max_size: int = 400,
) -> Optional[Path]:
    """
    Extract a thumbnail frame from a video file.

    Args:
        video_path: Path to video file
        output_path: Where to save thumbnail (temp file if None)
        frame_ratio: Which frame to extract (0.0-1.0)
        max_size: Maximum dimension in pixels

    Returns:
        Path to thumbnail image, or None on failure
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning(f"Could not open video: {video_path}")
            return None

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        target_frame = int(frame_count * frame_ratio)

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            logger.warning(f"Could not read frame from: {video_path}")
            return None

        # Resize if needed
        h, w = frame.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Save thumbnail
        if output_path is None:
            fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            output_path = Path(tmp_path)

        cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return output_path

    except Exception as e:
        logger.error(f"Failed to extract thumbnail: {e}")
        return None


def send_pushover_notification(
    title: str,
    message: str,
    url: Optional[str] = None,
    url_title: Optional[str] = None,
    image_path: Optional[Path] = None,
    priority: Optional[int] = None,
    sound: Optional[str] = None,
    config: Optional[PushoverConfig] = None,
) -> bool:
    """
    Send a Pushover notification.

    Args:
        title: Notification title
        message: Notification body text
        url: Optional URL to include
        url_title: Optional title for the URL
        image_path: Optional path to image attachment
        priority: Override default priority (-2 to 2)
        sound: Override default sound
        config: Override default config

    Returns:
        True if notification was sent successfully
    """
    if config is None:
        config = PushoverConfig.from_env()

    if not config.is_valid():
        logger.warning("Pushover not configured or disabled")
        return False

    # Build request data
    data = {
        "token": config.api_token,
        "user": config.user_key,
        "title": title,
        "message": message,
        "priority": priority if priority is not None else config.priority,
        "sound": sound or config.sound,
    }

    if url:
        data["url"] = url
        if url_title:
            data["url_title"] = url_title

    # Prepare files for image attachment
    files = None
    if image_path and image_path.exists():
        try:
            files = {"attachment": ("image.jpg", open(image_path, "rb"), "image/jpeg")}
        except Exception as e:
            logger.warning(f"Could not attach image: {e}")

    try:
        response = requests.post(
            "https://api.pushover.net/1/messages.json",
            data=data,
            files=files,
            timeout=30,
        )

        if files:
            files["attachment"][1].close()

        if response.status_code == 200:
            logger.info(f"Pushover notification sent: {title}")
            return True
        else:
            logger.error(f"Pushover API error: {response.status_code} - {response.text}")
            return False

    except requests.RequestException as e:
        logger.error(f"Pushover request failed: {e}")
        return False


def notify_wildlife_detection(
    event_id: str,
    category: str,
    confidence: float,
    video_path: Optional[Path] = None,
    review_url: Optional[str] = None,
    timestamp: Optional[datetime] = None,
    config: Optional[PushoverConfig] = None,
    force: bool = False,
    timing_info: Optional[tuple[float, float]] = None,
) -> bool:
    """
    Send notification for a wildlife detection event.

    Args:
        event_id: Unique event identifier (e.g., "2025-12-29/segment_173000")
        category: Detection category ("deer", "person", "animal", etc.)
        confidence: Detection confidence (0-1)
        video_path: Path to video clip for thumbnail extraction
        review_url: URL to mobile review page
        timestamp: Event timestamp (defaults to now)
        config: Override default Pushover config
        force: Bypass rate limiting
        timing_info: Optional (start_sec, end_sec) tuple for detection timing

    Returns:
        True if notification was sent
    """
    # Rate limiting
    if not force and not _rate_limiter.should_send(category):
        remaining = _rate_limiter.time_until_next(category)
        logger.debug(f"Rate limited: {category} (wait {remaining:.0f}s)")
        return False

    # Build notification content
    if timestamp is None:
        timestamp = datetime.now()

    time_str = timestamp.strftime("%H:%M")
    date_str = timestamp.strftime("%Y-%m-%d")

    # Friendly category names
    category_titles = {
        "deer": "Deer Detected",
        "animal": "Wildlife Detected",
        "person": "Person Detected",
        "vehicle": "Vehicle Detected",
    }
    title = category_titles.get(category.lower(), f"{category.title()} Detected")

    # Message body with timing info
    if timing_info:
        start_sec, end_sec = timing_info
        timing_str = f" ({start_sec:.0f}s-{end_sec:.0f}s)"
    else:
        timing_str = ""
    message = f"{category.title()} @ {confidence:.0%}{timing_str}\n{date_str} {time_str}"

    # Extract thumbnail if video provided
    thumbnail_path = None
    if video_path and video_path.exists():
        thumbnail_path = extract_thumbnail(video_path)

    # Send notification
    success = send_pushover_notification(
        title=title,
        message=message,
        url=review_url,
        url_title="Review Clip",
        image_path=thumbnail_path,
        config=config,
    )

    # Clean up temp thumbnail
    if thumbnail_path and thumbnail_path.exists():
        try:
            thumbnail_path.unlink()
        except Exception:
            pass

    if success:
        _rate_limiter.record_send(category)

    return success


def send_test_notification() -> bool:
    """Send a test notification to verify configuration."""
    config = PushoverConfig.from_env()

    if not config.is_valid():
        print("ERROR: Pushover not configured. Check PUSHOVER_USER_KEY and PUSHOVER_API_TOKEN in .env")
        return False

    return send_pushover_notification(
        title="DeerVision Test",
        message="Pipeline notification test successful!",
        priority=0,
        config=config,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Pushover notifications")
    parser.add_argument("--test", action="store_true", help="Send test notification")
    parser.add_argument("--title", type=str, default="Test", help="Notification title")
    parser.add_argument("--message", type=str, default="Test message", help="Notification message")
    parser.add_argument("--image", type=str, help="Path to image attachment")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.test:
        success = send_test_notification()
        print("Test notification sent!" if success else "Failed to send notification")
        sys.exit(0 if success else 1)
    else:
        image_path = Path(args.image) if args.image else None
        success = send_pushover_notification(
            title=args.title,
            message=args.message,
            image_path=image_path,
        )
        sys.exit(0 if success else 1)
