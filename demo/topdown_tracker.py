import argparse, collections, time, csv, json, os, sys, math, threading
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
from ultralytics import YOLO


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


_ensure_repo_root_on_path()

from env_loader import load_env_file, require_env  # noqa: E402

load_env_file()
DEFAULT_RTSP = os.getenv("WYZE_TABLETOP_RTSP")

from cutebot.automation.tracker import ReolinkGPTTracker, CutebotPose  # noqa: E402

from calibration.tabletop_geometry import (
    BOARD_HEIGHT_FT,
    BOARD_HEIGHT_IN,
    BOARD_WIDTH_FT,
    BOARD_WIDTH_IN,
    IMAGE_POINTS,
    image_to_world_homography,
)

# ---------------------- CONFIG (EDIT THESE) ----------------------
# 1) Camera RTSP and YOLO model
MODEL_PATH = "runs/detect/ultraYOLODetection1_v13/weights/best.pt"


# 2) Map/world size in real units (e.g., feet)
WORLD_W_FT = BOARD_WIDTH_FT
WORLD_H_FT = BOARD_HEIGHT_FT
WORLD_W_IN = BOARD_WIDTH_IN
WORLD_H_IN = BOARD_HEIGHT_IN

# 3) Canvas (pixels) for the 2D map view (matches background image size)
CANVAS_W = 2873
CANVAS_H = 3074

# 4) Background image for the top-down map
MAP_BACKGROUND = "demo/topdown.jpg"
BACKGROUND_PROPERTY_PTS = np.array([
    [284, 3063],   # corresponds to WORLD (0,0) bottom-left
    [2735, 2997],  # corresponds to (W,0) bottom-right
    [2752,  120],  # corresponds to (W,H) top-right
    [ 365,  120],  # corresponds to (0,H) top-left
], dtype=np.float32)

# Optional: filter to specific classes (names in your model)
KEEP_CLASSES = {"horse", "alien_maggie", "cutebot"}  # empty set = keep all
SINGLETON_CLASSES = {"cutebot"}  # classes expected to appear at most once
CONF_THR = 0.2
IOU_THR  = 0.45
TRAIL_LEN = 60  # points per track in world space
SAVE_EVENTS_CSV = "detections_world.csv"  # set to "" to disable
# ---------------------------------------------------------------
MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_WAIT_SEC = 2.0

ARROW_LENGTH_FT = 0.35
ARROW_COLOR = (40, 200, 60)
ARROW_OUTLINE = (10, 50, 20)
HEADING_OVERLAY_POS = (40, 60)
HEADING_OVERLAY_BG = (15, 15, 15)
DESTINATION_COLOR = (255, 60, 180)
DESTINATION_RADIUS = 9


class GPTOverlayState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_pose: Optional[CutebotPose] = None
        self._nose_inches: Optional[Tuple[float, float]] = None
        self._heading_degrees: Optional[float] = None
        self._confidence: Optional[float] = None
        self._updated_ts: float = 0.0
        self._heading_history: collections.deque[str] = collections.deque(maxlen=5)
        self._last_error: Optional[str] = None

    def update_from_pose(self, pose: CutebotPose) -> None:
        heading_text: Optional[str] = None
        if pose.heading_degrees is not None:
            heading_text = f"{pose.heading_degrees:.1f} deg"
        with self._lock:
            self._last_pose = pose
            self._nose_inches = (pose.x_in, pose.y_in)
            self._heading_degrees = pose.heading_degrees
            self._confidence = pose.confidence
            self._updated_ts = pose.timestamp
            self._last_error = None
            if heading_text:
                if not self._heading_history or self._heading_history[0] != heading_text:
                    self._heading_history.appendleft(heading_text)

    def record_error(self, message: str) -> None:
        with self._lock:
            self._last_error = message

    def snapshot(self) -> dict:
        with self._lock:
            pose = self._last_pose
            return {
                "pose": pose,
                "nose_inches": self._nose_inches,
                "heading": self._heading_degrees,
                "confidence": self._confidence,
                "updated_ts": self._updated_ts,
                "heading_history": list(self._heading_history),
                "error": self._last_error,
            }


def start_gpt_pose_worker(
    *,
    poll_interval: float,
    min_confidence: float,
    tracker_kwargs: dict,
) -> tuple[GPTOverlayState, threading.Event, threading.Thread]:
    state = GPTOverlayState()
    stop_event = threading.Event()

    def worker():
        tracker: Optional[ReolinkGPTTracker] = None
        try:
            tracker = ReolinkGPTTracker(**tracker_kwargs)
            tracker.start()
        except Exception as exc:
            state.record_error(f"GPT tracker init failed: {exc}")
            return

        try:
            interval = max(0.5, float(poll_interval))
            while not stop_event.is_set():
                try:
                    pose = tracker.detect_once()
                    if pose and pose.confidence >= min_confidence:
                        state.update_from_pose(pose)
                    elif pose:
                        state.record_error(f"Pose below confidence {pose.confidence:.2f} < {min_confidence}")
                except Exception as exc:
                    state.record_error(str(exc))
                finally:
                    stop_event.wait(interval)
        finally:
            try:
                tracker.close()
            except Exception:
                pass

    thread = threading.Thread(target=worker, name="gpt-pose-worker", daemon=True)
    thread.start()
    return state, stop_event, thread


def color_for_id(identifier):
    if isinstance(identifier, str):
        seed = abs(hash(identifier)) % (2**32)
    else:
        seed = int(identifier)
    np.random.seed(seed)
    return tuple(int(x) for x in np.random.randint(80, 255, size=3))

def bottom_center(xyxy):
    x1, y1, x2, y2 = map(float, xyxy)
    return (x1 + x2) / 2.0, y2

def load_calibration_matrix(path: str) -> np.ndarray:
    """
    Load a precomputed image->world calibration matrix from JSON.
    """
    data = json.loads(Path(path).read_text())
    matrix = np.asarray(data.get("matrix"), dtype=np.float32)
    if matrix.shape == (2, 3):
        matrix = np.vstack([matrix, np.array([0.0, 0.0, 1.0], dtype=np.float32)])
    if matrix.shape != (3, 3):
        raise ValueError(f"Calibration matrix must be 3x3, got shape {matrix.shape}")
    return matrix

def project_points(H, pts_xy):
    # pts_xy: Nx2 pixels -> Nx2 world
    pts = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return proj

# Homography from world feet -> background canvas ROI
world_corners = np.array([
    [0.0, 0.0],
    [WORLD_W_FT, 0.0],
    [WORLD_W_FT, WORLD_H_FT],
    [0.0, WORLD_H_FT],
], dtype=np.float32)

H_world2canvas, _ = cv2.findHomography(world_corners, BACKGROUND_PROPERTY_PTS, method=0)
if H_world2canvas is None:
    raise RuntimeError("Could not compute world->canvas homography. Check BACKGROUND_PROPERTY_PTS.")


H_canvas2world = np.linalg.inv(H_world2canvas)


def world_to_canvas_pts(world_xy):
    pts = np.asarray(world_xy, np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H_world2canvas).reshape(-1, 2)
    return out.astype(int)


def canvas_to_world_pts(canvas_xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(canvas_xy, np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H_canvas2world).reshape(-1, 2)
    return out


MAP_BG = cv2.imread(MAP_BACKGROUND)
if MAP_BG is None:
    raise FileNotFoundError(f"Background image not found: {MAP_BACKGROUND}")
MAP_BG = cv2.resize(MAP_BG, (CANVAS_W, CANVAS_H))
_MAP_CANVAS_BASE = None

def draw_map_background():
    global _MAP_CANVAS_BASE
    if _MAP_CANVAS_BASE is None:
        canvas = cv2.convertScaleAbs(MAP_BG, alpha=1.05, beta=6)
        for i in range(4):
            p1 = tuple(BACKGROUND_PROPERTY_PTS[i].astype(int))
            p2 = tuple(BACKGROUND_PROPERTY_PTS[(i + 1) % 4].astype(int))
            cv2.line(canvas, p1, p2, (255, 255, 255), 2)
        draw_world_grid(canvas)
        draw_world_axes(canvas)
        _MAP_CANVAS_BASE = canvas
    return _MAP_CANVAS_BASE.copy()

def clamp_canvas_pt(pt):
    x = int(np.clip(pt[0], 0, CANVAS_W-1))
    y = int(np.clip(pt[1], 0, CANVAS_H-1))
    return x, y

def draw_calibration_quad(img, pts, color=(0, 200, 255)):
    p = pts.astype(int)
    for i in range(4):
        cv2.line(img, tuple(p[i]), tuple(p[(i + 1) % 4]), color, 2)
        cv2.circle(img, tuple(p[i]), 5, (255, 255, 255), -1)
        cv2.putText(
            img,
            ["TL", "TR", "BR", "BL"][i],
            (p[i][0] + 6, p[i][1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

def draw_pixel_grid(img, step=100, color=(60, 60, 60)):
    h, w = img.shape[:2]
    for x in range(0, w, step):
        cv2.line(img, (x, 0), (x, h), color, 1)
        cv2.putText(img, str(x), (x + 2, 18), cv2.FONT_HERSHEY_PLAIN, 1, color, 1, cv2.LINE_AA)
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w, y), color, 1)
        cv2.putText(img, str(y), (2, y - 4), cv2.FONT_HERSHEY_PLAIN, 1, color, 1, cv2.LINE_AA)


def draw_world_grid(canvas, step_in=6.0, color=(75, 75, 75)):
    """Overlay straight world-coordinate grid lines at the requested spacing (inches)."""
    step_ft = step_in / 12.0
    # Vertical lines (constant X)
    x = 0.0
    while x <= WORLD_W_FT + 1e-6:
        pts = world_to_canvas_pts([[x, 0.0], [x, WORLD_H_FT]])
        p1, p2 = tuple(pts[0]), tuple(pts[1])
        cv2.line(canvas, p1, p2, color, 1, cv2.LINE_AA)
        label_vec = world_to_canvas_pts([[x, 0.0]])[0] + np.array([0, -12], dtype=int)
        label_pt = (int(label_vec[0]), int(label_vec[1]))
        cv2.putText(
            canvas,
            f"{int(round(x * 12))}\"",
            label_pt,
            cv2.FONT_HERSHEY_PLAIN,
            1.0,
            color,
            1,
            cv2.LINE_AA,
        )
        x += step_ft
    # Horizontal lines (constant Y)
    y = 0.0
    while y <= WORLD_H_FT + 1e-6:
        pts = world_to_canvas_pts([[0.0, y], [WORLD_W_FT, y]])
        p1, p2 = tuple(pts[0]), tuple(pts[1])
        cv2.line(canvas, p1, p2, color, 1, cv2.LINE_AA)
        label_vec = world_to_canvas_pts([[0.0, y]])[0] + np.array([6, 4], dtype=int)
        label_pt = (int(label_vec[0]), int(label_vec[1]))
        cv2.putText(
            canvas,
            f"{int(round(y * 12))}\"",
            label_pt,
            cv2.FONT_HERSHEY_PLAIN,
            1.0,
            color,
            1,
            cv2.LINE_AA,
        )
        y += step_ft


def draw_world_axes(canvas):
    """Draw origin marker and axis labels."""
    origin_px = world_to_canvas_pts([[0.0, 0.0]])[0]
    origin_pt = (int(origin_px[0]), int(origin_px[1]))
    cv2.circle(canvas, origin_pt, 6, (0, 120, 255), -1)
    label_origin = origin_px + np.array([8, -6], dtype=int)
    label_origin_pt = (int(label_origin[0]), int(label_origin[1]))
    cv2.putText(canvas, "(0,0)", label_origin_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, "(0,0)", label_origin_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    # axis labels near positive directions
    x_axis_pt = world_to_canvas_pts([[WORLD_W_FT, 0.0]])[0]
    label_x = x_axis_pt + np.array([-60, -10], dtype=int)
    label_x_pt = (int(label_x[0]), int(label_x[1]))
    cv2.putText(canvas, "X →", label_x_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, "X →", label_x_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    y_axis_pt = world_to_canvas_pts([[0.0, WORLD_H_FT]])[0]
    label_y = y_axis_pt + np.array([6, 24], dtype=int)
    label_y_pt = (int(label_y[0]), int(label_y[1]))
    cv2.putText(canvas, "↑ Y", label_y_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, "↑ Y", label_y_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)


def draw_heading_history(
    canvas: np.ndarray,
    headings: List[str],
    position: Tuple[int, int] = HEADING_OVERLAY_POS,
    status: Optional[str] = None,
) -> None:
    if not headings and not status:
        status = "Waiting for GPT pose…"
    x0, y0 = position
    line_height = 30
    width = 260
    content_lines = headings if headings else []
    height = line_height * (max(len(content_lines), 1) + 1) + 12
    top_left = (max(0, x0 - 16), max(0, y0 - 28))
    bottom_right = (min(CANVAS_W - 1, top_left[0] + width), min(CANVAS_H - 1, top_left[1] + height))
    cv2.rectangle(canvas, top_left, bottom_right, HEADING_OVERLAY_BG, -1)
    cv2.putText(
        canvas,
        "Headings",
        (x0, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )
    if status:
        cv2.putText(
            canvas,
            status,
            (x0, y0 + line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (200, 200, 200),
            2,
            cv2.LINE_AA,
        )
        base_idx = 2
    else:
        base_idx = 1
    for idx, heading in enumerate(headings, start=base_idx):
        y = y0 + idx * line_height
        cv2.putText(
            canvas,
            heading,
            (x0, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (150, 255, 150),
            3,
            cv2.LINE_AA,
        )


def draw_cutebot_marker(
    canvas: np.ndarray,
    base_world_ft: Tuple[float, float],
    heading_deg: Optional[float],
    nose_label: Optional[str],
) -> Tuple[int, int]:
    """Draw a heading-aware marker for the Cutebot and return the base canvas coordinates."""
    base_canvas = world_to_canvas_pts([base_world_ft])[0]
    if heading_deg is not None:
        angle_rad = math.radians(heading_deg)
        dx = math.sin(angle_rad) * ARROW_LENGTH_FT
        dy = math.cos(angle_rad) * ARROW_LENGTH_FT
    else:
        dx = 0.0
        dy = ARROW_LENGTH_FT * 0.5
    tip_world = (base_world_ft[0] + dx, base_world_ft[1] + dy)
    base_px, tip_px = world_to_canvas_pts([base_world_ft, tip_world])
    base_pt = tuple(int(v) for v in base_px)
    tip_pt = tuple(int(v) for v in tip_px)
    cv2.arrowedLine(canvas, base_pt, tip_pt, ARROW_OUTLINE, 6, tipLength=0.18, line_type=cv2.LINE_AA)
    cv2.arrowedLine(canvas, base_pt, tip_pt, ARROW_COLOR, 3, tipLength=0.24, line_type=cv2.LINE_AA)
    cv2.circle(canvas, base_pt, 5, (0, 0, 0), -1, lineType=cv2.LINE_AA)
    if nose_label:
        text_origin = (base_pt[0] + 12, base_pt[1] - 16)
        cv2.putText(
            canvas,
            nose_label,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
    return base_pt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rtsp",
        default=DEFAULT_RTSP,
        help="RTSP source (default: value from WYZE_TABLETOP_RTSP)",
    )
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--show_video", action="store_true")
    ap.add_argument("--show_map", action="store_true")
    ap.add_argument("--save_map", default="", help="Optional: write a map-only MP4")
    ap.add_argument("--fps_cap", type=float, default=0)
    ap.add_argument("--conf", type=float, default=CONF_THR, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=IOU_THR, help="NMS IoU threshold")
    ap.add_argument("--imgsz", type=int, default=1280, help="Inference image size (pixels)")
    ap.add_argument("--max_det", type=int, default=300, help="Max detections per frame")
    ap.add_argument("--calibration", default="", help="Optional JSON file with image->world matrix.")
    ap.add_argument("--disable-yolo", action="store_true", help="Skip YOLO tracking and rely on GPT pose only.")
    ap.add_argument(
        "--gpt-poll-sec",
        type=float,
        default=0.0,
        help="If >0, poll Reolink GPT tracker at this interval (seconds) for heading/nose overlays.",
    )
    ap.add_argument("--gpt-min-confidence", type=float, default=0.45)
    ap.add_argument("--gpt-rtsp", default=None, help="Override Reolink RTSP URL for GPT pose polling.")
    ap.add_argument(
        "--gpt-snapshot-dir",
        type=Path,
        default=Path("tmp/reolink_snapshots"),
        help="Directory to store temporary GPT snapshots.",
    )
    ap.add_argument("--gpt-keep-snapshots", action="store_true", help="Keep GPT snapshots for debugging.")
    ap.add_argument("--gpt-transport", choices=("tcp", "udp"), default="tcp")
    ap.add_argument("--gpt-ffmpeg", default="ffmpeg")
    ap.add_argument(
        "--gpt-transform",
        default="calibration/reolink_gpt/transform.json",
        help="Affine transform JSON for GPT inches calibration (use 'none' to disable).",
    )
    ap.add_argument(
        "--gpt-capture-timeout",
        type=float,
        default=20.0,
        help="Timeout (seconds) when capturing GPT snapshots.",
    )
    args = ap.parse_args()

    # Homography (image -> world)
    if args.calibration:
        try:
            H = load_calibration_matrix(args.calibration)
            print(f"[info] Loaded calibration matrix from {args.calibration}")
        except Exception as exc:
            raise SystemExit(f"Failed to load calibration from {args.calibration}: {exc}") from exc
    else:
        H = image_to_world_homography()

    # Model + capture
    yolo_enabled = not args.disable_yolo
    model = None
    if yolo_enabled:
        model = YOLO(args.model)

    rtsp_source = args.rtsp or require_env("WYZE_TABLETOP_RTSP")
    use_capture = yolo_enabled or args.show_video
    cap = None
    if use_capture:
        cap = cv2.VideoCapture(rtsp_source)
        if not cap.isOpened():
            raise SystemExit(f"Could not open source: {rtsp_source}")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Map writer (optional)
    writer = None
    if args.save_map:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save_map, fourcc, cap.get(cv2.CAP_PROP_FPS) or 20,
                                 (CANVAS_W, CANVAS_H))

    # Trails in WORLD space
    trails = collections.defaultdict(lambda: collections.deque(maxlen=TRAIL_LEN))

    # CSV logger
    csv_writer = None
    if SAVE_EVENTS_CSV:
        new_file = not os.path.exists(SAVE_EVENTS_CSV)
        csvf = open(SAVE_EVENTS_CSV, "a", newline="")
        csv_writer = csv.writer(csvf)
        if new_file:
            csv_writer.writerow(["ts", "track_id", "class", "conf", "world_x_ft", "world_y_ft"])

    gpt_state = GPTOverlayState()
    gpt_stop_event: Optional[threading.Event] = None
    gpt_thread: Optional[threading.Thread] = None
    if args.gpt_poll_sec > 0:
        tracker_kwargs = dict(
            rtsp_url=args.gpt_rtsp,
            snapshot_dir=args.gpt_snapshot_dir,
            ffmpeg_path=args.gpt_ffmpeg,
            transport=args.gpt_transport,
            cleanup_snapshots=not args.gpt_keep_snapshots,
            capture_timeout_sec=args.gpt_capture_timeout,
            transform_path=None if str(args.gpt_transform).lower() == "none" else args.gpt_transform,
        )
        gpt_state, gpt_stop_event, gpt_thread = start_gpt_pose_worker(
            poll_interval=args.gpt_poll_sec,
            min_confidence=args.gpt_min_confidence,
            tracker_kwargs=tracker_kwargs,
        )

    next_tick = time.time()
    frame_interval = 0 if args.fps_cap <= 0 else 1.0 / args.fps_cap
    print("[info] Running. Press q to quit.")

    reconnect_failures = 0

    destination_marker: dict[str, Optional[Tuple[float, float]]] = {"canvas": None, "world": None}

    def on_map_click(event, x, y, flags, *_):
        if event == cv2.EVENT_LBUTTONUP:
            world = canvas_to_world_pts([[x, y]])[0]
            destination_marker["canvas"] = (int(x), int(y))
            destination_marker["world"] = (float(world[0]), float(world[1]))
            print(
                f"[dest] Selected target at world=({world[0]:.2f} ft, {world[1]:.2f} ft) "
                f"≈ ({world[0]*12:.1f}\", {world[1]*12:.1f}\")"
            )
        elif event == cv2.EVENT_RBUTTONUP or (event == cv2.EVENT_MBUTTONUP):
            destination_marker["canvas"] = None
            destination_marker["world"] = None
            print("[dest] Cleared destination marker.")

    if args.show_map:
        cv2.namedWindow("map", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("map", on_map_click)

# Main loop
    try:
        while True:
            frame = None
            if cap:
                ok, frame = cap.read()
                if not ok:
                    reconnect_failures += 1
                    print(f"[warn] Video capture read failed ({reconnect_failures}/{MAX_RECONNECT_ATTEMPTS}); attempting reconnect.")
                    cap.release()
                    time.sleep(RECONNECT_WAIT_SEC)
                    cap = cv2.VideoCapture(rtsp_source)
                    if not cap.isOpened():
                        print("[error] Reconnect attempt failed: unable to reopen source.")
                        break
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    if reconnect_failures >= MAX_RECONNECT_ATTEMPTS:
                        print("[error] Maximum reconnect attempts exceeded; stopping tracker loop.")
                        break
                    continue
                reconnect_failures = 0
                now = time.time()
            else:
                # No capture; idle until next update interval
                time.sleep(frame_interval if frame_interval else 0.1)
                now = time.time()

            skip_yolo = False
            if frame_interval and now < next_tick:
                skip_yolo = True
            else:
                next_tick = now + (frame_interval if frame_interval else 0.1)

            res = None
            if model is not None and frame is not None and not skip_yolo:
                try:
                    results = model.track(
                        source=frame,
                        stream=True,
                        persist=True,
                        conf=args.conf,
                        iou=args.iou,
                        imgsz=args.imgsz,
                        max_det=args.max_det,
                        tracker="bytetrack.yaml",
                        verbose=False,
                    )
                    res = next(results)
                except StopIteration:
                    pass
                except Exception as exc:
                    print(f"[warn] YOLO tracking failed: {exc}")
                    res = None

            # Prepare canvases
            annotated = frame.copy() if frame is not None else None
            map_canvas = draw_map_background()
            gpt_snapshot = gpt_state.snapshot()
            cutebot_drawn = False

            if res is not None and res.boxes is not None and len(res.boxes) > 0:
                boxes = res.boxes.xyxy.cpu().numpy()
                cls   = res.boxes.cls.cpu().numpy().astype(int)
                conf  = res.boxes.conf.cpu().numpy()
                ids   = res.boxes.id
                ids   = ids.cpu().numpy().astype(int) if ids is not None else np.full(len(boxes), -1)

                detections = []
                for (xyxy, c, s, tid) in zip(boxes, cls, conf, ids):
                    name = model.names.get(int(c), str(c))
                    if KEEP_CLASSES and name not in KEEP_CLASSES:
                        continue

                    track_key = tid if tid >= 0 else f"{name}_det"
                    if name in SINGLETON_CLASSES:
                        track_key = f"{name}_singleton"

                    detections.append({
                        "xyxy": xyxy,
                        "class_id": int(c),
                        "conf": float(s),
                        "tid": int(tid),
                        "name": name,
                        "track_key": track_key,
                    })

                if detections:
                    filtered = []
                    singleton_best = {}
                    for det in detections:
                        if det["name"] in SINGLETON_CLASSES:
                            current = singleton_best.get(det["name"])
                            if current is None or det["conf"] > current["conf"]:
                                singleton_best[det["name"]] = det
                        else:
                            filtered.append(det)
                    filtered.extend(singleton_best.values())

                    for det in filtered:
                        xyxy = det["xyxy"]
                        name = det["name"]
                        conf_score = det["conf"]
                        tid = det["tid"]
                        track_key = det["track_key"]

                        x1, y1, x2, y2 = map(int, xyxy)
                        col = color_for_id(track_key if name in SINGLETON_CLASSES or tid < 0 else tid)
                        if annotated is not None:
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), col, 2)
                            if name in SINGLETON_CLASSES:
                                label = f"{name} {conf_score:.2f}"
                            else:
                                label = f"{name}#{tid} {conf_score:.2f}" if tid >= 0 else f"{name} {conf_score:.2f}"
                            cv2.putText(annotated, label, (x1, max(20, y1-6)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

                        bc = bottom_center(xyxy)
                        world_xy = project_points(H, [bc])[0]
                        canvas_pt = world_to_canvas_pts([world_xy])[0]

                        trails[track_key].append((world_xy[0], world_xy[1]))
                        pts_world = np.array(trails[track_key], dtype=np.float32)
                        pts_canvas = world_to_canvas_pts(pts_world)
                    for i in range(1, len(pts_canvas)):
                        p0 = tuple(pts_canvas[i - 1])
                        p1 = tuple(pts_canvas[i])
                        cv2.line(map_canvas, p0, p1, (0, 0, 0), 6)
                        cv2.line(map_canvas, p0, p1, col, 3)

                    mx, my = clamp_canvas_pt(canvas_pt)
                    marker_base = (mx, my)
                    mlabel = ""
                    if name == "cutebot":
                        pose: Optional[CutebotPose] = gpt_snapshot.get("pose")
                        heading = gpt_snapshot.get("heading")
                        if pose:
                            base_world = (pose.x_ft, pose.y_ft)
                            heading = pose.heading_degrees
                        else:
                            base_world = (float(world_xy[0]), float(world_xy[1]))
                        nose_inches = gpt_snapshot.get("nose_inches")
                        confidence = gpt_snapshot.get("confidence")
                        nose_label = None
                        if nose_inches and confidence is not None:
                            nose_label = f"({nose_inches[0]:.1f},{nose_inches[1]:.1f},{confidence:.2f})"
                        marker_base = draw_cutebot_marker(map_canvas, base_world, heading, nose_label)
                        mlabel = ""
                        cutebot_drawn = True
                    else:
                        cv2.circle(map_canvas, (mx, my), 8, (0, 0, 0), -1)
                        cv2.circle(map_canvas, (mx, my), 5, col, -1)
                        if name in SINGLETON_CLASSES:
                            mlabel = name
                        else:
                            mlabel = f"{name}#{tid}" if tid >= 0 else name
                    if mlabel:
                        cv2.putText(map_canvas, mlabel, (marker_base[0] + 10, marker_base[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
                        cv2.putText(map_canvas, mlabel, (marker_base[0] + 10, marker_base[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2, cv2.LINE_AA)

                    if csv_writer:
                        csv_writer.writerow([int(now), str(track_key), name, conf_score, float(world_xy[0]), float(world_xy[1])])

            if not cutebot_drawn:
                pose: Optional[CutebotPose] = gpt_snapshot.get("pose")
                if pose:
                    base_world = (pose.x_ft, pose.y_ft)
                    heading = pose.heading_degrees
                    nose_inches = gpt_snapshot.get("nose_inches")
                    confidence = gpt_snapshot.get("confidence")
                    nose_label = None
                    if nose_inches and confidence is not None:
                        nose_label = f"({nose_inches[0]:.1f},{nose_inches[1]:.1f},{confidence:.2f})"
                    draw_cutebot_marker(map_canvas, base_world, heading, nose_label)

            heading_history = gpt_snapshot.get("heading_history") or []
            status_text: Optional[str] = None
            if heading_history:
                updated_ts = gpt_snapshot.get("updated_ts") or 0.0
                if updated_ts:
                    age = max(0.0, time.time() - updated_ts)
                    status_text = f"Updated {age:.1f}s ago"
            else:
                if gpt_snapshot.get("pose"):
                    status_text = "Latest GPT pose missing heading."
                elif gpt_snapshot.get("error"):
                    status_text = "GPT polling error."
            draw_heading_history(map_canvas, heading_history, status=status_text)

            dest_canvas = destination_marker["canvas"]
            if dest_canvas:
                dcx, dcy = int(dest_canvas[0]), int(dest_canvas[1])
                cv2.circle(map_canvas, (dcx, dcy), DESTINATION_RADIUS + 3, (0, 0, 0), -1, lineType=cv2.LINE_AA)
                cv2.circle(map_canvas, (dcx, dcy), DESTINATION_RADIUS, DESTINATION_COLOR, -1, lineType=cv2.LINE_AA)
                dest_world = destination_marker.get("world")
                if dest_world:
                    target_text = f"Goal ≈ ({dest_world[0]*12:.1f}\", {dest_world[1]*12:.1f}\")"
                    cv2.putText(
                        map_canvas,
                        target_text,
                        (dcx + 12, dcy + 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 230, 255),
                        2,
                        cv2.LINE_AA,
                    )

            gpt_error = gpt_snapshot.get("error")
            if gpt_error:
                cv2.putText(
                    map_canvas,
                    f"GPT pose: {gpt_error}",
                    (30, CANVAS_H - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (40, 40, 220),
                    2,
                    cv2.LINE_AA,
                )

            # Show
            if args.show_video and annotated is not None:
                draw_calibration_quad(annotated, IMAGE_POINTS)
                draw_pixel_grid(annotated, step=100)
                cv2.imshow("video", annotated)
            if args.show_map or writer:
                cv2.imshow("map", map_canvas) if args.show_map else None
                if writer:
                    writer.write(map_canvas)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
    except KeyboardInterrupt:
        print("\n[info] Interrupted by user; shutting down.")
    finally:
        if writer:
            writer.release()
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        if gpt_stop_event:
            gpt_stop_event.set()
        if gpt_thread:
            gpt_thread.join(timeout=2.0)
        print("[info] Done.")
    
if __name__ == "__main__":
    main()
