import argparse, collections, time, csv, json, os, sys
from pathlib import Path

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
CANVAS_W = 865
CANVAS_H = 934

# 4) Background image for the top-down map
MAP_BACKGROUND = "demo/IndoorSimOverhead.jpg"
BACKGROUND_PROPERTY_PTS = np.array([
    [160, 770],   # corresponds to WORLD (0,0) bottom-left
    [705, 770],   # corresponds to (W,0) bottom-right
    [705,  95],   # corresponds to (W,H) top-right
    [160,  95],   # corresponds to (0,H) top-left
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

def world_to_canvas_pts(world_xy):
    pts = np.asarray(world_xy, np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H_world2canvas).reshape(-1, 2)
    return out.astype(int)


MAP_BG = cv2.imread(MAP_BACKGROUND)
if MAP_BG is None:
    raise FileNotFoundError(f"Background image not found: {MAP_BACKGROUND}")
MAP_BG = cv2.resize(MAP_BG, (CANVAS_W, CANVAS_H))

# draw panel outline once for sanity checking
for i in range(4):
    p1 = tuple(BACKGROUND_PROPERTY_PTS[i].astype(int))
    p2 = tuple(BACKGROUND_PROPERTY_PTS[(i + 1) % 4].astype(int))
    cv2.line(MAP_BG, p1, p2, (255, 255, 255), 2)

def draw_map_background():
    # Slightly brighten the background so overlays stand out
    canvas = cv2.convertScaleAbs(MAP_BG, alpha=1.1, beta=8)
    draw_world_grid(canvas)
    draw_world_axes(canvas)
    return canvas

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
    model = YOLO(args.model)
    rtsp_source = args.rtsp or require_env("WYZE_TABLETOP_RTSP")

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

    next_tick = 0
    frame_interval = 0 if args.fps_cap <= 0 else 1.0 / args.fps_cap
    print("[info] Running. Press q to quit.")

    reconnect_failures = 0

    while True:
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
        if frame_interval and now < next_tick:
            # just display/map background if throttled
            if args.show_video:
                cv2.imshow("video", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if args.show_map or writer:
                canvas = draw_map_background()
                cv2.imshow("map", canvas) if args.show_map else None
                if writer: writer.write(canvas)
            continue
        next_tick = now + frame_interval

        # Track on the frame
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

# Prepare canvases
        annotated = frame.copy()
        map_canvas = draw_map_background()

        if res.boxes is not None and len(res.boxes) > 0:
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
                    cv2.circle(map_canvas, (mx, my), 8, (0, 0, 0), -1)
                    cv2.circle(map_canvas, (mx, my), 5, col, -1)
                    if name in SINGLETON_CLASSES:
                        mlabel = name
                    else:
                        mlabel = f"{name}#{tid}" if tid >= 0 else name
                    cv2.putText(map_canvas, mlabel, (mx + 10, my - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(map_canvas, mlabel, (mx + 10, my - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2, cv2.LINE_AA)

                    if csv_writer:
                        csv_writer.writerow([int(now), str(track_key), name, conf_score, float(world_xy[0]), float(world_xy[1])])

        # Show
        if args.show_video:
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

    if writer: writer.release()
    cap.release()
    cv2.destroyAllWindows()
    print("[info] Done.")
    
if __name__ == "__main__":
    main()
