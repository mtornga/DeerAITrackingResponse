import argparse, collections, time, csv, json, os, sys
import cv2
import numpy as np
from pathlib import Path
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

# ---------------------- CONFIG (EDIT THESE) ----------------------
# 1) Camera RTSP and YOLO model
MODEL_PATH = "runs/detect/train4/weights/best.pt"

# 2) Map/world size in real units (e.g., feet)
WORLD_W_FT = 3   # width of your property (X)
WORLD_H_FT = 3   # height of your property (Y)

# 3) Canvas (pixels) for the 2D map view (matches background image size)
CANVAS_W = 865
CANVAS_H = 934

# 4) Background image for the top-down map
MAP_BACKGROUND = "demo/IndoorSimOverhead.jpg"
BACKGROUND_PROPERTY_PTS = np.array([
    [160,  90],   # corresponds to WORLD (0,0)
    [705,  95],   # corresponds to (W,0)
    [705, 770],   # corresponds to (W,H)
    [160, 770],   # corresponds to (0,H)
], dtype=np.float32)

# 5) Four corresponding points for homography (order: TL, TR, BR, BL)
#    IMAGE_POINTS are pixel coords in the camera frame where the property corners appear.
#    WORLD_POINTS are real-world coords (e.g., feet) of those corners in your world frame.
#    Define your world frame so (0,0) is the property’s top-left and (W,H) is bottom-right.
IMAGE_POINTS = np.array([
    [475,  390],   # top-left  corner in image (x,y)
    [1433, 424],   # top-right
    [1914, 1100],   # bottom-right
    [  0,  1040],  # bottom-left
], dtype=np.float32)

WORLD_POINTS = np.array([
    [0.0,         0.0        ],   # top-left in feet
    [WORLD_W_FT,  0.0        ],   # top-right
    [WORLD_W_FT,  WORLD_H_FT ],   # bottom-right
    [0.0,         WORLD_H_FT ],   # bottom-left
], dtype=np.float32)

# Optional: filter to specific classes (names in your model)
KEEP_CLASSES = {"horse", "alien_maggie", "cutebot"}  # empty set = keep all
CONF_THR = 0.2
IOU_THR  = 0.45
TRAIL_LEN = 60  # points per track in world space
SAVE_EVENTS_CSV = "detections_world.csv"  # set to "" to disable
# ---------------------------------------------------------------


def color_for_id(i: int):
    np.random.seed(i)
    return tuple(int(x) for x in np.random.randint(80, 255, size=3))

def bottom_center(xyxy):
    x1, y1, x2, y2 = map(float, xyxy)
    return (x1 + x2) / 2.0, y2

def build_homography(image_pts, world_pts):
    # Homography maps image->world (planar) using Direct Linear Transform
    H, mask = cv2.findHomography(image_pts, world_pts, method=0)
    if H is None:
        raise RuntimeError("Homography could not be computed. Check your points.")
    return H


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
    return cv2.convertScaleAbs(MAP_BG, alpha=1.1, beta=8)

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

def draw_grid(img, step=100, color=(60, 60, 60)):
    h, w = img.shape[:2]
    for x in range(0, w, step):
        cv2.line(img, (x, 0), (x, h), color, 1)
        cv2.putText(img, str(x), (x + 2, 18), cv2.FONT_HERSHEY_PLAIN, 1, color, 1, cv2.LINE_AA)
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w, y), color, 1)
        cv2.putText(img, str(y), (2, y - 4), cv2.FONT_HERSHEY_PLAIN, 1, color, 1, cv2.LINE_AA)

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
        H = build_homography(IMAGE_POINTS, WORLD_POINTS)

    # Model + capture
    model = YOLO(args.model)
    rtsp_source = args.rtsp or require_env("WYZE_TABLETOP_RTSP")

    cap = cv2.VideoCapture(rtsp_source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {rtsp_source}")

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

    while True:
        ok, frame = cap.read()
        if not ok:
            break

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

            for (xyxy, c, s, tid) in zip(boxes, cls, conf, ids):
                name = model.names.get(int(c), str(c))
                if KEEP_CLASSES and name not in KEEP_CLASSES:
                    continue

                # Draw bbox on video
                x1, y1, x2, y2 = map(int, xyxy)
                col = color_for_id(int(tid) if tid >= 0 else c)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), col, 2)
                label = f"{name}#{tid} {s:.2f}" if tid >= 0 else f"{name} {s:.2f}"
                cv2.putText(annotated, label, (x1, max(20, y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

                # Ground contact point in image
                bc = bottom_center(xyxy)
                # Project to WORLD (feet)
                world_xy = project_points(H, [bc])[0]  # (X_ft, Y_ft)

                # Convert to canvas px using world->canvas homography
                canvas_pt = world_to_canvas_pts([world_xy])[0]

                # Update trails in world coords (store feet for logging)
                if tid >= 0:
                    trails[tid].append((world_xy[0], world_xy[1]))
                    # Draw trail in canvas coords
                    pts_world = np.array(trails[tid], dtype=np.float32)
                    pts_canvas = world_to_canvas_pts(pts_world)
                    for i in range(1, len(pts_canvas)):
                        p0 = tuple(pts_canvas[i - 1])
                        p1 = tuple(pts_canvas[i])
                        cv2.line(map_canvas, p0, p1, (0, 0, 0), 6)
                        cv2.line(map_canvas, p0, p1, col, 3)

                # Draw point + label on map
                mx, my = clamp_canvas_pt(canvas_pt)
                cv2.circle(map_canvas, (mx, my), 8, (0, 0, 0), -1)
                cv2.circle(map_canvas, (mx, my), 5, col, -1)
                mlabel = f"{name}#{tid}" if tid >= 0 else name
                cv2.putText(map_canvas, mlabel, (mx + 10, my - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(map_canvas, mlabel, (mx + 10, my - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2, cv2.LINE_AA)

                # Optional CSV log
                if csv_writer and tid >= 0:
                    csv_writer.writerow([int(now), int(tid), name, float(s), float(world_xy[0]), float(world_xy[1])])

        # Show
        if args.show_video:
            draw_calibration_quad(annotated, IMAGE_POINTS)
            draw_grid(annotated, step=100)
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
