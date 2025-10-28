import argparse, collections, time
import cv2
import numpy as np
from ultralytics import YOLO

# ---------- Utils ----------
def color_for_id(i: int):
    # deterministic BGR color from track id
    np.random.seed(i)
    return tuple(int(x) for x in np.random.randint(64, 255, size=3))

def box_center(xyxy):
    x1, y1, x2, y2 = map(float, xyxy)
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="YOLO + ByteTrack wildlife/object tracker")
    ap.add_argument(
        "--rtsp",
        default="demo/tracking_milestone.mp4",
        help="RTSP URL (or local file path)",
    )
    ap.add_argument("--model", default="best.pt", help="Path to trained YOLO model")
    ap.add_argument("--conf", type=float, default=0.2, help="Confidence threshold")
    ap.add_argument("--iou",  type=float, default=0.45, help="NMS IoU threshold")
    ap.add_argument("--classes", default="", help="Comma list of class names to keep (optional)")
    ap.add_argument("--show", action="store_true", help="Show window")
    ap.add_argument("--save", default="", help="Optional output video path (e.g., out.mp4)")
    ap.add_argument("--trail", type=int, default=30, help="Trail length (points per track)")
    ap.add_argument("--fps_cap", type=float, default=0, help="Max processing FPS (0=unlimited)")
    ap.add_argument("--imgsz", type=int, default=1280, help="Inference image size (pixels)")
    ap.add_argument("--max_det", type=int, default=300, help="Max detections per frame")
    args = ap.parse_args()

    model = YOLO(args.model)

    # Prepare class filter (names -> indices)
    keep_class_idxs = None
    if args.classes.strip():
        wanted = {c.strip().lower() for c in args.classes.split(",")}
        name_to_idx = {v.lower(): k for k, v in model.names.items()}
        keep_class_idxs = [name_to_idx[c] for c in wanted if c in name_to_idx]
        if not keep_class_idxs:
            print(f"[warn] None of {wanted} found in model classes: {model.names}")
            keep_class_idxs = None

    cap = cv2.VideoCapture(args.rtsp)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.rtsp}")

    # Optional writer
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))
        print(f"[info] Saving annotated video to {args.save} at ~{fps:.1f} FPS")

    # Trails: id -> deque of (x,y)
    trails = collections.defaultdict(lambda: collections.deque(maxlen=args.trail))

    # Throttle
    next_tick = 0
    frame_interval = 0 if args.fps_cap <= 0 else 1.0 / args.fps_cap

    # Use Ultralytics streaming tracker (ByteTrack)
    # We’ll pull frames ourselves to keep control over throttling & display.
    print("[info] Starting… press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[info] End of stream or read error.")
            break

        now = time.time()
        if frame_interval and now < next_tick:
            # skip processing to respect FPS cap
            if args.show:
                cv2.imshow("tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if writer: writer.write(frame)
            continue
        next_tick = now + frame_interval

        # Run tracking on this single frame
        results = model.track(
            source=frame,
            stream=True,               # return generator
            verbose=False,
            persist=True,              # keep tracks across calls
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            max_det=args.max_det,
            tracker="bytetrack.yaml",  # built-in tracker
        )

        # The generator yields one result per call when source is an image array
        result = next(results)

        annotated = frame.copy()
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            cls   = result.boxes.cls.cpu().numpy().astype(int)
            conf  = result.boxes.conf.cpu().numpy()
            ids   = result.boxes.id
            ids   = ids.cpu().numpy().astype(int) if ids is not None else np.full(len(boxes), -1)

            for (xyxy, c, s, tid) in zip(boxes, cls, conf, ids):
                if keep_class_idxs is not None and c not in keep_class_idxs:
                    continue

                x1, y1, x2, y2 = map(int, xyxy)
                name = model.names.get(int(c), str(c))
                col = color_for_id(int(tid) if tid >= 0 else c)

                # draw bbox
                cv2.rectangle(annotated, (x1, y1), (x2, y2), col, 2)

                # label: class, id, conf
                label = f"{name}"
                if tid >= 0: label += f" #{tid}"
                label += f" {s:.2f}"
                (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), col, -1)
                cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

                # update & draw trail
                cx, cy = box_center(xyxy)
                if tid >= 0:
                    trails[tid].append((cx, cy))
                    pts = list(trails[tid])
                    for i in range(1, len(pts)):
                        cv2.line(annotated, pts[i - 1], pts[i], col, 2)

                # dot at center
                cv2.circle(annotated, (cx, cy), 3, col, -1)

        # HUD
        fps_txt = f"FPS: {cap.get(cv2.CAP_PROP_FPS) or 0:.1f}"
        cv2.putText(annotated, fps_txt, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 3, cv2.LINE_AA)
        cv2.putText(annotated, fps_txt, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 1, cv2.LINE_AA)

        if args.show:
            cv2.imshow("tracker", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if writer:
            writer.write(annotated)

    cap.release()
    if writer: writer.release()
    if args.show: cv2.destroyAllWindows()
    print("[info] Done.")

if __name__ == "__main__":
    main()
