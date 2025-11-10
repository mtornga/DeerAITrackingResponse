## Make a few concise, visual demos
1. The calibration process is a great candidate. The agent is using its perception (rtsp capture) and my measuring tape results to improve itself.
2. Camera view to 2d homography is cool.
3. Demo of MCP tools that prove useful. 


## Outdoors
1. Measure the back yard deer detection zone.
2. Pipeline to capture clips, detect, classify, re-identify were POC'd but proved weak due to the models used. Pivoting to fine-tuning of models.

⚙️ Recommended Real-Time Architecture
[Reolink RTSP streams]
          ↓
     (ffmpeg capture)
          ↓
   [YOLOv8/RT-DETR detector]
          ↓
   [DeepSORT / ByteTrack tracker]
          ↓
   [AprilTag or multi-camera transform]
          ↓
   [SQLite/Postgres data store]
          ↓
   [Realtime visualization + prediction model]

🧠 Detection + Tracking layer
✅ Use YOLOv8n-pose or RT-DETR-tiny

These models output bounding boxes plus keypoints, so you can estimate orientation (head vs tail) without extra classification.

Run them at 15–20 FPS on your Mac Mini (CPU or small GPU) for live streams.

Tracker choice:
Tracker	Strength	Notes
ByteTrack	Simple, very robust to low FPS	Great for nighttime or dropped frames
DeepSORT	Embedding-based re-ID	Better if multiple deer overlap

Both maintain a track ID and output:

{ "track_id": 7, "x": 412, "y": 233, "bbox": [x1,y1,x2,y2], "timestamp": 1731180001 }

📐 Coordinate mapping (the “real world” step)

To convert pixel positions to actual ground coordinates:

Calibrate each camera using AprilTags or a checkerboard on your lawn.
→ Use cv2.findHomography() or your existing tabletop_affine.json logic.

Apply homography to every detection:

X_world = H @ [x_center, y_bottom, 1]


giving you feet/inches in the yard coordinate frame.

(Optional) If you have overlapping cameras, fuse trajectories using Kalman filters + multi-camera association.

🧩 Behavior Classification

Once you have stable tracks, add a small posture classifier:

Input: cropped image + optional keypoints.

Output: {"posture": "alert"|"grazing"|"walking"|"running"}.

You can train this lightweight classifier on ~300–500 labeled examples per posture using YOLOv8-cls or a small ViT.

Combine with velocity:

speed = distance(current_xy, prev_xy) / Δt
heading = atan2(Δy, Δx)

🗺️ Path Storage Schema (example)

In SQLite or Postgres:

CREATE TABLE tracks (
    track_id INT,
    species TEXT,
    start_ts TIMESTAMP,
    end_ts TIMESTAMP
);

CREATE TABLE path_points (
    track_id INT,
    timestamp TIMESTAMP,
    x REAL,
    y REAL,
    heading REAL,
    speed REAL,
    posture TEXT
);


This lets you query, plot, and train models on historical paths.

🔮 Forecasting Future Movement

Once you have weeks of trajectories, you can model “likely next steps”:

Approach	When to use	Example
Kalman / Constant-Velocity	quick smoothing & short-term prediction	1–5 sec forecast
ARIMA or HMM	few days of data per deer	basic temporal pattern
LSTM / Transformer-based Trajectory Predictor	long-term learning across many deer	learns preferred travel corridors
Occupancy Grid / Heatmap	spatial frequency over time	visualize “deer highways”

For spatial forecasting, you’ll often rasterize your yard into a grid (1 ft² cells) and learn transition probabilities or use a small CNN that predicts next-step probability.

🎥 Visualization

Use your existing “animation recap” pipeline:

Store track points as JSON.

Render in matplotlib or plotly to replay paths over a map or video frame.

Add color per posture or speed.

Later you can overlay UGV predicted intercept paths.

🚀 Implementation Milestones

Integrate YOLOv8-pose or RT-DETR tracking (real-time pipeline).

Homography calibration for all Reolink cameras.

Write path logger (JSON → SQLite).

Train posture classifier.

Implement forecasting + visualization notebook.


## Indoors
1. Get the cutebot moving. Use agent to self-improve the movements and perception of position on the board.
2. Create scripts to react to horse detection and move to the area. 
3. Pay down some of this helper debt. Probably won't have the coordinate labels or heading labels in the yard. The big green arrow on cutebot is a "maybe" for the outdoor version. 