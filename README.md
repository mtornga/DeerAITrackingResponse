# Purpose 

This is a software platform for managing deer and other wildlife on the user's property. 

Inspired by the University of Minnesota paper "Geofenced Unmanned Aerial Robotic Defender for
Deer Detection and Deterrence (GUARD)" https://arxiv.org/pdf/2505.10770 /DeerAITrackingResponse/docs/GeofencedUARDDeer.pdf

# Approach

A full spatial intelligence system. 
1. Detection → find deer and other species in each frame.
2. Tracking → persist identities across frames.
3. Localization → convert pixel positions into yard coordinates (x, y in feet or inches).
4. Behavior state → classify posture, speed, direction.
5. Trajectory analysis → store and forecast path history.

## Structure

### Indoor tabletop simulation

A 24"'x 30" table in my office with a Wyze v2 camera for detection, a reolink E1 for overwatch, objects to stand-in for wildlife, and a CuteBot to move around. 

The milestones here are: 
1. Pull images from the RTSP stream when needed.
2. Train YOLO model for various objects 
3. Locate objects in the environment
4. Route the CuteBot to specified coordinates
5. Iterate with realtime tracking, cutebot behaviors, and more.

For indoor work Wyze should be used primarily to detect and suggest a location. This simulates the low angle outdoor camera that gets a side view of animals.
The Reolink E1 should be used to better determine the location of objects in the 24x30 tabletop.

The board is perceived from multiple angles, but bottom left from the viewpoint of the Wyze camera is considered 0",0", top left is 0",24", top right is 30",24", bottom right is 30",0". 


### Outdoor solution

Four to five Reolink cameras covering different areas of the property. Cameras are tracking actual wildlife. Ability to route unmanned ground vehicle(s), activate other deterrents to protect crops from wildlife. 

Milestones for outdoor:
1. Pull images from RTSP streams. Be efficient with space by quickly converting image data into coordinate data. 
2. Establish property coordinates, obstacles, boundaries. 
3. Visual display of animal and UGV travel paths
4. ML prediction of animal path
5. Workflow for producing animations and videos recapping events. 

## Configuration

Create a `.env` file in the project root (see the checked-in sample) with the RTSP connection details for the cameras:

```
WYZE_TABLETOP_RTSP="rtsp://user:password@camera-ip/live"
```

Scripts will read this variable automatically; override it via CLI flags when needed.

Project artifacts are now in the s3 bucket wildlife-ugv on us-east-2

## Resources
MacBook Pro 2017 Intel i7 Radeon Pro 560 16gb memory
Wyze Cam v2
Reolink E1 Zoom
Reolink Duo 3 PoE
CuteBot Pro with Micro:bit v2
Apriltags
IR floodlight
Ubuntu server with GTX 3080

### Shared storage
- Samba share: `smb://192.168.68.71/deer-share`
- Credentials: username `mtornga`, password `mtornga`
- Server mount point: `/srv/deer-share` (USB drive UUID `F04815E200F815F8`)
- Mac client example:
  - `mkdir -p ~/DeerShare`
  - `mount_smbfs //mtornga@192.168.68.71/deer-share ~/DeerShare`
- Expect a `hello-from-server.txt` smoke-test file at the root after provisioning. Use this share for exchanging tabletop captures and detector outputs that do not belong in the repo.

## Monitoring dashboard

Get a quick health read on the Mac, Ubuntu server, and background jobs by launching the tmux dashboard:

1. Make sure SSH keys are loaded (once per login).
   - `ssh-add --apple-use-keychain ~/.ssh/id_ed25519`
2. Verify `.env` contains the Ubuntu host and repo path you actually use (defaults work for `192.168.68.71` and `~/projects/deer-vision`).
3. From the repo root on the Mac, run:
   - `scripts/deervision_dashboard_tmux.sh`

This spawns/attaches a `deervision-dashboard` tmux session with four panes:

- Mac status (top-left) – wraps `scripts/deervision_status_mac.sh`.
- Ubuntu status (top-right) – SSH + `scripts/deervision_status_ubuntu.sh` (disk, GPU, services, long-running jobs, camera sockets).
- Ubuntu CPU/GPU monitor (bottom-left) – `htop` when available, otherwise `top`.
- Ubuntu logs (bottom-right) – `scripts/deervision_tail_logs.sh` tails `logs/*.log` on the server, emitting a reminder if no log files exist yet.

The scripts automatically source `.env` so you can extend health checks via new env vars (e.g., `DEERVISION_CVAT_URL`, `DEERVISION_STREAMLIT_URL`, or more `*_RTSP` endpoints). Keep the Ubuntu repo in sync so `scripts/deervision_status_ubuntu.sh` and `scripts/deervision_tail_logs.sh` are available at `${DEERVISION_UBUNTU_REPO_PATH}/scripts/`.
