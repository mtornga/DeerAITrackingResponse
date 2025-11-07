# Purpose 

This is a software platform for managing deer and other wildlife on the user's property

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

Create a `.env` file in the project root (see the checked-in sample) with the RTSP connection details for the Wyze camera:

```
WYZE_TABLETOP_RTSP="rtsp://user:password@camera-ip/live"
```

Scripts will read this variable automatically; override it via CLI flags when needed.

Project artifacts are now in the s3 bucket wildlife-ugv on us-east-2