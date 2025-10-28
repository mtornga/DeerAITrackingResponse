# click_panel_roi.py
import cv2, numpy as np

bg = cv2.imread("IndoorSimCamView.jpg")
pts = []

def cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
        pts.append((x, y))
        cv2.circle(bg, (x, y), 5, (0,255,255), -1)
        cv2.putText(bg, ["TL","TR","BR","BL"][len(pts)-1], (x+6,y-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

cv2.namedWindow("bg", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("bg", cb)

while True:
    cv2.imshow("bg", bg)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print("BACKGROUND_PROPERTY_PTS =", np.array(pts, dtype=np.float32))
