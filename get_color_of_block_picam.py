import cv2 as cv
import numpy as np
from picamera2 import Picamera2
import time

# Tolerance for HSV range
H_TOL = 10
S_TOL = 50
V_TOL = 50

hsv_frame = None  # Global to access in mouse callback

def mouse_callback(event, x, y, flags, param):
    global hsv_frame
    if event == cv.EVENT_LBUTTONDOWN:
        if hsv_frame is None:
            print("No frame yet!")
            return

        pixel = hsv_frame[y, x]  # Note: y is row, x is column
        h, s, v = int(pixel[0]), int(pixel[1]), int(pixel[2])

        # Calculate lower and upper bounds with clipping
        lower_bound = np.array([
            max(h - H_TOL, 0),
            max(s - S_TOL, 0),
            max(v - V_TOL, 0)
        ])
        upper_bound = np.array([
            min(h + H_TOL, 180),
            min(s + S_TOL, 255),
            min(v + V_TOL, 255)
        ])

        print(f"Clicked HSV: ({h}, {s}, {v})")
        print(f"Lower bound HSV: {lower_bound}")
        print(f"Upper bound HSV: {upper_bound}")

picam2 = Picamera2()
picam2.start()
time.sleep(2)

cv.namedWindow("Original")
cv.setMouseCallback("Original", mouse_callback)

while True:
    frame = picam2.capture_array()
    frame_bgr = cv.cvtColor(frame, cv.COLOR_RGB2BGR)  # Picamera2 gives RGB, convert to BGR for OpenCV display
    hsv_frame = cv.cvtColor(frame_bgr, cv.COLOR_BGR2HSV)

    cv.imshow("Original", frame_bgr)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv.destroyAllWindows()
