import cv2 as cv
import numpy as np
from picamera2 import Picamera2
import time

# --- CONFIGURATIONS ---
OBJECT_WIDTH_MM = 75
OBJECT_HEIGHT_MM = 25
DISTANCE_MM = 350  # Fixed camera distance
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# HSV color range for white
LOWER_WHITE = np.array([0, 0, 200])
UPPER_WHITE = np.array([180, 30, 255])

# Initialize camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)})
picam2.configure(config)
picam2.start()
time.sleep(2)

# Calibration: known size at known distance gives ratio (pixels per mm)
# We'll use this to convert pixel size to mm
def calculate_mm_per_pixel(expected_pixel_width):
    return OBJECT_WIDTH_MM / expected_pixel_width if expected_pixel_width else 1.0

while True:
    frame = picam2.capture_array()
    resized = cv.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    hsv = cv.cvtColor(resized, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, LOWER_WHITE, UPPER_WHITE)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    object_detected = False
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 500:  # Filter small noise
            x, y, w, h = cv.boundingRect(cnt)
            mm_per_pixel = calculate_mm_per_pixel(w)
            width_mm = round(w * mm_per_pixel, 2)
            height_mm = round(h * mm_per_pixel, 2)

            cv.rectangle(resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.putText(resized, f"W: {w}px / {width_mm}mm", (x, y - 25), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv.putText(resized, f"H: {h}px / {height_mm}mm", (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            object_detected = True
            break  # Only detect the first large object

    status_text = "Object Detected" if object_detected else "Object Not Detected"
    color = (0, 255, 0) if object_detected else (0, 0, 255)
    cv.putText(resized, status_text, (30, 40), cv.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    cv.imshow("White Object Detection", resized)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
picam2.close()
