import cv2 as cv
import numpy as np
from picamera2 import Picamera2
import time

# --- Reference Calibration ---
REF_DISTANCE = 350  # mm
REF_WIDTH_PX = 303
REF_HEIGHT_PX = 102
OBJ_WIDTH_MM = 75
OBJ_HEIGHT_MM = 25

# Frame size
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# HSV for white
LOWER_WHITE = np.array([0, 0, 200])
UPPER_WHITE = np.array([180, 30, 255])

# Initialize camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)})
picam2.configure(config)
picam2.start()
time.sleep(2)

while True:
    frame = picam2.capture_array()
    resized = cv.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    hsv = cv.cvtColor(resized, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, LOWER_WHITE, UPPER_WHITE)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    object_detected = False

    for cnt in contours:
        if cv.contourArea(cnt) > 500:
            rect = cv.minAreaRect(cnt)
            box = cv.boxPoints(rect)
            box = np.intp(box)

            width_px = min(rect[1])
            height_px = max(rect[1])

            # --- Estimate Distance ---
            scale_w = REF_WIDTH_PX / width_px
            scale_h = REF_HEIGHT_PX / height_px
            estimated_distance = REF_DISTANCE * ((scale_w + scale_h) / 2)

            # --- Compute mm/px dynamically ---
            px_width_at_current = OBJ_WIDTH_MM / width_px
            px_height_at_current = OBJ_HEIGHT_MM / height_px
            mm_per_px = (px_width_at_current + px_height_at_current) / 2

            width_mm = round(width_px * mm_per_px, 2)
            height_mm = round(height_px * mm_per_px, 2)
            est_dist = round(estimated_distance, 1)

            # --- Display results ---
            cv.drawContours(resized, [box], 0, (0, 255, 0), 2)
            cx, cy = int(rect[0][0]), int(rect[0][1])

            cv.putText(resized, f"{int(width_px)}px / {width_mm}mm", (cx - 80, cy - 20),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv.putText(resized, f"{int(height_px)}px / {height_mm}mm", (cx - 80, cy + 5),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv.putText(resized, f"Distance: {est_dist} mm", (cx - 80, cy + 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            object_detected = True
            break

    status = "Object Detected" if object_detected else "Object Not Detected"
    color = (0, 255, 0) if object_detected else (0, 0, 255)
    cv.putText(resized, status, (30, 40), cv.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    cv.imshow("Auto Distance & Measurement", resized)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
picam2.close()
