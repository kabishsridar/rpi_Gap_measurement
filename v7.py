import cv2 as cv
import numpy as np
from picamera2 import Picamera2
import time

# === Calibrated focal length (from 75x25 mm block at 350 mm) ===
focal_length = 1419.1  # Based on earlier calibration
ref_diag_mm = np.sqrt(75**2 + 25**2)

# === Start PiCamera2 ===
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (1280, 720)})
picam2.configure(config)
picam2.start()
time.sleep(2)

def detect_white_blocks(frame):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    mask = cv.inRange(hsv, lower_white, upper_white)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    detected = 0

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 1000:
            rect = cv.minAreaRect(cnt)
            box = cv.boxPoints(rect)
            box = np.int0(box)

            width_px = min(rect[1])
            height_px = max(rect[1])
            pixel_diag = np.sqrt(width_px**2 + height_px**2)

            # Estimate distance using diagonal
            distance_mm = (ref_diag_mm * focal_length) / pixel_diag

            # Convert to real-world units
            width_mm = (width_px * distance_mm) / focal_length
            height_mm = (height_px * distance_mm) / focal_length

            # Draw box
            cv.drawContours(frame, [box], 0, (0, 255, 0), 2)

            # Annotate with dimensions
            cx, cy = int(rect[0][0]), int(rect[0][1])
            label = f"{width_mm:.1f}mm x {height_mm:.1f}mm"
            cv.putText(frame, label, (cx - 60, cy), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                       (255, 255, 255), 2)

            detected += 1

    return frame, detected

# === Main Loop ===
while True:
    frame = picam2.capture_array()
    frame_resized = cv.resize(frame, (960, 540))

    result_frame, num_blocks = detect_white_blocks(frame_resized)
    status = f"{num_blocks} Block(s) Detected" if num_blocks > 0 else "No Block Detected"
    color = (0, 255, 0) if num_blocks > 0 else (0, 0, 255)

    cv.putText(result_frame, f"Status: {status}", (10, 510), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv.imshow("Multiple Block Detection", result_frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
picam2.stop()
