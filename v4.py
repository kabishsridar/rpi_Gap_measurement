import cv2 as cv
import numpy as np
from picamera2 import Picamera2
import time

#  THE SIZE OF THE LARGER BLOCK IS PERFECT BUT THE SMALLER BLOCK SHOULD BE 25 X 50 INSTEAD OF 35 X 70

# === Real block dimensions in mm ===
block_width_mm = 75.0
block_height_mm = 25.0
real_diag_mm = np.sqrt(block_width_mm**2 + block_height_mm**2)  # ~79.06 mm

# === Calibration data from Excel (Distance_mm, Width_px, Height_px) ===
calib_data = [
    (300, 116, 357),
    (350, 102, 303)
]

# Compute pixel diagonals
diagonals = []
for dist, w_px, h_px in calib_data:
    diag_px = np.sqrt(w_px**2 + h_px**2)
    diagonals.append((dist, diag_px))

# Compute focal length using multiple calibration points
focals = [(diag_px * dist) / real_diag_mm for dist, diag_px in diagonals]
focal_length = sum(focals) / len(focals)

# === Start PiCamera2 ===
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (1280, 720)})
picam2.configure(config)
picam2.start()
time.sleep(2)

def detect_white_block(frame):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    mask = cv.inRange(hsv, lower_white, upper_white)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    status = "Not Detected"

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 1000:
            rect = cv.minAreaRect(cnt)
            box = cv.boxPoints(rect)
            box = np.int0(box)

            width_px = min(rect[1])
            height_px = max(rect[1])
            pixel_diag = np.sqrt(width_px**2 + height_px**2)

            distance_mm = (real_diag_mm * focal_length) / pixel_diag

            # Convert pixel dims to mm
            width_mm = (width_px * distance_mm) / focal_length
            height_mm = (height_px * distance_mm) / focal_length

            # Draw and label
            cv.drawContours(frame, [box], 0, (0, 255, 0), 2)
            cv.putText(frame, f"Width: {width_px:.1f}px ({width_mm:.1f}mm)", (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv.putText(frame, f"Height: {height_px:.1f}px ({height_mm:.1f}mm)", (10, 60),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            status = "Object Detected"
            break

    return frame, status

# === Main Loop ===
while True:
    frame = picam2.capture_array()
    frame_resized = cv.resize(frame, (960, 540))

    result_frame, status = detect_white_block(frame_resized)
    cv.putText(result_frame, f"Status: {status}", (10, 510), cv.FONT_HERSHEY_SIMPLEX, 0.7,
               (0, 0, 255) if status == "Not Detected" else (0, 255, 0), 2)

    cv.imshow("White Block Detection", result_frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
picam2.stop()
