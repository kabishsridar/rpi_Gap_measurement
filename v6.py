import cv2 as cv
import numpy as np
from picamera2 import Picamera2
import time

# === Calibration data for 75x25 mm block ===
# (distance_mm, width_px, height_px)
calib_data = [
    (300, 116, 357),
    (350, 102, 303)
]

calib_real_width_mm = 75.0
calib_real_height_mm = 25.0

# Compute focal lengths from calibration
focal_w_list = [(w_px * d) / calib_real_width_mm for d, w_px, _ in calib_data]
focal_h_list = [(h_px * d) / calib_real_height_mm for d, _, h_px in calib_data]

focal_w = sum(focal_w_list) / len(focal_w_list)
focal_h = sum(focal_h_list) / len(focal_h_list)

# === Start camera ===
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

            # Calculate real size using inverse of focal calibration
            width_mm = (width_px * calib_data[0][0]) / focal_w
            height_mm = (height_px * calib_data[0][0]) / focal_h

            # Use the measured real-world diagonal to compute distance
            diag_px = np.sqrt(width_px**2 + height_px**2)
            real_diag_mm = np.sqrt(width_mm**2 + height_mm**2)
            diag_px_calib = np.sqrt(calib_data[0][1]**2 + calib_data[0][2]**2)
            real_diag_mm_calib = np.sqrt(calib_real_width_mm**2 + calib_real_height_mm**2)

            # Distance based on diagonal ratio (more accurate)
            distance_mm = (real_diag_mm * diag_px_calib * calib_data[0][0]) / (diag_px * real_diag_mm_calib)

            # Draw info
            cv.drawContours(frame, [box], 0, (0, 255, 0), 2)
            cv.putText(frame, f"Width: {width_mm:.1f} mm", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv.putText(frame, f"Height: {height_mm:.1f} mm", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv.putText(frame, f"Distance: {distance_mm:.1f} mm", (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            status = "Object Detected"
            break

    return frame, status

# === Main loop ===
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
