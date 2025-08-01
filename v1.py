import cv2 as cv
import numpy as np
from picamera2 import Picamera2
import time

# --- Calibration Constants ---
MM_PER_PX_WIDTH = 75 / 303      # ≈ 0.2475 mm/px
MM_PER_PX_HEIGHT = 25 / 102     # ≈ 0.2451 mm/px

# Frame Size
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# HSV Range for white
LOWER_WHITE = np.array([0, 0, 200])
UPPER_WHITE = np.array([180, 30, 255])

# Initialize PiCamera2
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
        area = cv.contourArea(cnt)
        if area > 500:  # Skip small objects
            x, y, w, h = cv.boundingRect(cnt)

            # Convert to mm
            width_mm = round(w * MM_PER_PX_WIDTH, 2)
            height_mm = round(h * MM_PER_PX_HEIGHT, 2)

            # Draw box and dimensions
            cv.rectangle(resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.putText(resized, f"{w}px / {width_mm}mm", (x, y - 25),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv.putText(resized, f"{h}px / {height_mm}mm", (x, y - 5),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            object_detected = True
            break  # Only show the first detected object

    # Show status
    status_text = "Object Detected" if object_detected else "Object Not Detected"
    status_color = (0, 255, 0) if object_detected else (0, 0, 255)
    cv.putText(resized, status_text, (30, 40),
               cv.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)

    # Display result
    cv.imshow("Object Measurement", resized)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
picam2.close()
