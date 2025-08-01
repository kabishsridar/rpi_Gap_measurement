from picamera2 import Picamera2
import cv2 as cv
import numpy as np
import time

# Initialize PiCamera2
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (4056, 3040)})
picam2.configure(config)
picam2.start()
time.sleep(2)

# Resize settings for performance
resize_width = 1280
resize_height = 960

# Real-world dimensions (used for mm conversion)
REAL_WIDTH_MM = 75.0  # Known object width in mm
REAL_HEIGHT_MM = 25.0  # Known object height in mm
EXPECTED_PIXEL_WIDTH = 320  # You can adjust based on one-time measurement
pixels_per_mm = EXPECTED_PIXEL_WIDTH / REAL_WIDTH_MM

# White color in HSV (low saturation, high brightness)
lower_hsv = np.array([0, 0, 200])
upper_hsv = np.array([180, 30, 255])

while True:
    # Capture frame and resize
    full_frame = picam2.capture_array()
    frame = cv.resize(full_frame, (resize_width, resize_height))

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower_hsv, upper_hsv)
    edges = cv.Canny(mask, 100, 200)

    mask_colored = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)

        # Relaxed filters for white block size
        if 30 < w < 1000 and 10 < h < 500:
            aspect_ratio = w / float(h)
            if 1.5 < aspect_ratio < 4.5:  # Expected: 75 / 25 = 3.0

                width_mm = round(w / pixels_per_mm, 1)
                height_mm = round(h / pixels_per_mm, 1)

                # Draw bounding box
                cv.rectangle(mask_colored, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Draw edges in blue inside bounding box
                roi_edges = edges[y:y + h, x:x + w]
                mask_colored[y:y + h, x:x + w][roi_edges != 0] = (255, 0, 0)

                # Overlay pixel + mm text
                label = f"W: {w}px/{width_mm}mm | H: {h}px/{height_mm}mm"
                cv.putText(mask_colored, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX,
                           0.6, (0, 255, 255), 2)

    # Show windows
    cv.imshow("Original Frame", frame)
    cv.imshow("Mask with Measurements", mask_colored)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
picam2.stop()
