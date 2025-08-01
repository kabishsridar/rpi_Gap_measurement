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

# Resize settings
resize_width = 1280
resize_height = 960

# Known real-world size (used to calculate pixel-to-mm)
REAL_WIDTH_MM = 75.0
REAL_HEIGHT_MM = 25.0

# HSV color bounds (adjust for your object)
lower_hsv = np.array([0, 0, 200])
upper_hsv = np.array([180, 30, 255])

# Tune this after initial detection — expected pixel width at 350mm distance
expected_pixel_width = 320  # pixel width corresponding to 75 mm
pixels_per_mm = expected_pixel_width / REAL_WIDTH_MM

while True:
    # Capture and resize
    full_frame = picam2.capture_array()
    frame = cv.resize(full_frame, (resize_width, resize_height))

    # Convert to HSV and threshold
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower_hsv, upper_hsv)

    # Edge detection
    edges = cv.Canny(mask, 100, 200)
    mask_colored = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)

    # Find contours
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)

        if 30 < w < 600 and 10 < h < 300:
            aspect_ratio = w / float(h)
            if 2.0 < aspect_ratio < 4.5:
                # Convert to mm
                width_mm = round(w / pixels_per_mm, 1)
                height_mm = round(h / pixels_per_mm, 1)

                # Draw bounding box
                cv.rectangle(mask_colored, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Draw Canny edges in blue
                roi_edges = edges[y:y+h, x:x+w]
                mask_colored[y:y+h, x:x+w][roi_edges != 0] = (255, 0, 0)

                # Text overlay with pixel and mm
                label = f"W: {w}px / {width_mm}mm | H: {h}px / {height_mm}mm"
                cv.putText(mask_colored, label, (x, y - 10), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Show results
    cv.imshow("Original Frame", frame)
    cv.imshow("Mask with Size Info", mask_colored)

    key = cv.waitKey(1)
    if key == ord('q'):
        break

cv.destroyAllWindows()
picam2.stop()
