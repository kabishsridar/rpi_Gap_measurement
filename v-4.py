from picamera2 import Picamera2
import cv2 as cv
import numpy as np
import time

# Camera init
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (4056, 3040)})
picam2.configure(config)
picam2.start()
time.sleep(2)

# Resize settings
resize_width = 1280
resize_height = 960

# Real-world object size in mm
REAL_WIDTH_MM = 75.0
REAL_HEIGHT_MM = 25.0

# HSV range for red-like color
lower_hsv = np.array([0, 0, 200])
upper_hsv = np.array([180, 30, 255])

while True:
    # Capture and resize
    full_frame = picam2.capture_array()
    frame = cv.resize(full_frame, (resize_width, resize_height))

    # HSV conversion & threshold
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower_hsv, upper_hsv)

    # Apply edge detection to the mask
    edges = cv.Canny(mask, 100, 200)

    # Find contours on the original mask (not on edges)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Create a colored mask to draw text and edges
    mask_colored = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)

    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)

        if 30 < w < 600 and 10 < h < 300:
            aspect_ratio = w / float(h)
            if 2.0 < aspect_ratio < 4.5:
                # Draw rectangle on both mask and original frame
                cv.rectangle(mask_colored, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Draw edges inside detected region on mask
                roi_edges = edges[y:y+h, x:x+w]
                mask_colored[y:y+h, x:x+w][roi_edges != 0] = (255, 0, 0)  # Blue edges

                # Put real-world size text on mask
                cv.putText(mask_colored, f"{REAL_WIDTH_MM}mm x {REAL_HEIGHT_MM}mm", 
                           (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Display windows
    cv.imshow("Frame", frame)
    cv.imshow("Mask with Size & Edges", mask_colored)

    key = cv.waitKey(1)
    if key == ord('q'):
        break

cv.destroyAllWindows()
picam2.stop()
