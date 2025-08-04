import cv2 as cv
import numpy as np
from picamera2 import Picamera2
import time

# Start Pi Camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (1280, 720)})
picam2.configure(config)
picam2.start()
time.sleep(1)

# White HSV range
lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 30, 255])

while True:
    frame = picam2.capture_array()

    # Convert to HSV and mask white areas
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower_white, upper_white)

    # Optional: Morphological clean-up
    kernel = np.ones((5, 5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 1000:  # Filter small noise
            rect = cv.minAreaRect(cnt)
            box = cv.boxPoints(rect)
            box = np.int0(box)

            # Get width and height from rect
            (center_x, center_y), (width, height), angle = rect
            width_px, height_px = int(width), int(height)

            width_mm = width * 0.247524752475
            # cv.putText(frame, f"W: {width_mm}mm", ((int(center_x - 60), int(center_y - 90))),
                       # cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            height_mm = height * 0.247524752475
            # cv.putText(frame, f"H: {height_mm}mm", ((int(center_x - 60), int(center_y - 60))),
                       # cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Draw rectangle and annotate dimensions
            cv.drawContours(frame, [box], 0, (0, 255, 0), 2)
            cv.putText(frame, f"W: {width_px}px", (int(center_x - 60), int(center_y - 10)),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv.putText(frame, f"H: {height_px}px", (int(center_x - 60), int(center_y + 20)),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                       
    # Display results
    cv.imshow("Frame", frame)
    cv.imshow("White Mask", mask)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
# the width of the block is 101 px which is 25 mm
# so 25/101 = 0.247524752475 This is the mm per pixel ratio