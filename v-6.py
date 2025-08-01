from picamera2 import Picamera2
import cv2 as cv
import numpy as np
import time

# Initialize Camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (4056, 3040)})
picam2.configure(config)
picam2.start()
time.sleep(2)

# Convert BGR to HSV manually to match your color
lower_hsv = np.array([0, 30, 200])   # Lower bound of red-ish in HSV
upper_hsv = np.array([10, 255, 255]) # Upper bound for red-ish
# You can adjust this range based on lighting and true color

# Constants for physical size & distance
REAL_WIDTH_MM = 75.0
REAL_HEIGHT_MM = 25.0
DISTANCE_MM = 350.0

# Optional: known focal length from calibration
FOCAL_LENGTH_MM = 3.6  # Replace with actual focal length in mm if known
SENSOR_WIDTH_MM = 3.68  # PiCamera v2 sensor width (adjust as per your model)
IMAGE_WIDTH_PX = 4056

# Calculate expected object pixel width based on pinhole model (if needed)
expected_pixel_width = (REAL_WIDTH_MM * IMAGE_WIDTH_PX) / SENSOR_WIDTH_MM / DISTANCE_MM * FOCAL_LENGTH_MM

while True:
    frame = picam2.capture_array()
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Threshold using HSV bounds
    mask = cv.inRange(hsv, lower_hsv, upper_hsv)
    result = cv.bitwise_and(frame, frame, mask=mask)

    # Find contours of detected regions
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        
        # Optional: filter by expected size (in pixels)
        if 100 < w < 2000 and 20 < h < 500:  # Rough range, tune this
            aspect_ratio = w / float(h)
            if 2.5 < aspect_ratio < 4.5:  # Expected aspect ratio = 75 / 25 = 3.0
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv.putText(frame, f"Object Detected", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    

    cv.imshow("Detected Object", frame)
    cv.imshow("Mask", mask)

    key = cv.waitKey(1)
    if key == ord('q'):
        break

cv.destroyAllWindows()
picam2.stop()
