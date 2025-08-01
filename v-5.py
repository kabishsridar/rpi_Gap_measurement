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

# HSV color range for detection (adjust as needed)
lower_hsv = np.array([0, 30, 200])
upper_hsv = np.array([10, 255, 255])

# Resize parameters
resize_width = 1280
resize_height = 960  # Maintain aspect ratio if needed

while True:
    # Capture full-resolution frame
    full_frame = picam2.capture_array()
    
    # Resize for processing and display
    frame = cv.resize(full_frame, (resize_width, resize_height))

    # Convert to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Threshold using HSV bounds
    mask = cv.inRange(hsv, lower_hsv, upper_hsv)
    result = cv.bitwise_and(frame, frame, mask=mask)

    # Find contours of detected regions
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        
        if 30 < w < 600 and 10 < h < 300:  # Tune based on resized image
            aspect_ratio = w / float(h)
            if 2.0 < aspect_ratio < 4.5:  # Based on 75mm x 25mm
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv.putText(frame, f"Object Detected", (x, y - 10), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show the frames
    cv.imshow("Detected Object", frame)
    cv.imshow("Mask", mask)

    key = cv.waitKey(1)
    if key == ord('q'):
        break

cv.destroyAllWindows()
picam2.stop()
