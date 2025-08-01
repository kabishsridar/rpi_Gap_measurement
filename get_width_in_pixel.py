from picamera2 import Picamera2
import cv2 as cv
import numpy as np
import time 

picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (1280, 720)},
    lores={"size": (640, 480)},
    display="lores"
)
picam2.configure(config)
picam2.start()
time.sleep(1)

calib_data_path = "/home/kabish/python_projects/rpi_Gap_measurement/calibrations/calibrated_data/MultiMatrix_rpi.npz"
calib_data = np.load(calib_data_path)
cam_mat = calib_data["camMatrix"]
dist_coef = calib_data["distCoef"]

lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 30, 255])

while True:
    frame = picam2.capture_array('main')
    undistorted = cv.undistort(frame, cam_mat, dist_coef)
    
    hsv = cv.cvtColor(undistorted, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower_white, upper_white)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    x, y, w, h = cv.boundingRect(mask)

    # Only show if object detected (filter small noise)
    if w > 10 and h > 10:
        cv.rectangle(undistorted, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(undistorted, f"Width: {w} px", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX,
                   0.6, (255, 0, 0), 2)

    cv.imshow('Undistorted Frame', undistorted)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv.destroyAllWindows()
# width in pixel = 103