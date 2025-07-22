import cv2 as cv # importing modules
from cv2 import aruco
import numpy as np
from picamera2 import Picamera2
import time

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (1280, 720)}, lores={"size": (640, 480)}, display="lores")
picam2.configure(config)

picam2.start()
time.sleep(1)

cv.namedWindow("RPI connection", cv.WINDOW_NORMAL)

# load in the calibration data
calib_data_path = "/home/kabish/python_projects/calib_data_rpi/MultiMatrix.npz" # assign path

calib_data = np.load(calib_data_path) # loads the data from the caliberated file
print(calib_data.files) # displays the files

cam_mat = calib_data["camMatrix"] # assigning the datas caliberated through the MultiMatrix file
dist_coef = calib_data["distCoef"]
r_vectors = calib_data["rVector"]
t_vectors = calib_data["tVector"]
print(cam_mat, dist_coef, r_vectors, t_vectors)
MARKER_SIZE = 6  # centimeters (measure your printed marker size)

marker_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250) # assigning the dictionary type

param_markers = aruco.DetectorParameters() # detects the parameters and stores it as param_markers (example: < cv2.aruco.DetectorParameters 000001CCB6B03D20>)

while True:
    frame = picam2.capture_array("main")
    frame_bgr = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # converting to grayscale
    marker_corners, marker_IDs, reject = aruco.detectMarkers(
        gray_frame, marker_dict, parameters=param_markers
    ) # detects the markers in the frame using param_markers
    print(f"There are {len(marker_corners)} corners available in the frame")
    if marker_corners:
        rVec, tVec, _ = aruco.estimatePoseSingleMarkers(
            marker_corners, MARKER_SIZE, cam_mat, dist_coef
        ) # Estimates each marker's pose (rotational and translational vector)
        total_markers = range(0, marker_IDs.size)
        for ids, corners, i in zip(marker_IDs, marker_corners, total_markers):
            cv.polylines(
                frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv.LINE_AA
            )
            corners = corners.reshape(4, 2)
            corners = corners.astype(int) # converting to integer
            top_right = corners[0].ravel() # converts the matrix to a single line
            top_left = corners[1].ravel()
            bottom_right = corners[2].ravel()
            bottom_left = corners[3].ravel()

            # Calculating the distance
            distance = np.sqrt(
                tVec[i][0][2] ** 2 + tVec[i][0][0] ** 2 + tVec[i][0][1] ** 2
            )
            # Draw the pose of the marker
            point = cv.drawFrameAxes(frame, cam_mat, dist_coef, rVec[i], tVec[i], 4, 4)
            cv.putText(
                frame,
                f"id: {ids[0]} Dist: {round(distance, 2)}",
                top_right,
                cv.FONT_HERSHEY_PLAIN,
                1.3,
                (0, 0, 255),
                2,
                cv.LINE_AA,
            ) # displays distance and ID
            cv.putText(
                frame,
                f"x:{round(tVec[i][0][0],1)} y: {round(tVec[i][0][1],1)} ",
                bottom_right,
                cv.FONT_HERSHEY_PLAIN,
                1.0,
                (0, 0, 255),
                2,
                cv.LINE_AA,
            ) # displays x and y coordinates
            # print(ids, "  ", corners)
    # cv.namedWindow("RPI connection", cv.WINDOW_NORMAL)
    cv.imshow("frame", frame_bgr)
    key = cv.waitKey(1)
    if key == ord("q"):
        break
picam2.stop()
cv.destroyAllWindows()