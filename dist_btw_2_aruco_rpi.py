import cv2 as cv
from cv2 import aruco
import numpy as np
from picamera2 import Picamera2
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

# Load camera calibration data
calib_data_path = "/home/kabish/python_projects/rpi_Gap_measurement/calibrated_data/MultiMatrix_rpi.npz"
calib_data = np.load(calib_data_path)

cam_mat = calib_data["camMatrix"]
dist_coef = calib_data["distCoef"]

# Set marker size in centimeters
MARKER_SIZE_CM = 10
MARKER_SIZE_MM = MARKER_SIZE_CM * 10  # for mm conversion

# ArUco settings
marker_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)
param_markers = aruco.DetectorParameters()
while True:
    frame = picam2.capture_array()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    marker_corners, marker_IDs, reject = aruco.detectMarkers(
        gray, marker_dict, parameters=param_markers
    )

    centers = {}
    tVecs_dict = {}

    if marker_corners and marker_IDs is not None:
        rVecs, tVecs, _ = aruco.estimatePoseSingleMarkers(
            marker_corners, MARKER_SIZE_CM, cam_mat, dist_coef
        )

        for i, (ids, corners) in enumerate(zip(marker_IDs, marker_corners)):
            corners = corners.reshape(4, 2).astype(int)
            center_x = int(np.mean(corners[:, 0]))
            center_y = int(np.mean(corners[:, 1]))
            centers[ids[0]] = (center_x, center_y)
            tVecs_dict[ids[0]] = tVecs[i][0]

            # Draw marker and axis
            cv.polylines(frame, [corners], True, (0, 255, 255), 2)
            cv.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)
            cv.drawFrameAxes(frame, cam_mat, dist_coef, rVecs[i], tVecs[i], 3)
            cv.putText(frame, f"ID:{ids[0]}", (center_x, center_y - 10),
                       cv.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 2)

        # If at least two markers are found
        if len(centers) >= 2:
            marker_ids = list(centers.keys())[:2]  # First two detected markers
            pt1, pt2 = centers[marker_ids[0]], centers[marker_ids[1]]
            pixel_distance = int(np.linalg.norm(np.array(pt1) - np.array(pt2)))

            # Draw line and pixel distance
            cv.line(frame, pt1, pt2, (0, 0, 255), 2)
            mid_pt = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
            cv.putText(frame, f"Pixel Dist: {pixel_distance}", (0, 400),
                       cv.FONT_HERSHEY_PLAIN, 1.4, (0, 0, 255), 2)

            # Calculate 3D distance in mm
            vec1 = tVecs_dict[marker_ids[0]]
            vec2 = tVecs_dict[marker_ids[1]]
            mm_distance = np.linalg.norm(np.array(vec1) - np.array(vec2)) * 10  # cm to mm

            cv.putText(frame, f"Real Dist: {round(mm_distance, 1)} mm", (0, 450),
                       cv.FONT_HERSHEY_PLAIN, 1.4, (0, 0, 255), 2)

    cv.imshow("Aruco Marker Distance", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv.destroyAllWindows()
