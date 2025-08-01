import cv2 as cv
import numpy as np
import time
from picamera2 import Picamera2
from group_lines import group_lines_by_y  # Ensure this returns list of flat lines [x1, y1, x2, y2]

def get_calib_data():
    calib_data_path = "/home/kabish/python_projects/rpi_Gap_measurement/calibrations/calibrated_data/MultiMatrix_rpi.npz"
    calib_data = np.load(calib_data_path)
    cam_mat = calib_data["camMatrix"]
    dist_coef = calib_data["distCoef"]
    return cam_mat, dist_coef

# Measured using checkerboard, replace with accurate value if needed
PIXEL_TO_MM = 136.4 / 500  # 0.2728 mm/pixel

def detect():
    cam_mat, dist_coef = get_calib_data()

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (1280, 720)},
        lores={"size": (640, 480)},
        display="lores"
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1)

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])

    while True:
        frame = picam2.capture_array()
        if frame is None:
            print("Cannot read image")
            break

        # Undistort
        undistorted = cv.undistort(frame, cam_mat, dist_coef)

        hsv = cv.cvtColor(undistorted, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, lower_white, upper_white)
        res = cv.bitwise_and(undistorted, undistorted, mask=mask)

        gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
        edges = cv.Canny(gray, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)

        lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

        if lines is not None:
            lines = [line[0] for line in lines]  # Flatten from [[[x1,y1,x2,y2]]] to [x1,y1,x2,y2]
            grouped_lines = group_lines_by_y(lines, threshold=10)

            if grouped_lines:
                largest_group = max(grouped_lines, key=len)

                def avg_y(line):
                    x1, y1, x2, y2 = line
                    return (y1 + y2) / 2

                sorted_group = sorted(largest_group, key=avg_y)
                top_line = sorted_group[0]
                bottom_line = sorted_group[-1]

                # Draw lines
                cv.line(undistorted, (top_line[0], top_line[1]), (top_line[2], top_line[3]), (0, 255, 0), 2)
                cv.line(undistorted, (bottom_line[0], bottom_line[1]), (bottom_line[2], bottom_line[3]), (0, 0, 255), 2)

                # Measure vertical distance between lines
                top_y = (top_line[1] + top_line[3]) / 2
                bottom_y = (bottom_line[1] + bottom_line[3]) / 2
                pixel_height = abs(bottom_y - top_y)
                real_height_mm = pixel_height * PIXEL_TO_MM

                cv.putText(undistorted, f"Height: {real_height_mm:.2f} mm",
                           (top_line[0], top_line[1] - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv.imshow("Measurement", undistorted)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    picam2.stop()
    cv.destroyAllWindows()

if __name__ == "__main__":
    detect()
