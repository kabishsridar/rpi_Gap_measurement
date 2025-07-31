import cv2 as cv
import numpy as np
from picamera2 import Picamera2
import time

def get_calib_data():
    calib_data_path = "/home/kabish/python_projects/rpi_Gap_measurement/calibrated_data/MultiMatrix_rpi.npz"
    calib_data = np.load(calib_data_path)
    cam_mat = calib_data["camMatrix"]
    dist_coef = calib_data["distCoef"]
    return cam_mat, dist_coef

# Convert pixel Y-coordinate to real-world Y (up to scale)
def pixel_to_world_y(image_y, cam_mat):
    fy = cam_mat[1, 1]  # focal length in y
    cy = cam_mat[1, 2]  # principal point y
    return (image_y - cy) / fy

def detect():
    cam_mat, dist_coeff = get_calib_data()

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
    measurements = []

    # Set once based on known pixel gap vs mm on flat surface
    SCALING_FACTOR_MM = (136.4 / 500)  # e.g., 50 mm corresponds to 100 pixel difference or 132 / 490
# we checked the aruco marker set up, the scale factor calculated above is correct
# because we measured the dimension with the steel scale between the aruco markers
    while True:
        frame = picam2.capture_array()
        if frame is None:
            print("Cannot read image")
            break

        # Undistort using calibrated values
        undistorted_frame = cv.undistort(frame, cam_mat, dist_coeff)

        # HSV thresholding for white
        hsv = cv.cvtColor(undistorted_frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, lower_white, upper_white)
        result = cv.bitwise_and(undistorted_frame, undistorted_frame, mask=mask)

        gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
        contours, _ = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        valid_blocks = []
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area > 5000:
                x, y, w, h = cv.boundingRect(cnt)
                aspect_ratio = w / float(h)
                if 0.3 < aspect_ratio < 3.0:
                    valid_blocks.append((y, x, w, h))

        valid_blocks.sort()

        line_img = undistorted_frame.copy()

        if len(valid_blocks) >= 2:
            y1, x1, w1, h1 = valid_blocks[0]
            y2, x2, w2, h2 = valid_blocks[1]

            top_y = y1 + h1       # bottom of top block
            bottom_y = y2         # top of bottom block

            cx1 = x1 + w1 // 2
            cx2 = x2 + w2 // 2
            center_x = (cx1 + cx2) // 2

            # Draw blocks
            cv.rectangle(line_img, (x1, y1), (x1 + w1, y1 + h1), (255, 255, 0), 2)
            cv.rectangle(line_img, (x2, y2), (x2 + w2, y2 + h2), (255, 255, 0), 2)

            # Horizontal lines on edges
            cv.line(line_img, (x1, top_y), (x1 + w1, top_y), (0, 255, 0), 2)
            cv.line(line_img, (x2, bottom_y), (x2 + w2, bottom_y), (0, 255, 0), 2)

            # Vertical center line
            cv.line(line_img, (center_x, top_y), (center_x, bottom_y), (0, 0, 255), 2)

            # Project pixel positions into normalized space
            #print(pixel_to_world_y(10, cam_mat), "pixel to world")
            #py_top = pixel_to_world_y(top_y, cam_mat)
            #print(py_top, top_y, "This is py_top and top_y")
            #py_bottom = pixel_to_world_y(bottom_y, cam_mat)
            real_distance = abs(bottom_y - top_y) * SCALING_FACTOR_MM # mm

            measurements.append(real_distance)
            if len(measurements) > 10:
                measurements.pop(0)
            avg_distance = sum(measurements) / len(measurements)

            cv.putText(line_img, f'Distance: {avg_distance:.2f} mm', (450, 380),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # print(f"Distance: {avg_distance:.2f} mm")

        else:
            cv.putText(line_img, "Need 2 white blocks", (50, 100),
                       cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

        cv.putText(line_img, 'Actual Distance: 150 mm', (450, 420),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        correction_ratio = {10 : 0.964, 20 : 0.964, 30 : 0.909, 40 : 0.929, 50 : 0.911, 60 : 0.949, 70 : 0.929, 80 : 0.919, 90 : 0.922, 100 : 0.914, 110 : 0.921,
                            120 : 0.930, 130 : 0.921, 140 : 0.919}
        
        final_distance = correction_ratio * real_distance
        # ERROR CORRECTION RATIOS:
        # i need a model for this dataset
        # 10 -> 0.964320154291
        # 20 -> 0.964785335263
        # 30 -> 0.9096422074
        # 40 -> 0.929152148664
        # 50 -> 0.911410864017
        # 60 -> 0.949216896061
        # 70 -> 0.929985385944
        # 80 -> 0.91932888991
        # 90 -> 0.922603792927
        # 100 -> 0.914160343724
        # 110 -> 0.921041614335
        # 120 -> 0.930016275285
        # 130 -> 0.921724333522
        # 140 -> 0.919721455788
        cv.putText(line_img, 'Calculated Distance after error correction : {final_distance}', (450, 440), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.imshow('Detected Blocks', cv.resize(line_img, (1500, 800)))

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    picam2.stop()
    cv.destroyAllWindows()

if __name__ == "__main__":
    cam_mat, dist_coeff = get_calib_data()
    detect()