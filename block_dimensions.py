from picamera2 import Picamera2
import cv2 as cv
import time
import numpy as np

# Define known block sizes (short x long)
BLOCK_SIZES_MM = [
    (25.0, 75.0),
    (25.0, 50.0)
]

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

def match_block_size(short_px, long_px):
    pixel_per_mm = short_px / 25.0
    approx_long_mm = long_px / pixel_per_mm
    best_match = None
    min_diff = float('inf')
    for short_mm, long_mm in BLOCK_SIZES_MM:
        diff = abs(approx_long_mm - long_mm)
        if diff < min_diff:
            min_diff = diff
            best_match = (short_mm, long_mm)
    return best_match

while True:
    calib_data_path = "/home/kabish/python_projects/rpi_Gap_measurement/calibrations/calibrated_data/MultiMatrix_rpi.npz"
    calib_data = np.load(calib_data_path)
    # print(calib_data.files)

    cam_mat = calib_data["camMatrix"] # assigning the datas caliberated through the MultiMatrix file
    dist_coef = calib_data["distCoef"]

    frame = picam2.capture_array()
    undistorted = cv.undistort(frame, cam_mat, dist_coef)
    if frame is None:
        print("Cannot read image")
        break

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower_white, upper_white)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    blocks = []

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 1000:
            x, y, w, h = cv.boundingRect(cnt)
            short_px = min(w, h)
            long_px = max(w, h)
            short_mm, long_mm = match_block_size(short_px, long_px)
            cx = x + w // 2
            cy = y + h // 2

            blocks.append({
                "x": x, "y": y, "w": w, "h": h,
                "cx": cx, "cy": cy,
                "short_mm": short_mm, "long_mm": long_mm
            })

            # Draw dimensions
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if w < h:  # Vertical block
                cv.line(frame, (x, y - 20), (x + w, y - 20), (255, 0, 0), 2)
                cv.putText(frame, f"{short_mm:.0f} mm", (x + w // 2 - 20, y - 25),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv.line(frame, (x - 20, y), (x - 20, y + h), (0, 0, 255), 2)
                cv.putText(frame, f"{long_mm:.0f} mm", (x - 100, y + h // 2),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:  # Horizontal block
                cv.line(frame, (x - 20, y), (x - 20, y + h), (0, 0, 255), 2)
                cv.putText(frame, f"{short_mm:.0f} mm", (x - 100, y + h // 2),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv.line(frame, (x, y + h + 20), (x + w, y + h + 20), (255, 0, 0), 2)
                cv.putText(frame, f"{long_mm:.0f} mm", (x + w // 2 - 20, y + h + 40),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Compute edge-to-edge distance between 2 blocks
    if len(blocks) == 2:
        b1, b2 = blocks[0], blocks[1]

        # Use average pixel-per-mm ratio from short side
        avg_pixel_per_mm = ((min(b1["w"], b1["h"]) + min(b2["w"], b2["h"])) / 2) / 25.0

        # Determine orientation and edge-to-edge distance
        if abs(b1["cx"] - b2["cx"]) > abs(b1["cy"] - b2["cy"]):
            # Side-by-side (horizontal gap)
            right1 = b1["x"] + b1["w"]
            left2 = b2["x"]
            if b2["x"] < b1["x"]:
                right1 = b2["x"] + b2["w"]
                left2 = b1["x"]
                p1 = (right1, b2["cy"])
                p2 = (left2, b1["cy"])
            else:
                p1 = (right1, b1["cy"])
                p2 = (left2, b2["cy"])
            edge_px = max(0, left2 - right1)
        else:
            # One above another (vertical gap)
            bottom1 = b1["y"] + b1["h"]
            top2 = b2["y"]
            if b2["y"] < b1["y"]:
                bottom1 = b2["y"] + b2["h"]
                top2 = b1["y"]
                p1 = (b2["cx"], bottom1)
                p2 = (b1["cx"], top2)
            else:
                p1 = (b1["cx"], bottom1)
                p2 = (b2["cx"], top2)
            edge_px = max(0, top2 - bottom1)

        # Convert to mm
        dist_mm = edge_px / avg_pixel_per_mm

        # Draw line and distance
        cv.line(frame, p1, p2, (0, 255, 255), 2)
        mid_x = (p1[0] + p2[0]) // 2
        mid_y = (p1[1] + p2[1]) // 2
        cv.putText(frame, f"{dist_mm:.1f} mm", (mid_x, mid_y - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv.imshow('Detected Blocks', frame)
    cv.imshow('White Mask', mask)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv.destroyAllWindows()
