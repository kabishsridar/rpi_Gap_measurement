import cv2 as cv
import numpy as np
from picamera2 import Picamera2
import time

# Start Pi Camera at higher resolution
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (1920, 1080)})
picam2.configure(config)

# ✅ Enable White Balance & Basic Corrections
picam2.set_controls({
    "AwbEnable": True,
    "Brightness": 0.0,
    "Contrast": 1.0,
    "Saturation": 1.0,
    "Sharpness": 1.0,
    "ExposureValue": 0.0
})

picam2.start()
time.sleep(1)

# White color HSV range
lower_white = np.array([0, 0, 197])
upper_white = np.array([95, 55, 255])

KNOWN_WIDTH_MM = 25.3  # mm (your reference block width)

def process_frame():
    while True:
        frame = picam2.capture_array()

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, lower_white, upper_white)

        # Smaller kernel for finer detail (1mm gap preservation)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        blocks = []
        scale_factor = None

        for cnt in contours:
            area = cv.contourArea(cnt)
            if area > 500:  # Lowered threshold for smaller blocks
                rect = cv.minAreaRect(cnt)
                box = cv.boxPoints(rect)
                box = np.intp(box)
                center, (w, h), angle = rect
                block_width_px = min(w, h)
                block_height_px = max(w, h)

                if block_width_px != 0:
                    scale_factor = KNOWN_WIDTH_MM / block_width_px

                blocks.append({
                    "rect": rect,
                    "box": box,
                    "width": w,
                    "height": h,
                    "center": center,
                    "area": area,
                    "contour": cnt
                })

                cv.drawContours(frame, [box], 0, (0, 255, 0), 2)

        if scale_factor:
            cv.putText(frame, f"Scale: {scale_factor:.4f} mm/px", (30, 40),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Gap detection if at least 2 blocks found
        if len(blocks) >= 2 and scale_factor:
            blocks.sort(key=lambda b: b["area"])
            block1 = blocks[0]
            block2 = blocks[1]

            box1 = sorted(block1["box"], key=lambda pt: pt[0])
            right_edge = [box1[2], box1[3]] if box1[2][1] < box1[3][1] else [box1[3], box1[2]]

            # Slight offset from edges: ~15% from top and bottom
            y1 = int(right_edge[0][1])
            y2 = int(right_edge[1][1])
            line_y_top = int(y1 + 0.15 * (y2 - y1))
            line_y_mid = int((y1 + y2) / 2)
            line_y_bot = int(y2 - 0.15 * (y2 - y1))
            x_start = int((right_edge[0][0] + right_edge[1][0]) / 2)

            distances_px = []
            line_positions = [line_y_top, line_y_mid, line_y_bot]

            for line_y in line_positions:
                end_x = x_start
                found = False
                while end_x < frame.shape[1] - 1:
                    test_point = (end_x, line_y)
                    inside = cv.pointPolygonTest(block2["contour"], test_point, False)
                    if inside >= 0:
                        found = True
                        break
                    end_x += 1

                if found:
                    dist_px = end_x - x_start
                    distances_px.append(dist_px)
                    cv.line(frame, (x_start, line_y), (end_x, line_y), (0, 0, 255), 2)

            if distances_px:
                avg_px = sum(distances_px) / len(distances_px)
                avg_mm = avg_px * scale_factor
                cv.putText(frame, f"Gap Avg: {avg_mm:.2f} mm", (x_start, line_y_mid + 30),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv.imshow("Frame", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()

if __name__ == "__main__":
    process_frame()
