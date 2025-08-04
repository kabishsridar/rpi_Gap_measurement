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

def get_scale():
    while True:
        frame = picam2.capture_array()

        # Convert to HSV and mask white areas
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, lower_white, upper_white)

        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        scale = None

        for cnt in contours:
            area = cv.contourArea(cnt)
            if area > 1000:
                rect = cv.minAreaRect(cnt)
                box = cv.boxPoints(rect)
                box = np.int0(box)

                (center_x, center_y), (width, height), angle = rect
                width_px = min(width, height)  # Assume the known width is the smaller side

                if width_px != 0:
                    scale = 25 / width_px  # mm per pixel

                    # Draw rectangle and display dimensions
                    cv.drawContours(frame, [box], 0, (0, 255, 0), 2)
                    cv.putText(frame, f"W: {int(width)}px", (int(center_x - 60), int(center_y - 10)),
                               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    cv.putText(frame, f"H: {int(height)}px", (int(center_x - 60), int(center_y + 20)),
                               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    cv.putText(frame, f"Scale factor: {scale:.6f} mm/px", (30, 50),
                               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv.putText(frame, f"Cross check (scale * width_px): {(scale * width_px):.6f} ", (30, 80),
                               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    break  # Only one block is enough to compute the scale

        # Show the result
        cv.imshow("Frame", frame)

        key = cv.waitKey(1)
        if key & 0xFF == ord('q') and scale is not None:
            break

    return scale
""" 
def measure_gap(scale_factor):
    while True:
        frame = picam2.capture_array()
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, lower_white, upper_white)

        # Morphological clean-up
        kernel = np.ones((5, 5), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        blocks = []

        for cnt in contours:
            area = cv.contourArea(cnt)
            if area > 1000:
                rect = cv.minAreaRect(cnt)
                box = cv.boxPoints(rect)
                box = np.int0(box)
                center, (w, h), angle = rect
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

        if len(blocks) >= 2:
            # Sort by area, get smallest and second block
            blocks.sort(key=lambda b: b["area"])
            block1 = blocks[0]
            block2 = blocks[1]

            # Get midpoint of right edge of smallest block
            box1 = block1["box"]
            box1 = sorted(box1, key=lambda pt: pt[0])  # Sort by x
            left_edge = [box1[0], box1[1]] if box1[0][1] < box1[1][1] else [box1[1], box1[0]]
            right_edge = [box1[2], box1[3]] if box1[2][1] < box1[3][1] else [box1[3], box1[2]]

            mid_x = int((right_edge[0][0] + right_edge[1][0]) / 2)
            mid_y = int((right_edge[0][1] + right_edge[1][1]) / 2)

            # Extend line to right and stop if it hits block2's contour
            end_x = mid_x
            found = False
            while end_x < frame.shape[1]:
                test_point = (end_x, mid_y)
                inside = cv.pointPolygonTest(block2["contour"], test_point, False)
                if inside >= 0:
                    found = True
                    break
                end_x += 1

            if found:
                # Draw the line
                cv.line(frame, (mid_x, mid_y), (end_x, mid_y), (0, 0, 255), 2)
                distance_px = end_x - mid_x
                distance_mm = distance_px * 0.247524752475

                # Display distance in px and mm
                cv.putText(frame, f"Gap: {distance_px}px", (mid_x, mid_y - 15),
                        cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv.putText(frame, f"Gap: {distance_mm:.2f}mm", (mid_x, mid_y + 10),
                        cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Show output
        cv.imshow("Frame", frame)
        # cv.imshow("White Mask", mask)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()
 """
if __name__ == "__main__":
    scale_factor = get_scale()
    # measure_gap(scale_factor)
