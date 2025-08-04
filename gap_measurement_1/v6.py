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

KNOWN_WIDTH_MM = 25.3  # Known width of the reference block in mm

def process_frame():
    while True:
        frame = picam2.capture_array()

        # Convert to HSV and mask white areas
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, lower_white, upper_white)

        # Morphological clean-up
        kernel = np.ones((5, 5), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        blocks = []
        scale_factor = None

        for cnt in contours:
            area = cv.contourArea(cnt)
            if area > 1000:
                rect = cv.minAreaRect(cnt)
                box = cv.boxPoints(rect)
                box = np.int0(box)
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

                # Draw block and scale
                cv.drawContours(frame, [box], 0, (0, 255, 0), 2)
                """ cv.putText(frame, f"W: {int(w)}px", (int(center[0] - 60), int(center[1] - 10)),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv.putText(frame, f"H: {int(h)}px", (int(center[0] - 60), int(center[1] + 20)),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2) """

        # Display scale factor on screen
        if scale_factor:
            cv.putText(frame, f"Scale: {scale_factor:.3f} mm/px", (30, 40),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv.putText(frame, f"Check: {scale_factor * block_width_px:.2f} mm", (30, 70),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # If at least 2 blocks are found, measure gap
        if len(blocks) >= 2 and scale_factor:
            blocks.sort(key=lambda b: b["area"])
            block1 = blocks[0]
            block2 = blocks[1]

            box1 = sorted(block1["box"], key=lambda pt: pt[0])  # sort by x
            right_edge = [box1[2], box1[3]] if box1[2][1] < box1[3][1] else [box1[3], box1[2]]
            mid_x = int((right_edge[0][0] + right_edge[1][0]) / 2)
            mid_y = int((right_edge[0][1] + right_edge[1][1]) / 2)

            # Extend line to right until hitting second block
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
                distance_px = end_x - mid_x
                distance_mm = distance_px * scale_factor  # ✅ Use dynamically calculated scale

                # Draw and display distance
                cv.line(frame, (mid_x, mid_y), (end_x, mid_y), (0, 0, 255), 2)
                cv.putText(frame, f"Gap: {distance_px}px", (mid_x, mid_y - 15),
                        cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv.putText(frame, f"Gap: {distance_mm:.2f}mm", (mid_x, mid_y + 10),
                        cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # === ADDITIONAL TOP AND BOTTOM LINE GAP MEASUREMENTS ===
                top_y = min(right_edge[0][1], right_edge[1][1])
                bottom_y = max(right_edge[0][1], right_edge[1][1])
                x_start = mid_x

                distances_px = [distance_px]  # include mid-point measurement

                for y in [top_y, bottom_y]:
                    end_x_extra = x_start
                    found_extra = False
                    while end_x_extra < frame.shape[1]:
                        test_point = (end_x_extra, int(y))
                        inside = cv.pointPolygonTest(block2["contour"], test_point, False)
                        if inside >= 0:
                            found_extra = True
                            break
                        end_x_extra += 1

                    if found_extra:
                        distances_px.append(end_x_extra - x_start)
                        cv.line(frame, (x_start, int(y)), (end_x_extra, int(y)), (0, 0, 255), 2)

                avg_px = sum(distances_px) / len(distances_px)
                avg_mm = avg_px * scale_factor
                cv.putText(frame, f"Avg: {avg_mm:.2f}mm", (mid_x, mid_y + 35),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv.imshow("Frame", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()

# Run everything together
if __name__ == "__main__":
    process_frame()
