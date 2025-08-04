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

KNOWN_WIDTH_MM = 24.5  # Known width of the reference block in mm

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
            blocks.sort(key=lambda b: b["center"][0])  # Sort left to right
            block1 = blocks[0]
            block2 = blocks[1]

            # Get box points
            box1 = sorted(block1["box"], key=lambda pt: pt[0])
            box2 = sorted(block2["box"], key=lambda pt: pt[0])

            # Get Y-range for averaging
            y_vals_block1 = [pt[1] for pt in box1]
            y_vals_block2 = [pt[1] for pt in box2]

            top_y = int((min(y_vals_block1) + min(y_vals_block2)) / 2)
            mid_y = int((np.mean(y_vals_block1) + np.mean(y_vals_block2)) / 2)
            bottom_y = int((max(y_vals_block1) + max(y_vals_block2)) / 2)

            y_positions = [top_y, mid_y, bottom_y]
            distances_px = []
            lines_to_draw = []

            for y in y_positions:
                # Start from right edge of block1 and scan to right
                x_start = int(max(box1, key=lambda pt: pt[0])[0])
                x = x_start
                found = False

                while x < frame.shape[1]:
                    inside = cv.pointPolygonTest(block2["contour"], (x, y), False)
                    if inside >= 0:
                        found = True
                        break
                    x += 1

                gap_px = max(0, x - x_start)
                distances_px.append(gap_px)
                if found and gap_px > 0:
                    lines_to_draw.append(((x_start, y), (x, y)))

            # Calculate average gap
            avg_gap_px = sum(distances_px) / len(distances_px)
            avg_gap_mm = avg_gap_px * scale_factor

            # Draw lines only if gap exists
            if avg_gap_px > 0:
                for pt1, pt2 in lines_to_draw:
                    cv.line(frame, pt1, pt2, (0, 0, 255), 2)

            # Display text regardless
            text_x = int(block1["center"][0])
            text_y = int(block1["center"][1])
            cv.putText(frame, f"Gap: {avg_gap_px:.1f}px", (text_x, text_y - 20),
                    cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv.putText(frame, f"Gap: {avg_gap_mm:.2f}mm", (text_x, text_y + 5),
                    cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)



        cv.imshow("Frame", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()

# Run everything together
if __name__ == "__main__":
    process_frame()
