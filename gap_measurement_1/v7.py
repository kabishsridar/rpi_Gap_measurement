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
        block_width_px = None

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

                # Draw block outline
                cv.drawContours(frame, [box], 0, (0, 255, 0), 2)

        # Display scale factor
        if scale_factor:
            cv.putText(frame, f"Scale: {scale_factor:.3f} mm/px", (30, 40),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv.putText(frame, f"Check: {scale_factor * block_width_px:.2f} mm", (30, 70),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Gap Measurement Logic
                # Gap Measurement Logic
                # Check for merged blocks (two blocks attached)
        if len(blocks) == 1 and scale_factor:
            block = blocks[0]
            block_width_px = min(block["width"], block["height"])
            block_width_mm = block_width_px * scale_factor

            if block_width_mm >= KNOWN_WIDTH_MM * 1.5:
                # Detected merged block
                center_x, center_y = map(int, block["center"])
                cv.putText(frame, f"Gap: 0.00mm (Blocks Attached)", (center_x - 60, center_y),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if len(blocks) >= 2 and scale_factor:
            # Sort blocks by area (smallest first)
            blocks.sort(key=lambda b: b["area"])
            block1 = blocks[0]  # Shortest/smallest block
            block2 = blocks[1]

            # Get right edge of block1
            box1 = sorted(block1["box"], key=lambda pt: pt[0])
            right_pts = sorted(box1[-2:], key=lambda pt: pt[1])
            top_y = right_pts[0][1]
            bottom_y = right_pts[1][1]
            mid_y = (top_y + bottom_y) / 2
            x_start = int((right_pts[0][0] + right_pts[1][0]) / 2)

            y_positions = [int(top_y), int(mid_y), int(bottom_y)]
            distances_px = []
            lines_to_draw = []
            valid_lines = 0

            for y in y_positions:
                x = x_start
                found = False
                while x < frame.shape[1]:
                    inside = cv.pointPolygonTest(block2["contour"], (x, y), False)
                    if inside >= 0:
                        found = True
                        break
                    x += 1

                if found and x > x_start:
                    gap_px = x - x_start
                    distances_px.append(gap_px)
                    lines_to_draw.append(((x_start, y), (x, y)))
                    valid_lines += 1

            if valid_lines == 3:
                avg_gap_px = sum(distances_px) / 3
                avg_gap_mm = avg_gap_px * scale_factor

                for pt1, pt2 in lines_to_draw:
                    cv.line(frame, pt1, pt2, (0, 0, 255), 2)

                # Show result
                text_x = int(block1["center"][0])
                text_y = int(block1["center"][1])
                cv.putText(frame, f"Gap: {avg_gap_px:.1f}px", (text_x, text_y - 20),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv.putText(frame, f"Gap: {avg_gap_mm:.2f}mm", (text_x, text_y + 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

            else:
                # Not all lines reached the other block → skip gap drawing
                pass


        cv.imshow("Frame", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()

# Run everything
if __name__ == "__main__":
    process_frame()
