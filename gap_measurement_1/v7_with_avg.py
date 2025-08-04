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

KNOWN_WIDTH_MM = 25.3  # Reference block width in mm

def process_frame():
    while True:
        frame = picam2.capture_array()
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, lower_white, upper_white)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        blocks = []
        scale_factor = None

        for cnt in contours:
            area = cv.contourArea(cnt)
            if area > 1000:
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
            cv.putText(frame, f"Scale: {scale_factor:.3f} mm/px", (30, 40),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv.putText(frame, f"Check: {scale_factor * block_width_px:.2f} mm", (30, 70),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if len(blocks) >= 2 and scale_factor:
            blocks.sort(key=lambda b: b["area"])
            block1 = blocks[0]
            block2 = blocks[1]

            # Check if blocks are too close (or overlapping), treat gap as zero
            center1 = np.array(block1["center"])
            center2 = np.array(block2["center"])
            center_distance = np.linalg.norm(center1 - center2)

            if center_distance < 10:  # You can fine-tune this threshold
                cv.putText(frame, f"Gap: 0 px", (30, 100),
                        cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv.putText(frame, f"Gap: 0.00 mm", (30, 130),
                        cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv.imshow("Frame", frame)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    return
                continue

            box = block1["box"]
            contour2 = block2["contour"]

            def line_gap(p1, direction, max_len=frame.shape[1]):
                x, y = p1
                dx, dy = direction
                while 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                    if cv.pointPolygonTest(contour2, (x, y), False) >= 0:
                        return (x, y)
                    x += dx
                    y += dy
                return None

            # Define edges and directions
            top = sorted(box, key=lambda p: p[1])[:2]
            bottom = sorted(box, key=lambda p: p[1])[2:]
            left = sorted(box, key=lambda p: p[0])[:2]
            right = sorted(box, key=lambda p: p[0])[2:]

            edges = [
                {"edge": left, "step": (-1, 0)},
                {"edge": right, "step": (1, 0)},
                {"edge": top, "step": (0, -1)},
                {"edge": bottom, "step": (0, 1)},
            ]

            for edge in edges:
                pt1, pt2 = edge["edge"]
                direction = edge["step"]

                # Get 3 points: top, center, bottom along the edge
                p_top = (int(0.9 * pt1[0] + 0.1 * pt2[0]), int(0.9 * pt1[1] + 0.1 * pt2[1]))
                p_mid = (int((pt1[0] + pt2[0]) / 2), int((pt1[1] + pt2[1]) / 2))
                p_bot = (int(0.1 * pt1[0] + 0.9 * pt2[0]), int(0.1 * pt1[1] + 0.9 * pt2[1]))

                points = [p_top, p_mid, p_bot]

                distances_px = []
                for p in points:
                    end = line_gap(p, direction)
                    if end:
                        cv.line(frame, p, end, (0, 0, 255), 2)
                        dist = int(np.linalg.norm(np.array(end) - np.array(p)))
                        distances_px.append(dist)

                if distances_px:
                    avg_px = sum(distances_px) / len(distances_px)
                    avg_mm = avg_px * scale_factor
                    disp_x, disp_y = p_mid
                    cv.putText(frame, f"Gap: {int(avg_px)}px", (disp_x, disp_y - 20),
                               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv.putText(frame, f"Gap: {avg_mm:.2f}mm", (disp_x, disp_y + 10),
                               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

        cv.imshow("Frame", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()

# Run
if __name__ == "__main__":
    process_frame()
