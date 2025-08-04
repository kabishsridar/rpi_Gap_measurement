import cv2 as cv
import numpy as np
from picamera2 import Picamera2
import time

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (1280, 720)})
picam2.configure(config)
picam2.start()
time.sleep(1)

hsv_samples = []
lower_bound = np.array([180, 255, 255])
upper_bound = np.array([0, 0, 0])

def mouse_callback(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        frame = param["frame"]
        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        pixel_hsv = hsv_frame[y, x]
        hsv_samples.append(pixel_hsv)

        global lower_bound, upper_bound
        lower_bound = np.minimum(lower_bound, pixel_hsv)
        upper_bound = np.maximum(upper_bound, pixel_hsv)

        print(f"Clicked HSV: {pixel_hsv}")
        print(f"Updated Lower HSV: {lower_bound}")
        print(f"Updated Upper HSV: {upper_bound}")

def main():
    cv.namedWindow("Click to Sample HSV")

    while True:
        frame = picam2.capture_array()
        clone = frame.copy()
        cv.setMouseCallback("Click to Sample HSV", mouse_callback, {"frame": clone})

        if len(hsv_samples) > 0:
            cv.putText(clone, f"Lower: {lower_bound}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv.putText(clone, f"Upper: {upper_bound}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv.imshow("Click to Sample HSV", clone)

        key = cv.waitKey(1)
        if key == ord('q'):
            break

    cv.destroyAllWindows()

    avg_hsv = np.mean(hsv_samples, axis=0).astype(int)
    print("\n--- Final HSV Range for Your Block ---")
    print(f"lower_white = np.array([{lower_bound[0]}, {lower_bound[1]}, {lower_bound[2]}])")
    print(f"upper_white = np.array([{upper_bound[0]}, {upper_bound[1]}, {upper_bound[2]}])")
    print(f"average_hsv = np.array([{avg_hsv[0]}, {avg_hsv[1]}, {avg_hsv[2]}])")

if __name__ == "__main__":
    main()
