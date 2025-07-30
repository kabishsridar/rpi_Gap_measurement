from picamera2 import Picamera2
import cv2 as cv

picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

while True:
    try:
        frame = picam2.capture_array()
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        cv.imshow("PiCam Test", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        print(f"Error: {e}")
        break

cv.destroyAllWindows()
