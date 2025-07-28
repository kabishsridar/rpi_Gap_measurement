from picamera2 import Picamera2
import cv2
import time

picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (1280, 720)},
    lores={"size": (640, 480)},
    display="lores"
)
picam2.configure(config)
picam2.start()
time.sleep(0)

while True:
    frame = picam2.capture_array()
    org = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    edges = cv2.Canny(org, 100, 450)
    cv2.imshow('canny', edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()