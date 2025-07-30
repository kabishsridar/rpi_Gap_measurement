import cv2 as cv
from picamera2 import Picamera2
import time

picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (1280, 720)},
    lores={"size": (640, 480)},
    display="lores"
)
picam2.configure(config)
picam2.start()
time.sleep(1)

while True:
    frame = picam2.capture_array("main")
    cv.imshow("frame", frame)

    key = cv.waitKey(1)
    if key == ord("q"):
        break

picam2.stop()
cv.destroyAllWindows()
# the camera can be in a distance of 55 to 60 cm to cover both of the blocks.