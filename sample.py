import cv2 as cv
from cv2 import aruco
import numpy as np

dummy_img = np.zeros((480, 640), dtype=np.uint8)
marker_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)
params = aruco.DetectorParameters()

# This line likely causes the crash
corners, ids, rejected = aruco.detectMarkers(dummy_img, marker_dict, parameters=params)
