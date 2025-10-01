from edge_detection import EdgeDetector
from app import EdgeDetectionApp
import tkinter as tk
import cv2 as cv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

root = tk.Tk()
root.withdraw()
app = EdgeDetector(root)
def get_image():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open Camera")
        exit()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        org = cv.flip(gray, 1)
        cv.imshow('original image', org)
        return org
