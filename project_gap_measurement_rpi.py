import cv2 as cv # importing the modules
import tkinter as tk
import sys
from pathlib import Path
# from app import EdgeDetectionApp # importing the files
# from processing.edge_detection import EdgeDetector
import numpy as np
# from collections import defaultdict
from group_lines import group_lines_by_y
from picamera2 import Picamera2
import time

def get_calib_data():
    calib_data_path = "/home/kabish/python_projects/rpi_Gap_measurement/calibrated_data/MultiMatrix.npz"
    calib_data = np.load(calib_data_path)
    # print(calib_data.files)

    cam_mat = calib_data["camMatrix"] # assigning the datas caliberated through the MultiMatrix file
    dist_coef = calib_data["distCoef"]
    r_vectors = calib_data["rVector"]
    t_vectors = calib_data["tVector"]
    # print(cam_mat, dist_coef, r_vectors, t_vectors)
    
    return cam_mat, dist_coef

def store_to_database(time, distance):# function to store the time and distance to db
    import mysql.connector as m
    from mysql.connector import Error
    try:
        con = m.connect(
            host='localhost', database='PROJECT_GAP_MEASUREMENT', user='root', password='2007kabish' # connecting to database
        )
        if con.is_connected():
            cur = con.cursor()
            cur.execute('CREATE TABLE IF NOT EXISTS GAP_MEASUREMENT(TIME FLOAT, DISTANCE FLOAT)') # if it is connected to database, create a table if that doesn't exists
            cur.execute(f'INSERT INTO GAP_MEASUREMENT VALUES({1}, {2})'.format(time, distance)) # and insert the values.
            
        else:
            print("Error while connecting to Mysql: Connection unsuccessful")
            return None
    except Error as e:
        print("Error while connecting to Mysql:", e)
        return None

def splash(boolean_val):
    if boolean_val:
        flash = cv.imread('OMAC_SOLUTION.png')
        resized = cv.resize(flash, (1500, 800))
        cv.imshow('waiting', resized)
    else:
        try:
            cv.destroyWindow('waiting')
        except cv.error:
            pass  # if the window has not yet started, now it will not result an error. With help of this, it will pass the opencv error

def detect():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (1280, 720)},
        lores={"size": (640, 480)},
        display="lores"
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1)

    cv.namedWindow("Omac Distance Measurement", cv.WINDOW_NORMAL)

    
    lower_white = np.array([0, 0, 200]) # the range of white color
    upper_white = np.array([180, 30, 255])
    measurements = [] # empty list to calculate average

    while True:
        frame = picam2.capture_array() # reading the video
        # frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        if frame is None: # if cannot read the image, return the error message and break the loop
            print("cannot read image")
            break

        undistorted_frame = cv.undistort(frame, cam_mat, dist_coeff)

        # image = cv.flip(undistorted_frame, 1) # the camera normally will mirror the images, now the flip will give the original image
        # app.original_image = undistorted_frame # set original image as the image we get

        # detector.apply_edge_detection(fast_mode=True) # apply the edge detection from the imported file
        # output = app.processed_image # this will process the image and will return it
        # cv.imshow('the output', output)
        
        hsv = cv.cvtColor(undistorted_frame, cv.COLOR_BGR2HSV) # converts the frame to HSV , so that we can select only the white color
        mask = cv.inRange(hsv, lower_white, upper_white) # creates a mask
        res = cv.bitwise_and(undistorted_frame, undistorted_frame, mask=mask) # merges the mask with the original frame

        edges = cv.Canny(res, 100, 800) # detecting the edges using Canny edge detection of the white image
        # Optional: Close small gaps between edge segments
        kernel = np.ones((3, 3), np.uint8)
        edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)

        lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=100, maxLineGap=10) # detecting the lines
        # print(lines)
        line_img = res.copy() # creating a copy frame to draw the lines
        # line_img = cv.flip(flipped_line_img, 1) # if we flip the image, the lines are not detected properly

        if lines is not None: # checks if lines are detected
            print(f'Number of lines: {len(lines)}') # prints the number of lines detected
            for i, line in enumerate(lines): # loops through the detected lines with index using enumerate
                x1, y1, x2, y2 = line[0]
                # cv.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 2) # draws the lines on the copy of the frame line_img
                mid_x = (x1 + x2) // 2
                mid_y = (y1 + y2) // 2
                # print(f"the coordinate for line {i + 1}= {mid_x}, {mid_y}")
                # cv.putText(line_img, f"#{i+1}", (mid_x, mid_y),
                            # cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2) # displays the number of the particular line on the frame

        if lines is None or len(lines) < 2: # if less than 2 lines are detected, we cannot calculate the distance
            print("Not enough lines detected to calculate distance.")
            # cv.putText(line_img,'OMAC GAP Measurement System',(50,100), cv.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv.LINE_AA)
            no_line = True
            splash(no_line)
        else:
            no_line = False
            splash(no_line)

            # Use grouped line detection
            line_groups = group_lines_by_y(lines, threshold=20)
            group_ys = sorted(line_groups.keys())
            
            if len(group_ys) <= 2:
                splash(True)


            else:
                # these are done by chatgpt
                two_lines = []
                for y in group_ys[1:3]:
                    lines_in_group = line_groups[y]
                    all_y = [l[1] for l in lines_in_group] + [l[3] for l in lines_in_group]
                    avg_y = int(np.mean(all_y))
                    two_lines.append(avg_y)
                    cv.line(line_img, (0, avg_y), (line_img.shape[1], avg_y), (0, 255, 0), 2)

                mid_y = (two_lines[0] + two_lines[1]) // 2
                pixel_gap = abs(two_lines[0] - two_lines[1])
                distance = pixel_gap * (135 / 183)
                measurements.append(distance)

                cv.putText(line_img, f'Distance: {distance:.2f} mm', (250, 380),
                        cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                print(f"Distance between grouped lines: {distance:.2f} mm")
                
                
        # cv.imshow("Live Feed", frame) # displays both the original frame and the frame in which edges are detected
        # cv.imshow("Edge Detection", output)
        cv.putText(line_img, 'Actual Distance: 50 mm', (250, 420),
                        cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.imshow('Detected Lines', cv.resize(line_img, (1500, 800)))

        if cv.waitKey(1) & 0xFF == ord('q'): # it breaks if we press q
            break
        # input('')
    print(measurements)
    print(f"number of measurements taken : {len(measurements)}")
    print(f"\nDistance between two lines : { sum(measurements)/ len(measurements)}")
        
    picam2.stop()
    cv.destroyAllWindows() # cleaning up the window


if __name__ == "__main__":
    # sys.path.insert(0, str(Path(__file__).resolve().parent))

    # root = tk.Tk() # creating a window
    # root.withdraw() # closing the window suddenly

    # app = EdgeDetectionApp(root) # calling the classes and storing it to the variables
    # detector = EdgeDetector(app)

    cam_mat, dist_coeff = get_calib_data()
    # print(dist_coeff)

    detect()