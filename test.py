import cv2
import numpy as np

cap = cv2.VideoCapture(0)
lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 30, 255])

if not cap.isOpened():
    print("cannot open the camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("the frame is not read")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # converts the frame to HSV , so that we can select only the white color
    mask = cv2.inRange(hsv, lower_white, upper_white) # creates a mask
    res = cv2.bitwise_and(frame, frame, mask=mask) # merges the mask with the original frame

    edges = cv2.Canny(res, 100, 800) # detecting the edges using Canny edge detection of the white image
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, maxLineGap=10) # detecting the lines
    line_img = res.copy() # creating a copy frame to draw the lines

    if lines is not None: # checks if lines are detected
        print(len(lines)) # prints the number of lines detected
        for i, line in enumerate(lines): # loops through the detected lines with index using enumerate
            x1, y1, x2, y2 = line[0]
            cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 2) # draws the lines on the copy of the frame line_img
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2
            cv2.putText(line_img, f"#{i+1}", (mid_x, mid_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2) # displays the number of the particular line on the frame

    cv2.imshow('Original Frame', frame) # displays all the frames
    cv2.imshow('White Filtered', res)
    cv2.imshow('Detected Lines', line_img)

    if cv2.waitKey(1) & 0xFF == ord('q'): # quit if q is pressed
        break

if lines is None or len(lines) < 2: # if less than 2 lines are detected, we cannot calculate the distance
        print("Not enough lines detected to calculate distance.")
else:
    x1_1, y1_1, x2_1, y2_1 = lines[0][0]
    x1_2, y1_2, x2_2, y2_2 = lines[1][0]

    line1_vector = np.array([x2_1 - x1_1, y2_1 - y1_1])
    point_on_line1 = np.array([x1_1, y1_1])
    point_on_line2 = np.array([x1_2, y1_2])

    numerator = np.abs(np.cross(line1_vector, point_on_line2 - point_on_line1))
    denominator = np.linalg.norm(line1_vector)
    dis = numerator / denominator
    distance = dis/2.1

    print(f"Distance between line 1 and line 2: {distance:.2f} mm")

cap.release()
cv2.destroyAllWindows()