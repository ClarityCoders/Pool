from math import inf
import cv2
import numpy as np
from numpy import ones,vstack
from numpy.linalg import lstsq

def line_equation(coords):
    points = [(coords[0], coords[1]), (coords[2], coords[3])]
    x_coords, y_coords = zip(*points)
    A = vstack([x_coords,ones(len(x_coords))]).T
    m, c = lstsq(A, y_coords)[0]
    #print(f"Line Solution is y = {m}x + {c}")

    counter = 0
    diff = inf
    while abs(diff) > .0001:
        x = 5000 + counter
        y = round(x*m+c)
        far_right = (x, y)
        counter += 1
        diff = round(x*m+c) - (x*m+c)
        #print(abs(diff), x)

    counter = 0
    diff = inf
    while abs(diff) > .0001:
        x = -5000 + counter
        y = round(x*m+c)
        far_left = (x, y)
        counter += 1
        diff = round(x*m+c) - (x*m+c)
        #print(abs(diff))

    return far_left, far_right

def find_lines(img):



    Thresh1 = 1001
    Thresh2 = 1489
    hThresh = 31
    hMinLine = 38
    hMaxGap = 8

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (ret, thresh3) = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY)
    lines_thresh = cv2.Canny(thresh3, threshold1=Thresh1, threshold2=Thresh2)
    #lines_thresh = cv2.dilate(lines_thresh,None)
    #ret, gray = cv2.threshold(img,254,255,cv2.THRESH_BINARY)


    HoughLines = cv2.HoughLinesP(lines_thresh, 1, np.pi/180, threshold = hThresh, minLineLength = hMinLine, maxLineGap = hMaxGap)

    if HoughLines is not None and len(HoughLines) < 50:
        print(f"Line Count: {len(HoughLines)}")
        cv2.putText(img, f'Line Count: {len(HoughLines)}', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255,255,255), 2, cv2.LINE_AA)
        for line in HoughLines:
            coords = line[0]
            far_left, far_right = line_equation(coords)
            #cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), [0,0,255], 3)
            cv2.line(img, far_left, far_right, [0,10,175], 3)

    return img