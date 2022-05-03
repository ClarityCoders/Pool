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


    l_h = 0
    l_s = 0
    l_v = 235
    u_h = 196
    u_s = 4
    u_v = 255
    Thresh1 = 1100
    Thresh2 = 1489
    hThresh = 31
    hMinLine = 38
    hMaxGap = 11

    lower_mask = np.array([l_h, l_s, l_v])
    upper_mask = np.array([u_h, u_s, u_v])

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask2 = cv2.inRange(hsv, lower_mask, upper_mask)
    output2 = cv2.bitwise_and(img,img, mask= mask2)


    #result = img

    #use mask
    #result = cv2.bitwise_and(img, img, mask=mask)


    lines = cv2.Canny(output2, threshold1=Thresh1, threshold2=Thresh2)


    #HoughLines = cv2.HoughLinesP(img_dilation, 1, np.pi/180, hThresh, hMinLine, hMaxGap
    HoughLines = cv2.HoughLinesP(lines, 1, np.pi/180, threshold = hThresh, minLineLength = hMinLine, maxLineGap = hMaxGap)
    if HoughLines is not None:
        for line in HoughLines:
            coords = line[0]
            far_right, far_left = line_equation(coords)
            #cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), [0,0,255], 3)

            cv2.line(img, far_left, far_right, [0,0,255], 3)

    return img