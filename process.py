import cv2
import numpy as np
from utils import line_equation



kernel = np.ones((3,3), np.uint8) 

img = cv2.imread('test.png', cv2.IMREAD_COLOR)


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
(ret, thresh3) = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY)
lines_gray = cv2.Canny(gray, threshold1=50, threshold2=150)
lines_thresh = cv2.Canny(thresh3, threshold1=1001, threshold2=1489)
#lines_thresh = cv2.dilate(lines_thresh,None)
#ret, gray = cv2.threshold(img,254,255,cv2.THRESH_BINARY)

hThresh = 31
hMinLine = 35
hMaxGap = 3

HoughLines = cv2.HoughLinesP(lines_thresh, 1, np.pi/180, threshold = hThresh, minLineLength = hMinLine, maxLineGap = hMaxGap)

if HoughLines is not None:
    print(f"Line Count: {len(HoughLines)}")
    cv2.putText(img, f'Line Count: {len(HoughLines)}', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255,255,255), 2, cv2.LINE_AA)
    for line in HoughLines:
        coords = line[0]
        far_left, far_right = line_equation(coords)
        #cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), [0,0,255], 3)
        cv2.line(img, far_left, far_right, [0,10,175], 3)

cv2.imshow("ORG", img)
cv2.imshow("lines Gray", lines_gray)
cv2.imshow("lines Thresh", lines_thresh)
cv2.waitKey(0)