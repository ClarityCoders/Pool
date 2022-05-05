import cv2
import numpy as np
from utils import line_equation

def nothing(x):
    pass

cv2.namedWindow("Trackbars")
cv2.createTrackbar("Thresh1", "Trackbars", 1001, 5000, nothing)
cv2.createTrackbar("Thresh2", "Trackbars", 1489, 5000, nothing)

cv2.createTrackbar("hThresh", "Trackbars", 31, 2000, nothing)
cv2.createTrackbar("hMinLine", "Trackbars", 39, 100, nothing)
cv2.createTrackbar("hMaxGap", "Trackbars", 13, 100, nothing)

while True:

    Thresh1 = cv2.getTrackbarPos("Thresh1", "Trackbars")
    Thresh2 = cv2.getTrackbarPos("Thresh2", "Trackbars")
    hThresh = cv2.getTrackbarPos("hThresh", "Trackbars")
    hMinLine = cv2.getTrackbarPos("hMinLine", "Trackbars")
    hMaxGap = cv2.getTrackbarPos("hMaxGap", "Trackbars")


    images = [
        cv2.imread('test.png', cv2.IMREAD_COLOR),
        cv2.imread('test2.png', cv2.IMREAD_COLOR),
    ]


    for i, img in enumerate(images):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        (ret, thresh3) = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY)

        
        kernel = np.ones((5, 5), 'uint8')

        #thresh3 = cv2.dilate(thresh3, kernel, iterations=1)

        lines_thresh = cv2.Canny(thresh3, threshold1=Thresh1, threshold2=Thresh2)
        lines_thresh = cv2.dilate(lines_thresh, kernel, iterations=1)
        #lines_thresh = cv2.dilate(lines_thresh,None)
        #ret, gray = cv2.threshold(img,254,255,cv2.THRESH_BINARY)


        HoughLines = cv2.HoughLinesP(lines_thresh, 1, np.pi/180, threshold = hThresh, minLineLength = hMinLine, maxLineGap = hMaxGap)

        if HoughLines is not None:
            print(f"Line Count: {len(HoughLines)}")
            cv2.putText(img, f'Line Count: {len(HoughLines)}', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255,255,255), 2, cv2.LINE_AA)
            for line in HoughLines:
                coords = line[0]
                #far_left, far_right = line_equation(coords)
                cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), [0,0,255], 3)
                #cv2.line(img, far_left, far_right, [0,10,175], 3)

        cv2.imshow(f"ORG{i}", img)
        cv2.imshow(f"lines Thresh{i}", lines_thresh)

        cv2.waitKey(1)