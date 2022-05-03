import cv2
import numpy as np
from utils import line_equation

def nothing(x):
    pass

cv2.namedWindow("Trackbars")
cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 235, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 196, 255, nothing)
cv2.createTrackbar("U - S", "Trackbars", 4, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("Thresh1", "Trackbars", 1001, 5000, nothing)
cv2.createTrackbar("Thresh2", "Trackbars", 1489, 5000, nothing)

cv2.createTrackbar("hThresh", "Trackbars", 31, 400, nothing)
cv2.createTrackbar("hMinLine", "Trackbars", 38, 100, nothing)
cv2.createTrackbar("hMaxGap", "Trackbars", 11, 100, nothing)

kernel = np.ones((3,3), np.uint8) 
while True:

    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    Thresh1 = cv2.getTrackbarPos("Thresh1", "Trackbars")
    Thresh2 = cv2.getTrackbarPos("Thresh2", "Trackbars")
    hThresh = cv2.getTrackbarPos("hThresh", "Trackbars")
    hMinLine = cv2.getTrackbarPos("hMinLine", "Trackbars")
    hMaxGap = cv2.getTrackbarPos("hMaxGap", "Trackbars")

    lower_blue = np.array([l_h, l_s, l_v])
    upper_blue = np.array([u_h, u_s, u_v])
    #lower_blue = np.array([0,0,168])
    #upper_blue = np.array([172,111,255])

    images = [
        cv2.imread('test.png', cv2.IMREAD_COLOR),
        cv2.imread('test2.png', cv2.IMREAD_COLOR),
    ]


    for i, img in enumerate(images):

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask2 = cv2.inRange(hsv, lower_blue, upper_blue)
        output2 = cv2.bitwise_and(img,img, mask= mask2)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #ret, gray = cv2.threshold(img,254,255,cv2.THRESH_BINARY)
        
        print(output2.shape)
        test = cv2.cvtColor(output2, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=18, minRadius=1, maxRadius=26)

        # ensure at least some circles were found
        if circles is not None:
            print("FUCK YEAH")
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")
            # loop over the (x, y) coordinates and radius of the circles
            for (x, y, r) in circles:
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                cv2.circle(output2, (x, y), r, (0, 255, 0), -1)
                #cv2.rectangle(test, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
              #  pass

        #cv2.imshow("gray",test)
        #cv2.waitKey(0)

        #use mask
        #result = cv2.bitwise_and(img, img, mask=mask)


        lines = cv2.Canny(gray, threshold1=Thresh1, threshold2=Thresh2)
        #img_dilation = cv2.dilate(lines, kernel, iterations=1) 

        #HoughLines = cv2.HoughLinesP(img_dilation, 1, np.pi/180, hThresh, hMinLine, hMaxGap
        HoughLines = cv2.HoughLinesP(lines, 1, np.pi/180, threshold = hThresh, minLineLength = hMinLine, maxLineGap = hMaxGap)
        if HoughLines is not None:
            for line in HoughLines:
                coords = line[0]
                far_right, far_left = line_equation(coords)
                cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), [0,0,255], 3)
                #cv2.line(img, far_left, far_right, [255,0,0], 3)
             
        cv2.imshow(f"Mask{i}", output2)
        cv2.imshow(f"Original{i}", img)


        cv2.imshow(f"Lines{i}", lines)

        cv2.waitKey(1)