import cv2
import numpy as np

def DrawImage(img, x1, x2 , y):
    kernel = np.ones((2,2), np.uint8)
    color = cv2.cvtColor(img, cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, np.array([151, 96, 175]), np.array([154, 255, 255]))
    color[mask>0]=(255,255,152)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lines = cv2.Canny(color, threshold1=271, threshold2=398)
    img_dilation = cv2.dilate(lines, kernel, iterations=1) 

    cv2.imshow(f"Dilation", img_dilation)
    cv2.waitKey(0)

    HoughLines = cv2.HoughLinesP(img_dilation, 1, np.pi/180, 37, 3, 22)
    if HoughLines is not None:
        for line in HoughLines:
            coords = line[0]
            cv2.line(color, (coords[0], coords[1]), (coords[2], coords[3]), [0,255,255], 3)

    cv2.imshow(f"HoughLines", color)
    cv2.waitKey(0)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=17, minRadius=10, maxRadius=14)

    if circles is not None:
        circles = np.uint16(circles)
        for pt in circles[0, :]:
            x, y, r = pt[0], pt[1], pt[2]
            cv2.circle(color, (x,y), r, (0, 0, 255), 5)
    else:
        print("No Circles")

    color = cv2.circle(color, (x1,y), radius=2, color=(0, 0, 0), thickness=-1)
    color = cv2.circle(color, (x2,y), radius=2, color=(0, 0, 255), thickness=1)
    cv2.imshow(f"Circles", color)
    cv2.waitKey(0)


if __name__ == "__main__":  
    img = cv2.imread('test.png', cv2.IMREAD_COLOR)


    l_h = 0
    l_s = 0
    l_v = 235
    u_h = 196
    u_s = 4
    u_v = 255
    Thresh1 = 146
    Thresh2 = 61
    hThresh = 31
    hMinLine = 18
    hMaxGap = 2

    lower_mask = np.array([l_h, l_s, l_v])
    upper_mask = np.array([u_h, u_s, u_v])

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask2 = cv2.inRange(hsv, lower_mask, upper_mask)
    output2 = cv2.bitwise_and(img,img, mask= mask2)


    DrawImage(output2, 218, 240, 23)