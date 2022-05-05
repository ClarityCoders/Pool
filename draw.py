import cv2
import mss
import numpy as np

SCT = mss.mss()
line = []
points = []
lock = True

def DrawLine(event, x, y, flags, param):
    global line, lock, points

    if event == cv2.EVENT_LBUTTONDOWN:
        line = [(x, y)]
        lock = False
    elif event == cv2.EVENT_LBUTTONUP:
        if len(line) > 1:
            line[1] = (x,y)
        else:
            line.append((x,y))
        lock = True
    elif len(line) == 1:
        line.append((x, y))
    elif lock == False and len(line) == 2:
        line[1] = (x,y)
    elif event == cv2.EVENT_RBUTTONDOWN:
        points.append((x, y))

cv2.namedWindow("PoolShark")
cv2.setMouseCallback("PoolShark", DrawLine)
while True:
    scr = SCT.grab({
                'left': 150,
                'top': 200,
                'width': 1600,
                'height': 880
            })
    img = np.array(scr)
    if len(line) == 2:
        cv2.line(img, line[0], line[1], [180,105,255], 2)
    
    if len(line) >= 1:
        for point in points:
            cv2.line(img, line[0], point, [255,0,0], 1)

    for point in points:
        cv2.circle(img, point, radius=4, color=(0, 255, 0), thickness=-1)

    cv2.imshow("PoolShark", img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("r"):
        points = []