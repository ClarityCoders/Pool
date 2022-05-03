import cv2
import keyboard
import mss
import numpy as np
import os
from utils import find_lines

SCT = mss.mss()



while not keyboard.is_pressed("q"):
    scr = SCT.grab({
                'left': 150,
                'top': 200,
                'width': 1600,
                'height': 880
            })
    img_org = np.array(scr)
    if keyboard.is_pressed("s"):
        cv2.imwrite("test2.png",img_org)
    img = find_lines(img_org)
    cv2.imshow("input", img)
    cv2.waitKey(1)