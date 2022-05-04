#!/usr/bin/env python
# coding: utf-8


import cv2
import numpy as np
import sys
import os


lower_yellow = np.array([22, 93, 0]) # hsv color space
upper_yellow = np.array([45, 255, 255]) # hsv color space

inpu = str(sys.argv[1])
outpu = str(sys.argv[2])

for name in os.listdir(inpu):

    img = cv2.imread(str(inpu) + '/' + str(name))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    not_yellow_mask = cv2.bitwise_not(yellow_mask)

    res = cv2.bitwise_and(img, img, mask = not_yellow_mask)

    not_yellow_mask2 = cv2.bitwise_not(not_yellow_mask)

    res_final = cv2.bitwise_not(res, res, mask = not_yellow_mask2)
    cv2.imwrite(str(outpu) + '/' + str(name), res_final)
