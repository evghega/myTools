import cv2
import sys
import os
import numpy as np
from matplotlib import pyplot as plt


def main(sourceFilename):
    img = cv2.imread(sourceFilename)
    rows, cols = img.shape[:2]

    # ===================== rotation======================
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 30, 0.5)
    dst = cv2.warpAffine(img, M, (cols, rows))

    cv2.imwrite('Rotation.jpg', dst)

    #===================== affine ======================
    pts1 = np.float32([[50,50],[200,50],[50,200]])
    pts2 = np.float32([[10,100],[200,50],[100,250]])

    M = cv2.getAffineTransform(pts1,pts2)
    dst = cv2.warpAffine(img,M,(cols,rows))

    cv2.imwrite('Affine.jpg', dst)

    # ===================== perspective ======================
    imgCopy= img.copy()
    cv2.circle(imgCopy, (993, 1312), 50, (0, 255, 0), -1)
    cv2.circle(imgCopy, (215, 2801), 50, (0, 255, 0), -1)
    cv2.circle(imgCopy, (3104, 813), 50, (0, 255, 0), -1)
    cv2.circle(imgCopy, (3848, 2367), 50, (0, 255, 0), -1)
    cv2.imwrite('Original.jpg', imgCopy)

    pts1 = np.float32([[993, 1312],[215, 2801], [3104, 813], [3848, 2367]])
    pts2 = np.float32([[0, 0],[0,rows], [cols, 0], [cols, rows]])

    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,(cols, rows))

    cv2.imwrite('Perspective.jpg', dst)

# ------------------------------------------------------------------------------
if __name__ == "__main__":
    if(len(sys.argv) < 2):
        print("the use \' Transformations.py '\sourcfilename ")
    main(sys.argv[1])
