import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===============================================================================
# MAIN FUNCTION
# ===============================================================================
def main():
    filename = '../figures/lena.jpg'
    img = cv2.imread(filename, 0)

    kernel = np.ones((5, 5), np.float32) / 25
    dst = cv2.filter2D(img, -1, kernel)

    plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(dst, cmap='gray'), plt.title('Averaging')
    plt.xticks([]), plt.yticks([])
    plt.show()

    blur = cv2.blur(img, (5, 5))
    cv2.imwrite('../figures/lenaBlur5.jpg', blur)

    imGaussian5 =  cv2.GaussianBlur(img, ksize = None, sigmaX=5)
    cv2.imwrite('../figures/lenaGaussian5.jpg', imGaussian5)

    imGaussian7 = cv2.GaussianBlur(img, ksize = None, sigmaX=7)
    cv2.imwrite('../figures/lenaGaussian7.jpg', imGaussian7)

    median = cv2.medianBlur(img,5)
    cv2.imwrite('../figures/Median.jpg', median)
#------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
