import cv2
from scipy import ndimage
import numpy as np



def  sobelFilter(img):
    # Compute gradients along the X and Y axis, respectively
    sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1)


    # The sobelX and sobelY images are now of the floating
    # point data type -- we need to convert them
    # back to an 8-bit unsigned integer
    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = np.uint8(np.absolute(sobelY))

    # combine Sobel gradient images using bitwise OR
    sobelCombined = cv2.bitwise_or(sobelX, sobelY)


    cv2.imwrite('../figures/sobelX.png', sobelX)
    cv2.imwrite('../figures/sobelY.png', sobelY)
    cv2.imwrite('../figures/sobel.png', sobelCombined)

def  prewittDerivates(img):
    imgPrewitt_v = ndimage.prewitt(img, axis=0)
    imgPrewitt_h = ndimage.prewitt(img,axis=1)
    imgPrewitt = imgPrewitt_v + imgPrewitt_h

    cv2.imwrite('../figures/prewitt_h.png', imgPrewitt_h)
    cv2.imwrite('../figures/prewitt_v.png', imgPrewitt_v)
    cv2.imwrite('../figures/prewitt.png', imgPrewitt)

#=================================================
def laplacianFilter(img):
    # Compute the Laplacian of the image
    lap = cv2.Laplacian(img, cv2.CV_64F)
    lap = np.uint8(np.absolute(lap))
    cv2.imwrite('../figures/laplacian.png', lap)
#=================================================
def laplacianOfgaussiansFilter(img):
    lapOfGauss = ndimage.filters.gaussian_laplace(img, sigma=1, mode='reflect')
    cv2.imwrite('../figures/lapOfGauss.png', lapOfGauss)
#=================================================
def cannyFilter(img):
    canny = cv2.Canny(img,100,200)
    cv2.imwrite('../figures/canny.png', canny)


# ===============================================================================
# MAIN FUNCTION
# ===============================================================================
def main():
    filename = '../figures/cameraman.tif'
    img = cv2.imread(filename, 0)

    sobelFilter(img)
    prewittDerivates(img)
    cannyFilter(img)
    laplacianFilter(img)

#------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
