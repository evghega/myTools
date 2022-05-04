import numpy as np
import cv2


#==========================================
def imnegative(im):
    imneg = 255 - im
    return imneg
#==========================================
def contrastStretching(im):
    b = im.max()
    a = im.min()
    print(a, b)
    # Converting im1 to float.
    c = im.astype(float)
    # Contrast stretching transformation.
    im1 = 255.0 * (c - a) / (b - a + 0.0000001)
    return im1
#==========================================
def clahe(im):
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE()#(clipLimit=2.0, tileGridSize=(20,20))
    cl1 = clahe.apply(im)
    return cl1
#=============================================
def gammaCorrection (im, gamma):

    # b is converted to type float.
    b1 = im.astype(float)
    # Maximum value in b1 is determined.
    b3 = np.max(b1)
    # b1 is normalized
    b2 = b1 / b3
    # gamma-correction exponent is computed.
    b4 = np.log(b2+0.0000001) * gamma
    # gamma-correction is performed.
    c = np.exp(b4) * 255.0
    # c is converted to type int.
    c1 = c.astype(int)
    # Displaying c1
    return c1
#=============================================

# ===============================================================================
# MAIN FUNCTION
# ===============================================================================
def main():
    filename = 'embryo.png'
    img = cv2.imread(filename, 0)

    img_clahe = clahe(img)
    img_hist_eq = cv2.equalizeHist(img)
    cv2.imwrite("hist_eq_output.png", img_hist_eq)
    cv2.imwrite("clahe_output.png", img_clahe)

#------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
