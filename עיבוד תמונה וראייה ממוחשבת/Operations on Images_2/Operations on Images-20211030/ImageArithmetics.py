import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('FamilyTrip.jpg', cv2.IMREAD_COLOR)
# Create a matrix with constant intensity.
matrix = np.ones(img.shape, dtype = 'uint8') * 100

# Create brighter and darker images.
img_brighter = cv2.add(img, matrix)
img_darker   = cv2.subtract(img, matrix)

# Display the images
plt.figure()
plt.subplot(131); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB));   plt.title('Original')
plt.subplot(132); plt.imshow(cv2.cvtColor(img_darker, cv2.COLOR_BGR2RGB));   plt.title('Darker')
plt.subplot(133); plt.imshow(cv2.cvtColor(img_brighter, cv2.COLOR_BGR2RGB)); plt.title('Brighter')
plt.show()

#----------------------------------------------------------------------------------------------
img = cv2.imread('FamilyTrip.jpg', cv2.IMREAD_COLOR)
opencv_logo = cv2.imread('opencv_logo.png', cv2.IMREAD_COLOR)
h,w = img.shape[0], img.shape[1]
opencv_logo = cv2.resize(opencv_logo, (w,h))
img_add = cv2.add(img, opencv_logo)

plt.figure()
plt.subplot(131); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB));   plt.title('First image')
plt.subplot(132); plt.imshow(cv2.cvtColor(opencv_logo, cv2.COLOR_BGR2RGB));   plt.title('Second image')
plt.subplot(133); plt.imshow(cv2.cvtColor(img_add, cv2.COLOR_BGR2RGB)); plt.title('Add')
plt.show()