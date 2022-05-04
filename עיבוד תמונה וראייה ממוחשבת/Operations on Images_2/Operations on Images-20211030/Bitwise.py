import cv2
import matplotlib.pyplot as plt
import numpy as np

rectangle = cv2.imread('rectangle.jpg', cv2.IMREAD_GRAYSCALE)
circle = cv2.imread('circle.jpg', cv2.IMREAD_GRAYSCALE)

bitwiseAnd = cv2.bitwise_and(rectangle, circle)
bitwiseOr = cv2.bitwise_or(rectangle, circle)
bitwiseXor = cv2.bitwise_xor(rectangle, circle)
bitwiseNot = cv2.bitwise_not(rectangle, rectangle)

# Display the images
plt.figure()
plt.subplot(141); plt.imshow(bitwiseAnd, cmap = 'gray'); plt.title('AND')
plt.subplot(142); plt.imshow(bitwiseOr, cmap = 'gray');  plt.title('OR')
plt.subplot(143); plt.imshow(bitwiseXor, cmap = 'gray'); plt.title('XOR')
plt.subplot(144); plt.imshow(bitwiseNot, cmap = 'gray'); plt.title('NOT')
plt.show()


########################### MASKING ###############################
img = cv2.imread('Billy.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

mask = np.zeros(img.shape[:2], dtype="uint8")
cv2.circle(mask, (500, 250), 250, 255, -1)
masked = cv2.bitwise_and(img, img, mask=mask)

plt.figure()
plt.subplot(131); plt.imshow(img); plt.title('Original')
plt.subplot(132); plt.imshow(mask, cmap = 'gray');  plt.title('Mask')
plt.subplot(133); plt.imshow(masked); plt.title('Result')
plt.show()

