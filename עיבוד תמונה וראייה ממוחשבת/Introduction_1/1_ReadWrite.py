import matplotlib.pyplot as plt
import cv2


#=============================== read/write using OPEN CV============================
# cv2 module's imread to read an image as an ndarray.
#always check the type, cv doesn't produce error message when something is wrong in read
imgColor = cv2.imread('RGBimage.jpg', cv2.IMREAD_COLOR)
if imgColor is None:
    print('error in reading the file')
    exit()

#print the shape of the image
print(imgColor.shape)
# IMPORTANT! OpenCV reads the image in BGR format, while other libraries, e.g., matplotlib,  use RGB format
# converting from BGR to RGB
imgColor_fixed = cv2.cvtColor(imgColor, cv2.COLOR_BGR2RGB)

#display the image before and after the color conversion
plt.subplot(121); plt.imshow(imgColor); plt.title('Before color conversion')
plt.subplot(122); plt.imshow(imgColor_fixed); plt.title('After color conversion')
plt.show()

#convert to grayscale and save the result
imgGrayscale = cv2.imread('RGBimage.jpg', cv2.IMREAD_GRAYSCALE)
#print the shape of the image
print(imgColor.shape)
cv2.imwrite('imgGrayscale.jpg', imgGrayscale)
##===================================================================================================================

# split color channels
colorImg = cv2.imread('RGBimage.jpg', cv2.IMREAD_COLOR)
b,g,r=cv2.split(colorImg)
print(b.shape)

#plt.figure()
plt.subplot(141); plt.imshow(r, cmap = 'gray'); plt.title('Red Channel')
plt.subplot(142); plt.imshow(g, cmap = 'gray'); plt.title('Green Channel')
plt.subplot(143); plt.imshow(b, cmap = 'gray'); plt.title('Blue Channel')
# Merge the individual channels into a BGR image.
imgMerged = cv2.merge((r, g, b))
# Display the merged output.
plt.subplot(144); plt.imshow(imgMerged); plt.title('Merged Output')
plt.show()

# cv2.imwrite('Red.jpeg', r)
# cv2.imwrite('Green.jpeg', g)
# cv2.imwrite('Blue.jpeg', b)

#==============================================================================================
# horizontal and vertical flip
imgColor = cv2.imread('RGBimage.jpg', cv2.IMREAD_COLOR)
imgColor = cv2.cvtColor(imgColor, cv2.COLOR_BGR2RGB)

imgH = cv2.flip(imgColor,0)
imgV = cv2.flip(imgColor,1)
imgVH = cv2.flip(imgColor,-1)

plt.subplot(141);  plt.imshow(imgColor); plt.title("Original image")
plt.subplot(142);  plt.imshow(imgH); plt.title("Horizontal flip")
plt.subplot(143);  plt.imshow(imgV); plt.title("Vertical flip")
plt.subplot(144);  plt.imshow(imgVH); plt.title("Horizontal and vertical flip")
plt.show()

#===================RESIZE================================================
# this will resize both axes by half:
small = cv2.resize(imgColor, (0,0), fx=0.5, fy=0.5)
# this will resize the image to have 100 cols (width) and 50 rows (height):
resized_image = cv2.resize(imgColor, (100, 50))

print(imgColor.shape)
print(small.shape)
print(resized_image.shape)

#======================================================================
#interpolations
im = cv2.imread('letterA.tif', 0)
im = cv2.resize(im, (100,100))
imNearest = cv2.resize(im, None, fx=3, fy=3, interpolation = cv2.INTER_NEAREST)
imLinear = cv2.resize(im, None, fx=3, fy=3, interpolation = cv2.INTER_LINEAR)
imBicubic = cv2.resize(im, None, fx=3, fy=3, interpolation = cv2.INTER_CUBIC)

cv2.imwrite('original.tif',im )
cv2.imwrite('Nearest.tif',imNearest )
cv2.imwrite('Bilinear.tif',imLinear )
cv2.imwrite('Bicubic.tif',imBicubic )