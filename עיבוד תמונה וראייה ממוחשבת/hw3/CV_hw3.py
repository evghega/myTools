#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import sys
import os


# In[2]:


### 1


# In[3]:


img_left = sys.argv[1]
img_right = sys.argv[2]
output = sys.argv[3]


il = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
ir = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
imgGray_left = img_left
imgGray_right = img_right

# height and width percent of right image
percent = imgGray_right.shape[0]/imgGray_right.shape[1]

# resized right image
imgGray_right = cv2.resize(imgGray_right, (int(imgGray_left.shape[0]/percent), int(imgGray_left.shape[0])))

# create keypoints and descriptors
orb = cv2.ORB_create()
keypoints_left, descriptors_left = orb.detectAndCompute(il, None)
keypoints_right, descriptors_right = orb.detectAndCompute(ir, None)


# In[5]:


### 2


# In[6]:


# best distance function for ORB is NORM_HAMMING
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches1 = matcher.match(descriptors_left, descriptors_right)

# sort matches by distance
matches2 = sorted(matches1, key = lambda x:x.distance)

# take 20% of lowest distance
matches3 = matches2[:int(0.2*len(matches2))]

# draw matches for two images
imMatches = cv2.drawMatches(imgGray_left, keypoints_left, imgGray_right, keypoints_right, matches3, None)


# In[7]:


### 3


# In[8]:


# convert to vector
src_pts = np.float32([ keypoints_right[m.trainIdx].pt for m in matches3 ]).reshape(-1,1,2)
dst_pts = np.float32([ keypoints_left[m.queryIdx].pt for m in matches3 ]).reshape(-1,1,2)

H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC) ### check sides


# In[10]:


### 4


# In[11]:


rows = int(imgGray_left.shape[0])
cols_left = int(imgGray_left.shape[1])
cols_right = int(imgGray_right.shape[1])

warp = cv2.warpPerspective(imgGray_right, H, (cols_left + cols_right, rows))


# In[13]:


### 5


# In[14]:


warp[0:imgGray_left.shape[0], 0:imgGray_left.shape[1]] = imgGray_left
res = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
stam,thresh = cv2.threshold(res, 1, 255, cv2.THRESH_BINARY)

contours, stam = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]

x,y,w,h = cv2.boundingRect(cnt)
panorama = warp[y:y+h,x:x+w]


# In[15]:


cv2.imwrite(os.path.join(output ,'Panorama.jpg'),panorama)

