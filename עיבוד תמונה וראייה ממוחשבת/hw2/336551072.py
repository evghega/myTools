#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import sys
import os
#import matplotlib.pyplot as plt

inpu = str(sys.argv[1])
outpu = str(sys.argv[2])


# In[4]:


### 1


# In[5]:


img = cv2.imread(inpu)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rows, cols = imgGray.shape[:2]

thres , imgB = cv2.threshold(imgGray, 135, 255, cv2.THRESH_BINARY)

cv2.imwrite('Desktop/ocv/list.jpg', imgB)


# In[6]:


### 2


# In[7]:


imgCopy = img.copy()
(cnts , _) = cv2.findContours(imgB ,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# In[8]:


### 3


# In[9]:


epsilon = 0
for i in range(len(cnts)):
    if 0.01*cv2.arcLength(cnts[i],True)>epsilon:
        epsilon = 0.01*cv2.arcLength(cnts[i],True)
        n = cnts[i]
        
approx = cv2.approxPolyDP(n,epsilon,True)


# In[10]:


### 4


# In[11]:


pt = [approx[0][0],approx[1][0],approx[2][0],approx[3][0]]
l = []
for i in range(4):
    l.append([pt[i][0],pt[i][1],pt[i][0]+pt[i][1]])


# In[12]:


pmin = l[0][2]
pmax = l[0][2]
pn = [l[0][0],l[0][1]]
px = [l[0][0],l[0][1]]
onetwo = []
points = []

for i in range(4):
    if l[i][2]<pmin:
        pmin = l[i][2]
        pn = [l[i][0],l[i][1]]
    if l[i][2]>pmax:
        pmax = l[i][2]
        px = [l[i][0],l[i][1]]
for i in range(4):
    if l[i][2] != pmin and l[i][2] != pmax:
        onetwo.append([l[i][0],l[i][1]])
        
points.append(pn)
points.append(px)

if onetwo[0][0]<onetwo[1][0]:
    points.append(onetwo[0])
    points.append(onetwo[1])
elif onetwo[0][0]>onetwo[1][0]:
    points.append(onetwo[1])
    points.append(onetwo[0])


# In[14]:


### 5


# In[17]:


rows, cols = imgGray.shape[:2]

pts1 = np.float32([points[0],points[2], points[3], points[1]])
pts2 = np.float32([[0 , 0],[0 ,rows],[cols, 0],[cols, rows]])

M = cv2.getPerspectiveTransform(pts1, pts2)

###6
dst = cv2.warpPerspective(img, M, (cols, rows))


### 7
cv2.imwrite(outpu + '\\' + str(inpu[-7:]) +'.jpg', dst)

