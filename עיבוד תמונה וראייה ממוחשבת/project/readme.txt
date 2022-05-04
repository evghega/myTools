The author's contact information:
Evghenii Gaisinschii 336551072

Title of the project:
Gender Classification from Handwriting.

Installation instruction:
Need to install following libraries: 
	import cv2
	import numpy as np
	import sys
	import os
	importimport pandas as pd
	from skimage import feature (!!!) Could be a problem with "skimage" install. Instead nead to install "scikit-image" package
	from sklearn.model_selection import GridSearchCV
	from sklearn.svm import SVC
	from sklearn.metrics import accuracy_score
	from sklearn.metrics import confusion_matrix


How to Run Your Program:
The programm needs to be executed in command line. 
Need to write python code file, train directory,validation directory and test directory where are all our images.
The result text file will be saved in test directory.
