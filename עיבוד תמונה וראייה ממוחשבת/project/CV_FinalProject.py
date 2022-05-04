
import cv2
import numpy as np
import sys
import os
import pandas as pd
from skimage import feature

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


path_train = sys.argv[1]
path_val = sys.argv[2]
path_test = sys.argv[3]
output = path_test


### 1
inpu = path_test + '\\female'

test_image = []
test_label = []

for name in os.listdir(inpu):
    
    img = cv2.imread(str(inpu) + '/' + str(name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    test_image.append(img)
    test_label.append(int(0))

inpu = path_test + '\\male'

for name in os.listdir(inpu):
    
    img = cv2.imread(str(inpu) + '/' + str(name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    test_image.append(img)
    test_label.append(int(1))
    
    
test_df = pd.DataFrame({'image': test_image, 'label': test_label})

inpu = path_train + '\\female'

train_image = []
train_label = []

for name in os.listdir(inpu):
    
    img = cv2.imread(str(inpu) + '/' + str(name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    train_image.append(img)
    train_label.append(int(0))

inpu = path_train + '\\male'

for name in os.listdir(inpu):
    
    img = cv2.imread(str(inpu) + '/' + str(name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    train_image.append(img)
    train_label.append(int(1))
    
    
train_df = pd.DataFrame({'image': train_image, 'label': train_label})

inpu = path_val + '\\female'

valid_image = []
valid_label = []

for name in os.listdir(inpu):
    
    img = cv2.imread(str(inpu) + '/' + str(name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    valid_image.append(img)
    valid_label.append(int(0))

inpu = path_val + '\\male'

for name in os.listdir(inpu):
    
    img = cv2.imread(str(inpu) + '/' + str(name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    valid_image.append(img)
    valid_label.append(int(1))
    
    
valid_df = pd.DataFrame({'image': valid_image, 'label': valid_label})

test_df = test_df.sample(frac=1)
train_df = train_df.sample(frac=1)
valid_df = valid_df.sample(frac=1)


### 2
numP = 24
r = 3

def lbphist(img):
    numPoints = numP
    radius = r

    lbp = feature.local_binary_pattern(img, numPoints, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=range(0, numPoints + 3), range=(0, numPoints + 2))

    eps = 1e-7
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    
    return hist


def hist(df):

    hist = []
    for img in df['image']:
        hist.append(lbphist(img))

    return hist

train_df['hist'] = hist(train_df)
test_df['hist'] = hist(test_df)
valid_df['hist'] = hist(valid_df)


X_train_df = np.array(list(train_df['hist']), dtype=np.float64)
X_test_df = np.array(list(test_df['hist']), dtype=np.float64)
X_valid_df = np.array(list(valid_df['hist']), dtype=np.float64)

y_train_df = train_df['label']
y_test_df = test_df['label']
y_valid_df = valid_df['label']


### linear
clf = SVC(kernel='linear')
clf.fit(X_train_df, y_train_df)

train_pred = clf.predict(X_train_df)
valid_pred = clf.predict(X_valid_df)

train_acc = accuracy_score(y_train_df, train_pred)
valid_acc = accuracy_score(y_valid_df, valid_pred)


### RBF
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]}
clf = GridSearchCV(SVC(kernel='rbf'), param_grid, verbose=1)
clf.fit(X_train_df, y_train_df)

train_pred = clf.predict(X_train_df)
valid_pred = clf.predict(X_valid_df)

train_acc = accuracy_score(y_train_df, train_pred)
valid_acc = accuracy_score(y_valid_df, valid_pred)


### RBF
clf = SVC(kernel='rbf', C = 100, gamma=10)
clf.fit(X_train_df, y_train_df)

train_pred = clf.predict(X_train_df)
valid_pred = clf.predict(X_valid_df)

train_acc = accuracy_score(y_train_df, train_pred)
valid_acc = accuracy_score(y_valid_df, valid_pred)


### 4
### RBF
clf = SVC(kernel='rbf', C = 100, gamma=10)
clf.fit(X_train_df, y_train_df)

train_pred = clf.predict(X_train_df)
valid_pred = clf.predict(X_valid_df)
test_pred = clf.predict(X_test_df)

train_acc = accuracy_score(y_train_df, train_pred)
valid_acc = accuracy_score(y_valid_df, valid_pred)
test_acc = accuracy_score(y_test_df, test_pred)
acc = str("%.2f" % (test_acc*100))



from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test_df, test_pred)
table1 = '      | male | female\nmale  | ' + str(conf_matrix[0][0]) + '   | ' + str(conf_matrix[0][1])
table2 = '\nfemale| ' + str(conf_matrix[1][0]) + '    | ' + str(conf_matrix[1][1])
table3 = table1 + table2


f = open(os.path.join(output ,'results.txt'),"w+")
f.write("Number of points: " + str(numP) + "\nRadius: " + str(r) + '\nKernel: RBF\nC: 100\nGamma: 10\n' + "Accuracy: "+ acc + "%\n\n" + table3)
f.close()