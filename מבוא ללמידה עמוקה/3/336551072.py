#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec

from sklearn.decomposition import PCA
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

import time
from datetime import datetime 

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

start_time = datetime.now() 


# In[2]:


test = pd.read_csv("C:\\MyFoulder\\SCE\שנה ג\\סמסטר ב\\מבוא ללמידה עמוקה\\3\\ex3_test_data.csv")
train = pd.read_csv("C:\\MyFoulder\\SCE\שנה ג\\סמסטר ב\\מבוא ללמידה עמוקה\\3\\ex3_train_data.csv")
train


# In[3]:


test


# In[4]:


train['sentence'] = train['sentence'].str.replace('[^\w\s]','')
train['sentence'] = train['sentence'].str.replace('[^a-zA-z ]','')

test['sentence'] = test['sentence'].str.replace('[^\w\s]','')
test['sentence'] = test['sentence'].str.replace('[^a-zA-z ]','')


stop_words = set(stopwords.words('english'))

for ID in  range(len(train['sentence'])):
    querywords = train['sentence'][ID].split()
    resultwords  = [word for word in querywords if word not in stop_words]
    train['sentence'][ID] = ' '.join(resultwords)
    
for ID in  range(len(test['sentence'])):
    querywords = test['sentence'][ID].split()
    resultwords  = [word for word in querywords if word not in stop_words]
    test['sentence'][ID] = ' '.join(resultwords)


# In[5]:


train


# In[6]:


test


# In[7]:


train.isnull().sum()


# In[8]:


test.isnull().sum()


# In[9]:


train.dropna(subset = ['sentence'], inplace=True)
test.dropna(subset = ['sentence'], inplace=True)


# In[10]:


X = train.drop(['id', 'label'], axis=1)
X


# In[11]:


y = train['label']
y


# In[12]:


j = 0
for i in range(len(y)):
    if y[i]==1:
        j+=1
print('Num of 1\'s is: ', j)


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[14]:


X_train
#X_test
#y_train
#y_test


# In[ ]:





# In[15]:


sentences = [w.split() for w in X_test['sentence']]
print(len(sentences))
model = Word2Vec(sentences, min_count=1, vector_size=300)
print(model)

words = list(model.wv.index_to_key)
len(words)

zer = numpy.array([0.0 for x in range(300)])

sen = list(X_test['sentence'])
vecs = []
vecs2 = []
for s in sen:
    if len(s.split())<40:
        for w in s.split():
            vecs.append(numpy.array(model.wv.get_vector(w, norm=True)))
        if len(s.split()) == 0:
            vecs.append(zer)
        vecs2.append(numpy.array(vecs))
        vecs.clear()
    else:
        vecs2.append(numpy.array([zer for x in range(30)]))
    
for i in range(len(vecs2)):
    while len(vecs2[i])<40:
        vecs2[i] = numpy.vstack([vecs2[i], zer])

X_test_vecs = vecs2


# In[16]:


numpy.array(X_test_vecs).shape


# In[ ]:





# In[17]:


sentences = [w.split() for w in X_train['sentence']]
print(len(sentences))
model = Word2Vec(sentences, min_count=1, vector_size=300)
print(model)

words = list(model.wv.index_to_key)
len(words)

zer = numpy.array([0.0 for x in range(300)])

sen = list(X_train['sentence'])
vecs = []
vecs2 = []
for s in sen:
    if len(s.split())<40:
        for w in s.split():
            vecs.append(numpy.array(model.wv.get_vector(w, norm=True)))
        if len(s.split()) == 0:
            vecs.append(zer)
        vecs2.append(numpy.array(vecs))
        vecs.clear()
    else:
        vecs2.append(numpy.array([zer for x in range(30)]))
    
for i in range(len(vecs2)):
    while len(vecs2[i])<40:
        vecs2[i] = numpy.vstack([vecs2[i], zer])

X_train_vecs = vecs2


# In[18]:


numpy.array(vecs2).shape


# In[80]:


from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Conv1D, LeakyReLU, Dropout

model = Sequential()
model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(40,300)))
#model.add(LeakyReLU(alpha=0.2))
#model.add(Dropout(0.4))
model.add(Conv1D(64, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(numpy.array(X_train_vecs), y_train, validation_data=(numpy.array(X_test_vecs), y_test), epochs=5)


# In[88]:


sentences = [w.split() for w in test['sentence']]
print(len(sentences))
model2 = Word2Vec(sentences, min_count=1, vector_size=300)
print(model2)

words = list(model2.wv.index_to_key)
len(words)

zer = numpy.array([0.0 for x in range(300)])

sen = list(test['sentence'])
vecs = []
vecs2 = []
for s in sen:
    if len(s.split())<40:
        for w in s.split():
            vecs.append(numpy.array(model2.wv.get_vector(w, norm=True)))
        if len(s.split()) == 0:
            vecs.append(zer)
        vecs2.append(numpy.array(vecs))
        vecs.clear()
    else:
        vecs2.append(numpy.array([zer for x in range(30)]))
    
for i in range(len(vecs2)):
    while len(vecs2[i])<40:
        vecs2[i] = numpy.vstack([vecs2[i], zer])

f_vecs = vecs2


# In[94]:


test_pred = (model.predict_classes(numpy.array(f_vecs)) > 0.5).astype("int32")


# In[91]:


df3 = pd.read_csv("C:\\MyFoulder\\SCE\שנה ג\\סמסטר ב\\מבוא ללמידה עמוקה\\3\\ex3_sampleSubmission.csv")
df3['label'] = test_pred

df3.to_csv('Project3.12.csv', index = False)


# In[33]:


print('Time elapsed in (hh:mm:ss.ms): "{}"'.format(datetime.now() - start_time))


# In[ ]:




