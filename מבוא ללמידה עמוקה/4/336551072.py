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


test = pd.read_csv("C:\\MyFoulder\\SCE\שנה ג\\סמסטר ב\\מבוא ללמידה עמוקה\\4\\test_ex4_dl2021b.csv")
train = pd.read_csv("C:\\MyFoulder\\SCE\שנה ג\\סמסטר ב\\מבוא ללמידה עמוקה\\4\\training_ex4_dl2021b.csv")
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


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[14]:


X_train
#X_test
#y_train
#y_test


# In[ ]:





# In[15]:


# y_trains = []
# y_tests = []
# for t in range(len(y_train)):
#     y_trains.append(int(0))
# for t in range(len(y_test)):
#     y_tests.append(int(0))


# In[16]:


#y_trains


# In[17]:


#y_tests


# In[18]:


# from sklearn.utils import shuffle
# X_train, y_trains = shuffle(X_train, y_trains)
# X_test, y_tesst = shuffle(X_test, y_tests)


# In[ ]:





# In[19]:


sentences = [w.split() for w in X_test['sentence']]
print(len(sentences))
model = Word2Vec(sentences, min_count=1, vector_size=300)
print(model)

words = list(model.wv.index_to_key)
len(words)

temp = []

for s in X_test['sentence'].tolist():
    temp.append(len(s.split()))

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
        vecs2.append(numpy.array([zer for x in range(40)]))
    
for i in range(len(vecs2)):
    while len(vecs2[i])<40:
        vecs2[i] = numpy.vstack([vecs2[i], zer])

X_test_vecs = vecs2


# In[ ]:





# In[20]:


import matplotlib
import matplotlib.pyplot as plt

plt.plot(range(0,len(temp)), temp)                                             
plt.xlabel('Sentense index')
plt.ylabel('Lenght')
plt.xticks()
plt.title('Sentence lenght')
plt.yticks()
plt.style.use('default')
plt.show()


# In[ ]:





# In[21]:


numpy.array(X_test_vecs).shape


# In[22]:


sentences = [w.split() for w in X_train['sentence']]
print(len(sentences))
model = Word2Vec(sentences, min_count=1, vector_size=300)
print(model)

words = list(model.wv.index_to_key)
len(words)

temp2 = []

for s in X_test['sentence'].tolist():
    temp2.append(len(s.split()))

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
        vecs2.append(numpy.array([zer for x in range(40)]))
    
for i in range(len(vecs2)):
    while len(vecs2[i])<40:
        vecs2[i] = numpy.vstack([vecs2[i], zer])

X_train_vecs = vecs2


# In[ ]:





# In[ ]:





# In[23]:


plt.plot(range(0,len(temp2)), temp2)                                             
plt.xlabel('Sentense index')
plt.ylabel('Lenght')
plt.xticks()
plt.title('Sentence lenght')
plt.yticks()
plt.style.use('default')
plt.show()


# In[ ]:





# In[24]:


numpy.array(vecs2).shape


# In[25]:


# from sklearn.utils import shuffle
# X_train_vecs, y_train = shuffle(X_train_vecs, y_train)
# X_test_vecs, y_test = shuffle(X_test_vecs, y_test)


# In[26]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import LSTM, GRU, Dropout, Bidirectional


# In[27]:


model = Sequential()
#model.add(GRU(40, return_sequences=True, dropout=0.5,recurrent_dropout=0.5, input_shape=(40,300)))
#model.add(GRU(40, return_sequences=False, dropout=0.5,recurrent_dropout=0.5))

model.add(Bidirectional(LSTM(40, return_sequences=False, dropout=0.1,recurrent_dropout=0.1, input_shape=(40,300)))) ### !!!
#model.add((LSTM(40, return_sequences=False, dropout=0.1,recurrent_dropout=0.1))) #only last is false
#model.add(Dense(20,  activation=None))
#model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#model.summary()

#model.fit(numpy.array(X_train_vecs), y_train, validation_data=(numpy.array(X_test_vecs), y_test), epochs=10) ###batch size???
model.fit(numpy.array(X_train_vecs), y_train, epochs=10, batch_size=10)


# In[28]:


model.summary()

loss, accuracy = model.evaluate(numpy.array(X_train_vecs), y_train, verbose=False)
print('Train Accuracy: ', accuracy)
loss, accuracy = model.evaluate(numpy.array(X_test_vecs), y_test, verbose=False)
print('Test Accuracy: ', accuracy)


# In[29]:


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
        vecs2.append(numpy.array([zer for x in range(40)]))
    
for i in range(len(vecs2)):
    while len(vecs2[i])<40:
        vecs2[i] = numpy.vstack([vecs2[i], zer])

f_vecs = vecs2


# In[30]:


test_pred = (model.predict_classes(numpy.array(f_vecs)) > 0.5).astype("int32")
test_pred


# In[ ]:





# In[31]:


j = 0
for i in range(len(test_pred)):
    if test_pred[i]==1:
        j+=1
print(j)


# In[32]:


df3 = pd.read_csv("C:\\MyFoulder\\SCE\שנה ג\\סמסטר ב\\מבוא ללמידה עמוקה\\4\\sampleSubmission.csv")
df3['label'] = test_pred

df3.to_csv('Project4.14.csv', index = False)


# In[33]:


print('Time elapsed in (hh:mm:ss.ms): "{}"'.format(datetime.now() - start_time))


# In[ ]:





# In[ ]:




