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


# In[ ]:





# In[2]:


test = pd.read_csv("C:\\MyFoulder\\SCE\שנה ג\\סמסטר ב\\מבוא ללמידה עמוקה\\2\\test_ex2_dl2021b.csv")
train = pd.read_csv("C:\\MyFoulder\\SCE\שנה ג\\סמסטר ב\\מבוא ללמידה עמוקה\\2\\train_ex2_dl2021b.csv")
train


# In[3]:


test


# In[4]:


train['sentence'] = train['sentence'].str.replace('[^\w\s]','')
train['sentence'] = train['sentence'].str.replace('[^a-zA-z ]','')

test['sentence'] = test['sentence'].str.replace('[^\w\s]','')
test['sentence'] = test['sentence'].str.replace('[^a-zA-z ]','')


# In[5]:


train.isnull().sum()


# In[6]:


test.isnull().sum()


# In[7]:


X = train.drop(['sid', 'label'], axis=1)
X


# In[8]:


y = train['label']
y


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[10]:


X_train
#X_test
#y_train
#y_test


# In[ ]:





# In[11]:


sentences = [w.split() for w in X_train['sentence']]
print(len(sentences))
model = Word2Vec(sentences, min_count=1, vector_size=300)
print(model)


# In[12]:


words = list(model.wv.index_to_key)
#print(model.wv.get_vector("the", norm=True))
#model.wv.most_similar('data')


# In[ ]:





# In[13]:


sen = list(X_train['sentence'])
avgs = []
vecs = []
for s in sen:
    for w in s.split():
        #if w in words:
        vecs.append(numpy.array(model.wv.get_vector(w, norm=True)))
    if len(s.split()) == 0:
        #avgs.append(sum(vecs)/1)
        avgs.append(numpy.array([avgs[0][0] for i in range(300)]))
        #print(avgs[len(avgs)-1])
    else:
        avgs.append(sum(vecs)/len(s.split()))
    vecs.clear()


# In[14]:


start_time = datetime.now() 


# In[15]:


def create_baseline():
    model = Sequential()
    model.add(Dense(300, input_dim=300, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=10, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, numpy.array(avgs), y_train, cv=kfold)

print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# In[ ]:





# In[16]:


print('Time elapsed in (hh:mm:ss.ms): "{}"'.format(datetime.now() - start_time))


# In[ ]:





# In[17]:


model2 = create_baseline()
model2.fit(numpy.array(avgs), y_train.values, epochs=100, batch_size=10, shuffle=True)


# In[ ]:





# In[18]:


sentences = [w.split() for w in test['sentence']]
print(len(sentences))
model = Word2Vec(sentences, min_count=1, vector_size=300)
print(model)


# In[19]:


sen = list(test['sentence'])
avgs = []
vecs = []
for s in sen:
    for w in s.split():
        vecs.append(numpy.array(model.wv.get_vector(w, norm=True)))
    if len(s.split()) == 0:
        avgs.append(numpy.array([avgs[0][0] for i in range(300)]))
    else:
        avgs.append(sum(vecs)/len(s.split()))
    vecs.clear()


# In[ ]:





# In[20]:


test_pred = (model2.predict_classes(numpy.array(avgs)) > 0.5).astype("int32")
#test_pred = model2.predict_classes(numpy.array(avgs))


# In[21]:


#test_pred


# In[22]:


#len(test_pred)


# In[23]:


j = 0
for i in range(430):
    if test_pred[i]==1:
        j+=1
print(j)


# In[24]:


# df3 = pd.read_csv("C:\\MyFoulder\\SCE\שנה ג\\סמסטר ב\\מבוא ללמידה עמוקה\\2\\sample_submission_ex2_dl2021b.csv")
# df3['label'] = test_pred

# df3.to_csv('Project2.11.csv', index = False)

#2.11 11 


# In[ ]:





# In[ ]:





# In[25]:


# kuku = pd.read_csv("Project2.8.csv")
# kuku_len = kuku.shape
# kuku_len


# In[ ]:





# In[ ]:





# In[26]:


# print('X_train len: ',len(X_train))
# print('y_train len: ',len(y_train))
# print('avgs len: ',len(avgs))
# print('vecs len: ',len(vecs))
# print('words: ',len(words))
# print('X_train: ',X_train.shape)
# print('y_train: ',y_train.shape)


# In[ ]:





# In[27]:


# for i in range(len(avgs)):
#     if isinstance(avgs[i], float):
#         print('fdbsfbsd')
#         print(avgs[i],i)
#         avgs[i]=numpy.array(avgs[i-1])


# In[ ]:





# In[28]:


# df = pd.read_csv("C:\MyFoulder\SCE\שנה ג\סמסטר ב\מבוא ללמידה עמוקה\kaggle\\train.csv")
# df
# df.isnull().sum()
# X = df.drop(['id', 'label'], axis=1)
# X
# y = df['label']
# y
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# X_train
# #X_test
# #y_train
# #y_test
# y_train.shape
# sentences = [w.split() for w in X_train['data']]
# print(len(sentences))
# model = Word2Vec(sentences, min_count=1, vector_size=300)
# print(model)
# words = list(model.wv.index_to_key)
# words
# #print(model.wv.get_vector("tea", norm=True))
# #model.wv.most_similar('data')
# q = []
# for x in words:
#     q.append(model.wv.get_vector(x, norm=True).tolist())
# print(type(q[0]))


# In[ ]:





# In[29]:


# from numpy import loadtxt
# from keras.models import Sequential
# from keras.layers import Dense


# # define the keras model
# model = Sequential()
# model.add(Dense(300, input_dim=300, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# # compile the keras model
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# # fit the keras model on the dataset
# model.fit(avgs, y, epochs=150, batch_size=10)
# # evaluate the keras model
# _, accuracy = model.evaluate(numpy.asarray(avgs), y_train)
# print('Accuracy: %.2f' % (accuracy*100))

