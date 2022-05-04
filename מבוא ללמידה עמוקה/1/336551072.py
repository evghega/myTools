#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.naive_bayes import GaussianNB

from sklearn import tree

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier

from sklearn.neighbors import RadiusNeighborsClassifier

from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC

from sklearn.multiclass import OutputCodeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import RidgeClassifierCV

from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.linear_model import RidgeClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.naive_bayes import BernoulliNB

from sklearn.calibration import CalibratedClassifierCV

from sklearn.semi_supervised import LabelPropagation

from sklearn.semi_supervised import LabelSpreading

from sklearn.linear_model import LogisticRegressionCV

from sklearn.naive_bayes import MultinomialNB

from sklearn.neighbors import NearestCentroid

from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVC

from sklearn.linear_model import Perceptron

df = pd.read_csv("C:\\MyFoulder\\SCE\שנה ג\\סמסטר ב\\מבוא ללמידה עמוקה\\1\\training_ex1_dl2021b.csv")
df


# In[ ]:





# In[2]:


y = df['label']
X = df.drop(['ID', 'label'], axis=1)
X


# In[3]:


y


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# In[5]:


#X_train
#X_test
#y_train
#y_test


# In[6]:


X_train


# In[7]:


X_test


# In[8]:


y_train


# In[9]:


y_test


# In[ ]:





# In[10]:


train_for_graph = []
test_for_graph = []


# In[ ]:





# In[11]:



clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X_train, y_train)

print(clf.predict(X_test))
test_pred = clf.predict(X_test)
train_pred = clf.predict(X_train)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)
train_for_graph.append(train_acc)
test_for_graph.append(test_acc)
print("train_acc: ", train_acc)
print("test_acc: ", test_acc)


# In[ ]:





# In[12]:


from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

clf = HistGradientBoostingClassifier(max_iter=1000).fit(X_train, y_train)
#clf.score(X_test, y_test)

print(clf.predict(X_test))
test_pred = clf.predict(X_test)
train_pred = clf.predict(X_train)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)
train_for_graph.append(train_acc)
test_for_graph.append(test_acc)
print("train_acc: ", train_acc)
print("test_acc: ", test_acc)


# In[ ]:





# In[13]:



clf = GaussianNB()
clf.fit(X_train, y_train)

print(clf.predict(X_test))
test_pred = clf.predict(X_test)
train_pred = clf.predict(X_train)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)
train_for_graph.append(train_acc)
test_for_graph.append(test_acc)
print("train_acc: ", train_acc)
print("test_acc: ", test_acc)


# In[ ]:





# In[14]:


##########################4
clf = tree.DecisionTreeClassifier(random_state=1, max_depth=9)
clf = clf.fit(X_train, y_train)

print(clf.predict(X_test))
test_pred = clf.predict(X_test)
train_pred = clf.predict(X_train)


train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)
train_for_graph.append(train_acc)
test_for_graph.append(test_acc)
print("train_acc: ", train_acc)
print("test_acc: ", test_acc)


# In[ ]:





# In[15]:



clf = MLPClassifier(random_state=1, max_iter=400).fit(X_train, y_train)
clf.predict_proba(X_test)

print(clf.predict(X_test))
test_pred = clf.predict(X_test)
train_pred = clf.predict(X_train)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)
train_for_graph.append(train_acc)
test_for_graph.append(test_acc)
print("train_acc: ", train_acc)
print("test_acc: ", test_acc)


# In[ ]:





# In[16]:



neigh = KNeighborsClassifier(n_neighbors=6)
neigh.fit(X_train, y_train)

print(neigh.predict(X_test))
test_pred = neigh.predict(X_test)
train_pred = neigh.predict(X_train)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)
train_for_graph.append(train_acc)
test_for_graph.append(test_acc)
print("train_acc: ", train_acc)
print("test_acc: ", test_acc)


# In[ ]:





# In[17]:



kernel = 1.0 * RBF(1.0)
gpc = GaussianProcessClassifier(kernel=kernel, random_state=0).fit(X_train, y_train)

print(gpc.predict(X_test))
test_pred = gpc.predict(X_test)
train_pred = gpc.predict(X_train)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)
train_for_graph.append(train_acc)
test_for_graph.append(test_acc)
print("train_acc: ", train_acc)
print("test_acc: ", test_acc)


# In[ ]:





# In[18]:


################8
clf = RandomForestClassifier(random_state=0)
clf.fit(X_train, y_train)

print(clf.predict(X_test))
test_pred = clf.predict(X_test)
train_pred = clf.predict(X_train)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)
train_for_graph.append(train_acc)
test_for_graph.append(test_acc)
print("train_acc: ", train_acc)
print("test_acc: ", test_acc)


# In[ ]:





# In[19]:



clf = AdaBoostClassifier(n_estimators=30, random_state=0)
clf.fit(X_train, y_train)

print(clf.predict(X_test))
test_pred = clf.predict(X_test)
train_pred = clf.predict(X_train)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)
train_for_graph.append(train_acc)
test_for_graph.append(test_acc)
print("train_acc: ", train_acc)
print("test_acc: ", test_acc)


# In[ ]:





# In[20]:



clf = QuadraticDiscriminantAnalysis()
clf.fit(X_train, y_train)

print(clf.predict(X_test))
test_pred = clf.predict(X_test)
train_pred = clf.predict(X_train)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)
train_for_graph.append(train_acc)
test_for_graph.append(test_acc)
print("train_acc: ", train_acc)
print("test_acc: ", test_acc)


# In[ ]:





# In[21]:



clf = LogisticRegression().fit(X_train, y_train)

print(clf.predict(X_test))
test_pred = clf.predict(X_test)
train_pred = clf.predict(X_train)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)
train_for_graph.append(train_acc)
test_for_graph.append(test_acc)
print("train_acc: ", train_acc)
print("test_acc: ", test_acc)


# In[ ]:





# In[22]:



clf = GradientBoostingClassifier(random_state=0)
clf.fit(X_train, y_train)

print(clf.predict(X_test))
test_pred = clf.predict(X_test)
train_pred = clf.predict(X_train)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)
train_for_graph.append(train_acc)
test_for_graph.append(test_acc)
print("train_acc: ", train_acc)
print("test_acc: ", test_acc)


# In[ ]:





# In[23]:



clf = LinearDiscriminantAnalysis()
clf.fit(X_train, y_train)

print(clf.predict(X_test))
test_pred = clf.predict(X_test)
train_pred = clf.predict(X_train)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)
train_for_graph.append(train_acc)
test_for_graph.append(test_acc)
print("train_acc: ", train_acc)
print("test_acc: ", test_acc)


# In[ ]:





# In[24]:


##################14
clf = ExtraTreesClassifier(n_estimators=100, random_state=2)
clf.fit(X_train, y_train)

print(clf.predict(X_test))
test_pred = clf.predict(X_test)
train_pred = clf.predict(X_train)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)
train_for_graph.append(train_acc)
test_for_graph.append(test_acc)
print("train_acc: ", train_acc)
print("test_acc: ", test_acc)


# In[ ]:





# In[25]:


#####################15
clf = BaggingClassifier(base_estimator=SVC(), n_estimators=9, random_state=0).fit(X_train, y_train)

print(clf.predict(X_test))
test_pred = clf.predict(X_test)
train_pred = clf.predict(X_train)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)
train_for_graph.append(train_acc)
test_for_graph.append(test_acc)
print("train_acc: ", train_acc)
print("test_acc: ", test_acc)


# In[ ]:





# In[26]:



neigh = RadiusNeighborsClassifier(radius=2.0)
neigh.fit(X_train, y_train)

print(neigh.predict(X_test))
test_pred = neigh.predict(X_test)
train_pred = neigh.predict(X_train)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)
train_for_graph.append(train_acc)
test_for_graph.append(test_acc)
print("train_acc: ", train_acc)
print("test_acc: ", test_acc)


# In[ ]:





# In[27]:



clf = OneVsOneClassifier(LinearSVC(random_state=0)).fit(X_train, y_train)

print(clf.predict(X_test))
test_pred = clf.predict(X_test)
train_pred = clf.predict(X_train)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)
train_for_graph.append(train_acc)
test_for_graph.append(test_acc)
print("train_acc: ", train_acc)
print("test_acc: ", test_acc)


# In[ ]:





# In[28]:


######################18
clf = OutputCodeClassifier(estimator=RandomForestClassifier(random_state=0), random_state=0).fit(X_train, y_train)

print(clf.predict(X_test))
test_pred = clf.predict(X_test)
train_pred = clf.predict(X_train)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)
train_for_graph.append(train_acc)
test_for_graph.append(test_acc)
print("train_acc: ", train_acc)
print("test_acc: ", test_acc)


# In[ ]:





# In[29]:



clf = OneVsRestClassifier(SVC()).fit(X_train, y_train)

print(clf.predict(X_test))
test_pred = clf.predict(X_test)
train_pred = clf.predict(X_train)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)
train_for_graph.append(train_acc)
test_for_graph.append(test_acc)
print("train_acc: ", train_acc)
print("test_acc: ", test_acc)


# In[ ]:





# In[30]:



clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
clf.fit(X_train, y_train)


print(clf.predict(X_test))
test_pred = clf.predict(X_test)
train_pred = clf.predict(X_train)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)
train_for_graph.append(train_acc)
test_for_graph.append(test_acc)
print("train_acc: ", train_acc)
print("test_acc: ", test_acc)


# In[ ]:





# In[31]:



clf = RidgeClassifierCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(X_train, y_train)

print(clf.predict(X_test))
test_pred = clf.predict(X_test)
train_pred = clf.predict(X_train)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)
train_for_graph.append(train_acc)
test_for_graph.append(test_acc)
print("train_acc: ", train_acc)
print("test_acc: ", test_acc)


# In[ ]:





# In[32]:



clf = PassiveAggressiveClassifier(max_iter=1000, random_state=0, tol=1e-3)
clf.fit(X_train, y_train)

print(clf.predict(X_test))
test_pred = clf.predict(X_test)
train_pred = clf.predict(X_train)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)
train_for_graph.append(train_acc)
test_for_graph.append(test_acc)
print("train_acc: ", train_acc)
print("test_acc: ", test_acc)


# In[ ]:





# In[33]:


####################23
clf = RidgeClassifier().fit(X_train, y_train)

print(clf.predict(X_test))
test_pred = clf.predict(X_test)
train_pred = clf.predict(X_train)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)
train_for_graph.append(train_acc)
test_for_graph.append(test_acc)
print("train_acc: ", train_acc)
print("test_acc: ", test_acc)


# In[ ]:





# In[34]:



clf1 = LogisticRegression(multi_class='multinomial', random_state=1)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GaussianNB()

clf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
clf = clf.fit(X_train, y_train)


print(clf.predict(X_test))
test_pred = clf.predict(X_test)
train_pred = clf.predict(X_train)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)
train_for_graph.append(train_acc)
test_for_graph.append(test_acc)
print("train_acc: ", train_acc)
print("test_acc: ", test_acc)


# In[ ]:





# In[35]:



clf = BernoulliNB()
clf.fit(X_train, y_train)

print(clf.predict(X_test))
test_pred = clf.predict(X_test)
train_pred = clf.predict(X_train)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)
train_for_graph.append(train_acc)
test_for_graph.append(test_acc)
print("train_acc: ", train_acc)
print("test_acc: ", test_acc)


# In[ ]:





# In[36]:



base_clf = GaussianNB()
calibrated_clf = CalibratedClassifierCV(base_estimator=base_clf, cv=3)
calibrated_clf.fit(X_train, y_train)

print(clf.predict(X_test))
test_pred = clf.predict(X_test)
train_pred = clf.predict(X_train)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)
train_for_graph.append(train_acc)
test_for_graph.append(test_acc)
print("train_acc: ", train_acc)
print("test_acc: ", test_acc)


# In[ ]:





# In[37]:


#################27
label_prop_model = LabelPropagation()
label_prop_model.fit(X_train, y_train)

print(clf.predict(X_test))
test_pred = clf.predict(X_test)
train_pred = clf.predict(X_train)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)
train_for_graph.append(train_acc)
test_for_graph.append(test_acc)
print("train_acc: ", train_acc)
print("test_acc: ", test_acc)


# In[ ]:





# In[38]:



label_prop_model = LabelSpreading()
label_prop_model.fit(X_train, y_train)

print(clf.predict(X_test))
test_pred = clf.predict(X_test)
train_pred = clf.predict(X_train)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)
train_for_graph.append(train_acc)
test_for_graph.append(test_acc)
print("train_acc: ", train_acc)
print("test_acc: ", test_acc)


# In[ ]:





# In[39]:



clf = LogisticRegressionCV(cv=3, random_state=0).fit(X_train, y_train)

print(clf.predict(X_test))
test_pred = clf.predict(X_test)
train_pred = clf.predict(X_train)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)
train_for_graph.append(train_acc)
test_for_graph.append(test_acc)
print("train_acc: ", train_acc)
print("test_acc: ", test_acc)


# In[ ]:





# In[40]:



clf = MultinomialNB()
clf.fit(X_train, y_train)

print(clf.predict(X_test))
test_pred = clf.predict(X_test)
train_pred = clf.predict(X_train)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)
train_for_graph.append(train_acc)
test_for_graph.append(test_acc)
print("train_acc: ", train_acc)
print("test_acc: ", test_acc)


# In[ ]:





# In[41]:



clf = NearestCentroid()
clf.fit(X_train, y_train)

print(clf.predict(X_test))
test_pred = clf.predict(X_test)
train_pred = clf.predict(X_train)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)
train_for_graph.append(train_acc)
test_for_graph.append(test_acc)
print("train_acc: ", train_acc)
print("test_acc: ", test_acc)


# In[ ]:





# In[42]:



clf = make_pipeline(StandardScaler(), NuSVC())
clf.fit(X_train, y_train)

print(clf.predict(X_test))
test_pred = clf.predict(X_test)
train_pred = clf.predict(X_train)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)
train_for_graph.append(train_acc)
test_for_graph.append(test_acc)
print("train_acc: ", train_acc)
print("test_acc: ", test_acc)


# In[ ]:





# In[43]:



clf = Perceptron(tol=1e-3, random_state=2)
clf.fit(X_train, y_train)

print(clf.predict(X_test))
test_pred = clf.predict(X_test)
train_pred = clf.predict(X_train)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)
train_for_graph.append(train_acc)
test_for_graph.append(test_acc)
print("train_acc: ", train_acc)
print("test_acc: ", test_acc)


# In[ ]:





# In[ ]:





# In[44]:


import matplotlib
import matplotlib.pyplot as plt

print(max(train_for_graph))
print(max(test_for_graph))
plt.plot(range(len(train_for_graph)), train_for_graph, marker = "o", linewidth = 1, color = 'steelblue')
plt.plot(range(len(train_for_graph)), test_for_graph, marker = "o", linewidth = 1, color = 'orange')

plt.xlabel('Library number')
plt.ylabel('Probability')
plt.xticks(range(len(train_for_graph)))
plt.title('Libraries probability')
plt.style.use('default')
plt.yticks()
plt.style.use('ggplot')
plt.show()


# In[ ]:





# In[45]:


#plt.plot(range(len(train_for_graph)), train_for_graph, marker = "o", linewidth = 1, color = 'steelblue')
plt.plot(range(len(train_for_graph)), test_for_graph, marker = "o", linewidth = 1, color = 'orange')

plt.xlabel('Library number')
plt.ylabel('Probability')
plt.xticks(range(len(train_for_graph)))
plt.title('Libraries probability')
plt.style.use('default')
plt.yticks([0.75,max(test_for_graph)])
plt.style.use('ggplot')
plt.show()


# In[ ]:





# In[47]:


df2 = pd.read_csv("C:\\MyFoulder\\SCE\שנה ג\\סמסטר ב\\מבוא ללמידה עמוקה\\1\\test_ex1_dl2021b.csv")


clf = OneVsRestClassifier(SVC()).fit(X_train, y_train)

print(clf.predict(df2.drop(['ID'], axis=1)))
test_pred = clf.predict(df2.drop(['ID'], axis=1))


# In[ ]:





# In[54]:


df3 = pd.read_csv("C:\\MyFoulder\\SCE\שנה ג\\סמסטר ב\\מבוא ללמידה עמוקה\\1\\sample_ex1_dl2021b.csv")
df3['label'] = test_pred

df3.to_csv('Project1.15.csv', index = False)


# In[ ]:





# In[1]:


# df2 = pd.read_csv("C:\\MyFoulder\\SCE\שנה ג\\סמסטר ב\\מבוא ללמידה עמוקה\\1\\test_ex1_dl2021b.csv")


# clf = tree.DecisionTreeClassifier(random_state=1, max_depth=9)
# clf = clf.fit(X_train, y_train)

# print(clf.predict(df2.drop(['ID'], axis=1)))
# test_pred = clf.predict(df2.drop(['ID'], axis=1))


# In[ ]:





# In[48]:


# df2 = pd.read_csv("C:\\MyFoulder\\SCE\שנה ג\\סמסטר ב\\מבוא ללמידה עמוקה\\1\\test_ex1_dl2021b.csv")

# clf = BaggingClassifier(base_estimator=SVC(), n_estimators=9, random_state=0).fit(X_train, y_train)

# print(clf.predict(df2.drop(['ID'], axis=1)))
# test_pred = clf.predict(df2.drop(['ID'], axis=1))


# In[ ]:





# In[49]:


# df2 = pd.read_csv("C:\\MyFoulder\\SCE\שנה ג\\סמסטר ב\\מבוא ללמידה עמוקה\\1\\test_ex1_dl2021b.csv")

# clf1 = LogisticRegression(multi_class='multinomial', random_state=1)
# clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
# clf3 = GaussianNB()

# clf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
# clf = clf.fit(X_train, y_train)

# print(clf.predict(df2.drop(['ID'], axis=1)))
# test_pred = clf.predict(df2.drop(['ID'], axis=1))


# In[ ]:





# In[50]:


# df2 = pd.read_csv("C:\\MyFoulder\\SCE\שנה ג\\סמסטר ב\\מבוא ללמידה עמוקה\\1\\test_ex1_dl2021b.csv")

# clf = GradientBoostingClassifier(random_state=0)
# clf.fit(X_train, y_train)

# print(clf.predict(df2.drop(['ID'], axis=1)))
# test_pred = clf.predict(df2.drop(['ID'], axis=1))


# In[ ]:





# In[51]:


# df2 = pd.read_csv("C:\\MyFoulder\\SCE\שנה ג\\סמסטר ב\\מבוא ללמידה עמוקה\\1\\test_ex1_dl2021b.csv")

# clf = OutputCodeClassifier(estimator=RandomForestClassifier(random_state=0), random_state=0).fit(X_train, y_train)

# print(clf.predict(df2.drop(['ID'], axis=1)))
# test_pred = clf.predict(df2.drop(['ID'], axis=1))


# In[ ]:





# In[52]:


# df2 = pd.read_csv("C:\\MyFoulder\\SCE\שנה ג\\סמסטר ב\\מבוא ללמידה עמוקה\\1\\test_ex1_dl2021b.csv")

# clf = ExtraTreesClassifier(n_estimators=100, random_state=2)
# clf.fit(X_train, y_train)

# print(clf.predict(df2.drop(['ID'], axis=1)))
# test_pred = clf.predict(df2.drop(['ID'], axis=1))


# In[ ]:





# In[53]:


# df2 = pd.read_csv("C:\\MyFoulder\\SCE\שנה ג\\סמסטר ב\\מבוא ללמידה עמוקה\\1\\test_ex1_dl2021b.csv")

# clf = RandomForestClassifier(random_state=0)
# clf.fit(X_train, y_train)

# print(clf.predict(df2.drop(['ID'], axis=1)))
# test_pred = clf.predict(df2.drop(['ID'], axis=1))
# #train_pred = clf.predict(X_train.drop(['label'], axis=1))


# #train_acc = accuracy_score(y_train, train_pred)
# #test_acc = accuracy_score(y_test, test_pred)

# #print("train_acc: ", train_acc)
# #print("test_acc: ", test_acc)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




