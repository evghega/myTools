import numpy
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
"""
יבגני גאיסינסקי 336551072
שניר בן יוסף 307908699
"""

def Kmeans(df, dftest):
    # pre process
    le = preprocessing.LabelEncoder()
    df['class'] = le.fit_transform(df['class'])
    train_label = df['class'].tolist()

    dftest['class'] = le.fit_transform(dftest['class'])
    test_label = dftest['class'].tolist()

    df.dropna(subset=['class'])  # Drop rows
    dftest.dropna(subset=['class'])  # Drop  rows


  #conver into numbers
    for col in df:
        if type(df[col][0]) == numpy.int64:  # numeric disc
            df[col] = le.fit_transform(pd.qcut(df[col], duplicates="drop"))  # Equal f requency func
            dftest[col] = le.fit_transform(pd.qcut(dftest[col], duplicates="drop"))  # Equal frequency func

        else:
            df[col] = le.fit_transform(df[col].tolist())
            dftest[col] = le.fit_transform(dftest[col].tolist())


    x = df.drop('class', axis=1)
    x = x.drop('index', axis = 1)
    y = df['class']
    x_test = dftest.drop('class', axis=1)
    y_test = dftest['class']

    n_clusters = int(input("clusters count: "))
    kmeans = KMeans(n_clusters= n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    pred_y = kmeans.fit_predict(x)


    dict = {}
    for i in range(0, n_clusters):
        dict[i] = [0,0]

    i = 0
    for ans in y:
        if ans == 'yes':
            dict[pred_y[i]][0] += 1
        else:
            dict[pred_y[i]][1] += 1
        i += 1


    for i in range(0, len(dict.keys())):
        if dict[i][0] > dict[i][1]:
            dict[i] = 'yes'
        else:
            dict[i] = 'no'

    for i in range(0, n_clusters):
        print(" number cluster: {0} : {1}".format(i, dict[i]))
