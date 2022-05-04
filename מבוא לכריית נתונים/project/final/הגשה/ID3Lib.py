import numpy
import pandas as pd
from sklearn import tree
from sklearn import preprocessing
import pickle

"""
יבגני גאיסינסקי 336551072
שניר בן יוסף 307908699
"""


def ID3(df, dftest):
    #turns values into integers
    le = preprocessing.LabelEncoder()
    ############################################################TRAIN#############################################################
    # transform class to list and converting class into numbers
    df['class'] = le.fit_transform(df['class'])
    train_label = df['class'].tolist()

    dftest['class'] = le.fit_transform(dftest['class'])
    test_label = dftest['class'].tolist()

    df.dropna(subset=['class'])  # Drop the rows where class is missing.
    dftest.dropna(subset=['class'])  # Drop the rows where class is missing.
    
    # converting string and bins into numbers
    for col in df:
        if type(df[col][0]) == numpy.int64:  # numeric discretization
            df[col] = le.fit_transform(pd.qcut(df[col], duplicates="drop"))  # Equal frequency func
            dftest[col] = le.fit_transform(pd.qcut(dftest[col], duplicates="drop"))  # Equal frequency func

        else:
            df[col] = le.fit_transform(df[col].tolist())
            dftest[col] = le.fit_transform(dftest[col].tolist())

    #build the tree based on train data
    X = df.drop('class', axis=1)
    y = df['class']
    classifier = tree.DecisionTreeClassifier()
    classifier.fit(X, y)

    #save model
    filename = 'final_ID3.sav'
    pickle.dump(classifier, open(filename, 'wb'))

    x_test = dftest.drop('class', axis=1)
    y_pred = classifier.predict(x_test)

    #count the hits and calculate success rate in %
    currect = 0
    counter = 0
    for i in range(0, len(y_pred)):
        counter += 1
        if y_pred[i] == list(dftest["class"])[i]:
            currect += 1
    print("Success percent is: ", currect / counter * 100, "%")
