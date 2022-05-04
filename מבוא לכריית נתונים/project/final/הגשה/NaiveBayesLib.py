import pickle

from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn import preprocessing
import numpy
from sklearn.metrics import accuracy_score

"""
יבגני גאיסינסקי 336551072
שניר בן יוסף 307908699
"""

def naive(df, dftest, printerr=None):
    # preprocessing
    le = preprocessing.LabelEncoder()
    ############################################################TRAIN#############################################################
    # transform class to list and converting class into numbers
    df['class'] = le.fit_transform(df['class'])
    train_label = df['class'].tolist()

    dftest['class'] = le.fit_transform(dftest['class'])
    test_label = dftest['class'].tolist()


    # converting string and bins into numbers
    for col in df:
       if type(df[col][0]) == numpy.int64:  # numeric discretization
           df[col] = le.fit_transform(pd.qcut(df[col], duplicates="drop"))  # Equal frequency func
           dftest[col] = le.fit_transform(pd.qcut(dftest[col], duplicates="drop"))  # Equal frequency func
           df[col]=le.fit_transform(pd.cut(df[col],duplicates="drop"))          #Equal width func
           dftest[col]=le.fit_transform(pd.cut(dftest[col],duplicates="drop"))  #Equal width func

       else:
            df[col] = le.fit_transform(df[col].tolist())
            dftest[col] = le.fit_transform(dftest[col].tolist())

    ##################################################GAUSSIANNB################################################################
    # Create a Gaussian Classifier
    model = GaussianNB()

    # Train the model using the training sets
    model.fit(df.drop('class',axis=1), train_label)

    # saving the model using pickle
    filename = 'final_naive_bayes.sav'
    pickle.dump(model, open(filename, 'wb'))


    # Predict Output
    predicted = model.predict(dftest.drop('class',axis=1))
    accuracy = accuracy_score(test_label, predicted) * 100
    print("Success percent is: %.2f%%" % accuracy)
