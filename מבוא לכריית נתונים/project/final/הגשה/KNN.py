from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn import preprocessing
import numpy
from sklearn.metrics import accuracy_score
"""
יבגני גאיסינסקי 336551072
שניר בן יוסף 307908699
"""

def KNN(df, dftest):
    # preprocessing
    le = preprocessing.LabelEncoder()
    ############################################################TRAIN#############################################################

    # transform class to list and converting class into numbers
    df['class'] = le.fit_transform(df['class'])
    train_label = df['class'].tolist()

    dftest['class'] = le.fit_transform(dftest['class'])
    test_label = dftest['class'].tolist()

    df.dropna(subset=['class'])  # Drop  rows
    dftest.dropna(subset=['class'])  # Drop  rows

    del df['class']
    del dftest['class']

    # converting string and bins into numbers
    for col in df:
        if type(df[col][0]) == numpy.int64:  # numeric diszrz
            df[col] = le.fit_transform(pd.qcut(df[col], duplicates="drop"))  # Equal frequency func
            dftest[col] = le.fit_transform(pd.qcut(dftest[col], duplicates="drop"))  # Equal frequency func
        else:
            df[col] = le.fit_transform(df[col].tolist())
            dftest[col] = le.fit_transform(dftest[col].tolist())

    ##################################################KNN################################################################

    # Create a KNN Classifier
    model = KNeighborsClassifier(n_neighbors=3)

    # Train the model using the training sets
    model.fit(df, train_label)  # train

    # Predict Output
    predicted = model.predict(dftest)

    accuracy = accuracy_score(test_label, predicted) * 100
    print("Success percent is: %.2f%%" % accuracy)
