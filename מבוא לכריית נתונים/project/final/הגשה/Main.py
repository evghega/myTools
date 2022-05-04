import numpy as np
import pandas as pd
from scipy.stats import entropy

import NaiveBayesLib as nb
import NaiveBayes as nbo
import KNN as kn
import Kmeans as km
import ID3 as id
import ID3Lib as idl

"""
יבגני גאיסינסקי 336551072
שניר בן יוסף 307908699
"""


def entropy_func(arr):
    x, counts = np.unique(arr, return_counts=True)
    return entropy(counts, base=2)


def main():
    ### Load Files ###
    pathoftrain = input("Enter the path to the train file: (without .csv): ")
    if not pathoftrain:
        pathoftrain = "train"
    train = pd.read_csv(pathoftrain + '.csv').applymap(lambda x: x.lower() if type(x) == str else x)

    pathofpredict = input("Enter the path to the predict file: (without .csv): ")
    if not pathofpredict:
        pathofpredict = "test"
    predict = pd.read_csv(pathofpredict + '.csv').applymap(lambda x: x.lower() if type(x) == str else x)

    ######  PreProcess Remove all Class that Blank  #####
    predict = predict[predict['class'].notna()].reset_index()  # remove rows that are null and
    # fixing the index of the table
    train = train[train['class'].notna()].reset_index()

    ## get Numeric and normal colums ##
    numeric = train.describe().columns
    normal = train.head().columns

    ##Numeric put mean in empty cells##
    for att in numeric:
        if (att != "index"):
            train[att] = train[str(att)].fillna(train[str(att)].mean())
            predict[att] = predict[att].fillna(predict[att].mean())

    ## replace all non numberic cells that empty with the most frequncy value
    for att in normal:
        if att not in numeric:
            train = train.fillna(train[att].value_counts().index[0])
            predict = predict.fillna(predict[att].value_counts().index[0])

    """בחירת סוג הדיסקרטיזציה"""
    optiondisc = input(
        "how would you like to do discretization?: \n\n"
        " 1. None \n2.Equal Frequnecy\n3.Equal width\n 4.Entropy Based\n")

    if (optiondisc == '2'):
        bins = int(input("how much bins ?"))
        for att in numeric:
            train[att] = pd.qcut(train[att], bins, duplicates="drop")

    if (optiondisc == '3'):
        bins = int(input("how much bins ?"))
        for att in numeric:
            train[att] = pd.qcut(train[att], bins, duplicates="drop")
    if (optiondisc == '4'):
        train = entropy_func(train)
    train.to_csv(r"train_clean.csv", index=False)
    predict.to_csv(r"predict_clean.csv", index=False)

    ##main menu##
    menu = input("\nChoose the algo for the model:\n\n1.naive bayes(implementaion)\n2.naive bayes\n"
                 "3.ID3(implementaion)\n4.ID3\n5.KNN\n6.Kmeans\n")

    if menu == '1':
        nbo.naive(train, predict)
    elif menu == '2':
        nb.naive(train, predict)
    elif menu == '3':
        id.ID3(train, predict)
    elif menu == '4':
        idl.ID3(train, predict)
    elif menu == '5':
        kn.KNN(train, predict)
    elif menu == '6':
        km.Kmeans(train, predict)


main()
