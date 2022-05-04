import pickle

import numpy as np
import pandas as pd

from numpy import log2 as log

"""
יבגני גאיסינסקי 336551072
שניר בן יוסף 307908699
"""


def ID3(df,dftest):
    ###pre proccessing##
    ##Delete Empty rows
    df = df.dropna(how='any', axis=0)
    eps = np.finfo(float).eps#epsilon
    def find_entropy(df):
        Class = df.keys()[-1]
        entropy = 0
        values = df[Class].unique()
        for value in values:
            fraction = df[Class].value_counts()[value] / len(df[Class])
            entropy += -fraction * np.log2(fraction)
        return entropy

    def find_entropy_attribute(df, attribute):
        Class = df.keys()[-1]
        target_variables = df[Class].unique()
        variables = df[attribute].unique()
        entropy2 = 0
        for variable in variables:
            entropy = 0
            for target_variable in target_variables:
                num = len(df[attribute][df[attribute] == variable][df[Class] == target_variable])
                den = len(df[attribute][df[attribute] == variable])
                fraction = num / (den + eps)
                entropy += -fraction * log(fraction + eps)
            fraction2 = den / len(df)
            entropy2 += -fraction2 * entropy
        return abs(entropy2)

    def find_winner(df): ## get the higer one
        Entropy_att = []
        IG = []
        for key in df.keys()[:-1]:
            IG.append(find_entropy(df) - find_entropy_attribute(df, key))
        return df.keys()[:-1][np.argmax(IG)]

    def get_subtable(df, node, value):
        return df[df[node] == value].reset_index(drop=True)

    def buildTree(df, tree=None):
        node = find_winner(df)

        attValue = pd.unique(df[node])

        if tree is None:
            tree = {}
            tree[node] = {}
        for value in attValue:
            subtable = get_subtable(df, node, value)
            clValue, counts = np.unique(subtable["class"], return_counts=True)
            if (len(df) < 5500):
                if len(counts) == 2:
                    if (counts[0] > counts[1]):
                        tree[node][value] = clValue[0]
                    else:
                        tree[node][value] = clValue[1]
                if len(counts) == 1:  # Checking purity of subset
                    tree[node][value] = clValue[0]
            else:
                tree[node][value] = buildTree(subtable)  # Calling the function recursively

        return tree

    t = buildTree(df)
    filename = 'final_id3(imp).sav'
    pickle.dump(t, open(filename, 'wb'))
    import pprint


    pprint.pprint(t)

