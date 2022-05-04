import pandas as pd

"""
יבגני גאיסינסקי 336551072
שניר בן יוסף 307908699
"""


def naive(df,dftest):
    prior = df.groupby('class').size().div(len(df))  # class/len(df)
    likelihood = {}
    testlist = []
    featurelist = []
    numeric = []
    ###pre proccessing##
    for col in df:
        if (type(df[col][0]) != str):
            numeric.append(col)  # numeric


        if "class" not in col:
            featurelist.append(col)  # feture without class
            likelihood[col] = df.groupby(['class', col]).size().div(len(df)).div(prior)  ## Creating Likelihhod##

    for row in dftest.iterrows():
        p_yes = 1
        p_no = 1
        for key, val in row[1].items():
            if "class" not in key:
                if val in likelihood[key]['yes'].index and val in likelihood[key]['no'].index:
                    p_yes *= likelihood[key]['yes'].loc[val]
                    p_no *= likelihood[key]['no'].loc[val]
        p_yes *= prior['yes']
        p_no *= prior['no']
        if p_yes > p_no:
            testlist.append("yes")
        else:
            testlist.append("no")

    col_temp = {'test': testlist}
    final = pd.DataFrame(col_temp)
    final['train'] = df['class']
    myresult = final.groupby(["test", "train"]).size()
    success = myresult['no']['no'] + myresult['yes']['yes']
    counttest = len(final)
    successrate = (success / counttest) * 100
    print("Success is: %.2f%%" % successrate)

