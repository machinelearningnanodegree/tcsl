from __future__ import print_function, absolute_import
import pandas as pd
import numpy as np


def getData(fileLoc=None):
    # Fetch the training data from the CSV file and map the columns to the data
    # Input: FileLoc = File location of the dataset
    # returns: DataFrame containing input data
    if fileLoc is None:
        fileLoc = "./data/adult.data"
    headers = ["age", "workclass", "fnlwgt", "education", "education-num",
               "marital-status", "occupation", "relationship", "race", "sex",
               "capital-gain", "capital-loss", "hours-per-week",
               "native-country", "income"
               ]
    return pd.read_csv(fileLoc, names=headers)


def getMap(series):
    # Create a dictionary mapping the string features to unique integers.
    return {k: v for v, k in enumerate(np.unique(series))}


def featureMap(df, strcols):
    # maps the lables to boolean values.
    # Input: df : DataFrame containing the training Dataset,
    #       strcols: list of cloumns containting string categories.
    # returns: df: with categorical representations of strings features.
    for col in strcols:
        m = getMap(df[col])
        df[col] = df[col].map(m)
        df[col].astype('category')
    return df


def fetchData():
    # Utility function to fetch and process the data for feature-engineering.
    # Input: None
    # Output: df containing the all features needed for baselining the task.
    rawData = getData()
    strcols = ['workclass', 'marital-status', 'occupation',
               'race', 'sex', 'native-country', 'income']
    trialData = featureMap(rawData, strcols)
    return trialData.drop(['fnlwgt', 'education', 'relationship'], axis=1)


if __name__ == "__main__":
    print(fetchData().head())
