"""
usage: dataWrangler.py [-h] [-i RAWDATALOC] [-s CV] [-r RS] [-o TEMPLOC]

DataWrangling Script - reads rawData from the /data folder and creates
features-labels(Test Train Split) and store it in /tmp folder

optional arguments:
  -h, --help     show this help message and exit
  -i RAWDATALOC  rawData file location <use absolute file path>.
                 default_value: ./data/adult.data
  -s CV          size of the cross_validation set. default_value: 0.30
  -r RS          random_state to use for splitting the data. default_value: 42
  -o TEMPLOC     file location to store temporary binary data for
                 test_train_split <use absolute file path>. default_value:
                 ./tmp/
"""

from __future__ import print_function, absolute_import
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import argparse


def getData(fileLoc=None):
    """
    # Fetch the training data from the CSV file and map the columns to the data
    # Input: FileLoc: File location of the dataset
    # returns: DataFrame containing input data
    """
    if fileLoc is None:
        fileLoc = "./data/adult.data"
    headers = ["age", "workclass", "fnlwgt", "education", "education-num",
               "marital-status", "occupation", "relationship", "race", "sex",
               "capital-gain", "capital-loss", "hours-per-week",
               "native-country", "income"
               ]
    return pd.read_csv(fileLoc, names=headers)


def getMap(series):
    """
    # Utility function Create a dictionary mapping the string features to ints.
    # Input: series: a numpy array containg the stirng data.
    # returns: A dictionary mapping unique stirngs to integers.
    """
    return {k: v for v, k in enumerate(np.unique(series))}


def featureMap(df, strcols):
    """
    # maps the lables to boolean values.
    # Input: df : DataFrame containing the training Dataset,
    #       strcols: list of cloumns containting string categories.
    # returns: DataFrame with categorical representations of strings features.
    """
    for col in strcols:
        m = getMap(df[col])
        df[col] = df[col].map(m)
        df[col].astype('category')
    return df


def fetchData(fileLoc=None):
    """
    # Utility function to fetch and process the data for feature-engineering.
    # Input: None
    # Output: DataFrame containing the all features needed for baselining\
the task.
    """
    rawData = getData(fileLoc)
    strcols = ['workclass', 'marital-status', 'occupation',
               'race', 'sex', 'native-country', 'income']
    trialData = featureMap(rawData, strcols)
    return trialData.drop(['fnlwgt', 'education', 'relationship'], axis=1)


def trainCvSplit(df, cvSize=0.30, rs=21):
    """
    # Simple Train and CV split on the dataset
    # Input: df: Transformed DataFrame of the Adult Dataset,
    #        cvSize: size of the cross_validation set,
    #        rs: random_state used for the CV split. (helps reproduce results)
    # returns: Tuple of Four np.arrays - (XTrain, XTest, yTrain, yTest).
    """
    labels = df['income'].values
    features = df.drop(['income'], axis=1).values
    kwargs = {'test_size': cvSize, 'random_state': rs}
    return train_test_split(features, labels, **kwargs)


def storeData(df, fileLoc='./tmp/', cv=0.30, rs=21):
    """
    # Store the train and CV data in the tmp location for the classifiers.
    # Input: df: Transformed DataFrame of the Adult Dataset.
    #        fileLoc: location of tmp where the binary data will be stored.
    #        cv: ratio of the cross_validation set in train-cv split
    #        rs: random_state used to the split.
    # returns: None
    # Note: data can be accessed using:
    #       Ex: data = np.load('./tmp/testTrainData.npz')
    #       and access the train/test using split using dictionary formatting.
    #       Ex: data['XTrain']
    """
    filename = fileLoc+'testTrainData'
    XTrain, XTest, yTrain, yTest = trainCvSplit(df, cv, rs)
    kwargs = {'XTrain': XTrain,
              'XTest': XTest,
              'yTrain': yTrain,
              'yTest': yTest
              }
    np.savez_compressed(filename, **kwargs)
    return None


def getCliArgs():
    """
    # Utility function to get the Command Line arguments from the user.
    # Inputs: None
    # returns: Command Line Arguments parser object with the following\
attributes:
    #       rawDataLoc: Location of Raw Input Data.
    #       cv: Ratio of the cv set in train-Cv split.
    #       rs: random_state to use while train-Cv split.
    #       tempLoc: location of tmp where the binary data will be stored.
    """
    desc = ("DataWrangling Script - reads rawData from the /data folder and \
            creates features-labels(Test Train Split) and store it \
             in /tmp folder")
    argpar = argparse.ArgumentParser(description=desc)
    argpar.add_argument('-i', action='store', dest='rawDataLoc',
                        default='./data/adult.data',
                        help='rawData file location <use absolute file path>.\
                        default_value: ./data/adult.data')
    argpar.add_argument('-s', action='store', dest='cv', type=float,
                        default=0.30, help='size of the cross_validation set.\
                        default_value: 0.30')
    argpar.add_argument('-r', action='store', dest='rs', type=int, default=42,
                        help='random_state to use for splitting the data.\
                        default_value: 42')
    argpar.add_argument('-o', action='store', dest='tempLoc', default='./tmp/',
                        help='file location to store temporary binary data for\
                         test_train_split <use absolute file path>.\
                          default_value: ./tmp/')
    return argpar.parse_args()

if __name__ == "__main__":
    args = getCliArgs()
    data = fetchData(args.rawDataLoc)
    storeData(data, fileLoc=args.tempLoc, cv=args.cv, rs=args.rs)
