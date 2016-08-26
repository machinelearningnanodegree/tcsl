

from __future__ import print_function, absolute_import
import os
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import RandomizedLogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest

import warnings
warnings.filterwarnings("ignore")


def readData(fileLoc=None):
    data = np.load(fileLoc)
    print(data.keys())
    return (data[item] for item in data.keys())


class PreProcess(object):

    def __init__(self, infile=None):
        self.infile = infile
        self.load_rawdata_file()

    def load_rawdata_file(self):
        """
        Fetch data from a CSV file and map the columns
        to the data to form a DataFrame.

        Input:
        FileLoc: File location of the dataset

        Returns:
        DataFrame containing input data
        """
        if not self.infile:
            # Look for file in the default location.
            self.infile = "{}/data/adult.data".format(os.getcwd())

        # The input file by default contains no headers.
        # Creating a list to map the fields to the field names.
        headers = ["age", "workclass", "fnlwgt", "education",
                   "education-num", "marital-status", "occupation",
                   "relationship", "race", "sex", "capital-gain",
                   "capital-loss", "hours-per-week", "native-country",
                   "income"
                   ]
        # Read the data into a DataFrame.
        self.rawdata = pd.read_csv(self.infile, names=headers)
        return self.rawdata

    def __getraw__(self):
        return self.rawdata

    def __getprep__(self):
        return self.convert_data()

    def _get_map(self, series):
        """
        Utility function creates a dictionary mapping the
        string features to categorical integers.

        Input:
        series: a numpy array containg the stirng data.

        Returns:
        A dictionary mapping unique stirngs to integers.
        """
        return {k: v for v, k in enumerate(np.unique(series))}

    def _store_map(self, col, map):
        self.col_map[col] = map

    def _get_strcols(self):
        """
        Utility function to get column names of string features.

        Input:
        rawdata: Dataframe with data read from file.

        Output:
        Index of string columns in the dataframe.
        """
        return self.prepdata.select_dtypes(include=[object]).columns

    def remove_non_features(self):
        removecols = ['fnlwgt', 'education', 'relationship']
        return self.rawdata.drop(removecols, axis=1)

    def feature_map(self):
        """
        Creates raw integer categorical features from str categories.

        Input:
        rawdata : DataFrame containing the training Dataset.
        strcols: Index of string columns in the dataframe.

        Returns:
        DataFrame with integer categories of strings features.
        """
        self.col_map = {}
        for col in self._get_strcols():
            m = self._get_map(self.prepdata[col])
            self._store_map(col, m)
            self.prepdata[col] = self.prepdata[col].map(m)
            self.prepdata[col].astype('category')
        return self.prepdata

    def convert_data(self,
                     removenonfeatures=True,
                     convertstrs=True):
        """
        Wrapper function loads the rawdata from file,
        remove unnecessary columns,
        converts string features to categories
        for feature-engineering.

        Input:
        None

        Output:
        DataFrame containing the all features needed for baselining
        the task.
        """
        if removenonfeatures:
            self.prepdata = self.remove_non_features()
        if convertstrs:
            self.feature_map()
        return self.prepdata


class FeatureSelector(object):
    def __init__(self, prepdata):
        self.prepdata = prepdata
        self.get_splits()

    def _array_to_df(self, features, colnames):
        return pd.DataFrame(features, columns=colnames)

    def get_splits(self):
        XTrain, XTest, yTrain, yTest = trainCvSplit(self.prepdata)
        columns = self.prepdata.columns.tolist()
        feature_cols = columns[:-1]
        label_cols = columns[-1:]
        self.features, self.labels = (self._array_to_df(XTrain, feature_cols),
                                      self._array_to_df(yTrain, label_cols))
        self.testfeatures, self.testlabels = (self._array_to_df(XTest,
                                                                feature_cols),
                                              self._array_to_df(yTest,
                                                                label_cols)
                                              )

    def k_best_features(self):
        # get total number of features.
        num_features = self.features.shape[1]
        feature_list = []
        # find k-best features, with k from 1 to num_features.
        for i in range(num_features):
            skBest = SelectKBest(k=i)
            skBest.fit_transform(self.features, self.labels)
            # get boolean indices of the best features.
            k_features = skBest.get_support()
            # append the features to the feature list.
            feature_list += self.features.columns[k_features].tolist()
        return feature_list

    def k_best_index(self):
        k_best_score = defaultdict(int)
        # get the features_list.
        k_features = self.k_best_features()
        # normaliztion factor for calculating the k-score.
        k_norm = float(len(k_features))
        # create a frequency histogram of the distribution.
        for item in k_features:
            k_best_score[item] += 1
        # create a probability mass function of the distribution.
        for key, value in k_best_score.iteritems():
            k_best_score[key] = value/k_norm
        return k_best_score

    def impurity_index(self, clf):
        # clf is any classifier that has the feature_importances_ attribute
        clf.fit(self.features, self.labels)
        # map feature_importances to the feature names
        impuritydict = {k: v for k, v in
                        zip(self.features.columns.tolist(),
                            map(lambda x: round(x, 4),
                                clf.feature_importances_
                                )
                            )
                        }
        return impuritydict

    def recurvise_index(self, clf,):
        # rank all features, i.e continue the elimination until the last one
        rfe = RFE(clf, n_features_to_select=1)
        rfe.fit(self.features, self.labels)
        # map recursive feature score to the feature names
        rfedict = {k: v for k, v in
                   zip(self.features.columns.tolist(),
                       map(lambda x: round(x, 4),
                           rfe.ranking_
                           )
                       )
                   }
        return rfedict

    def stabilty_index(self, clf):
        # sklearn implements stability selection in
        # RandomizedLogisticRegression class only
        clf.fit(self.features, self.labels)
        # map Feature scores between 0 and 1. to the feature names
        stabledict = {k: v for k, v in
                      zip(self.features.columns.tolist(),
                          map(lambda x: round(x, 4),
                              clf.scores_
                              )
                          )
                      }
        return stabledict

    def _get_clfs(self):
        clf_dict = {"rlrclf": RandomizedLogisticRegression(),
                    "rfclf": RandomForestClassifier(criterion='entropy'),
                    "dtrclf": DecisionTreeClassifier(criterion='entropy'),
                    "lrclf": LogisticRegression()
                    }
        return clf_dict

    def get_selection(self):
        clf_dict = self._get_clfs()
        summary_dict = {
            "k_best_score": self.k_best_index(),
            "dtrclf_impurity_score": self.impurity_index(clf_dict["dtrclf"]),
            "rfclf_impurity_score": self.impurity_index(clf_dict["rfclf"]),
            "lrclf_recursive_score": self.recurvise_index(clf_dict["lrclf"]),
            "rlrclf_stable_score": self.stabilty_index(clf_dict["rlrclf"])
            }
        return pd.DataFrame.from_dict(summary_dict)


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
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
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
    Utility function to get the Command Line arguments from the user.

    Inputs: None

    Returns:
    Command Line Arguments parser object with the following attributes:
        rawDataLoc: Location of Raw Input Data.
        v: Ratio of the cv set in train-Cv split.
        rs: random_state to use while train-Cv split.
        tempLoc: location of tmp where the binary data will be stored.
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
    # args = getCliArgs()r
    data = PreProcess()
    # storeData(data, fileLoc=args.tempLoc, cv=args.cv, rs=args.rs)
    selector = FeatureSelector(data)
    print(selector.get_selection())
