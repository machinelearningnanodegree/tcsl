from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest
import numpy as np


def baseline(*args):
    """
    # Conduct a baseline classification on the dataset using the CV Set
    # Inputs: args=(XTrain, XTest, yTrain, yTest)
    # returns: the classification accuracy_score
    """
    XTrain, XTest, yTrain, yTest = args
    clf = DecisionTreeClassifier()
    clf.fit(XTrain, yTrain)
    return clf.score(XTest, yTest), clf.feature_importances_


def selectFeatures(k_features=5, *args):
    """
    # Select k best features using the SelectKBest class in Sklearn.
    # Inputs: k=no. of features to select, args=(XTrain,yTrain)
    # returns: np array of k features.
    """
    X, y = args
    skb = SelectKBest(k=k_features)
    return skb.fit_transform(X, y)


if __name__ == "__main__":
    data = np.load('./tmp/testTrainData.npz')
    XTrain, XTest, yTrain, yTest = (data[item] for item in data.keys())
    print(baseline(XTrain, XTest, yTrain, yTest))
    print(selectFeatures(5, XTrain, yTrain))
