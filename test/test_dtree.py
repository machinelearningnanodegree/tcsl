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
    clf = DecisionTreeClassifier(random_state=42)
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
    XTrain, XTest, yTest, yTrain = (data[item] for item in data.keys())
    dtree_output = baseline(XTrain, XTest, yTrain, yTest)
    assert(np.allclose(dtree_output[0],0.81349,atol=1e-5))
    assert(np.allclose(dtree_output[1],
            np.array([ 0.18321826,  0.05276996,  0.14053585,  0.16994514,  0.0759449 ,
                       0.02099475,  0.01722604,  0.17374549,  0.04401084,  0.09797639,
                       0.02363239]), atol=0.01))
    k_best_output = selectFeatures(5, XTrain, yTrain)
    assert(k_best_output.shape == (22792, 5))
    #assert(np.allclose())
