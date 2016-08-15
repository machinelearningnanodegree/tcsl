from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest
from dataWrangler import fetchData


def trainCvSplit(df):
    # Simple Train and CV split on the dataset
    # Input: Transformed DataFrame of the Adult Dataset
    # returns: 4 np.array -[XTrain, XTest, yTrain, yTest]
    labels = df['income'].values
    features = df.drop(['income'], axis=1).values
    return train_test_split(features, labels, test_size=0.30, random_state=24)


def baseline(*args):
    # Conduct a baseline classification on the dataset using the CV Set
    # Inputs: args=(XTrain, XTest, yTrain, yTest)
    # returns: the classification accuracy_score
    XTrain, XTest, yTrain, yTest = args
    clf = DecisionTreeClassifier()
    clf.fit(XTrain, yTrain)
    return clf.score(XTest, yTest), clf.feature_importances_


def selectFeatures(k_features=5, *args):
    # Select k best features using the SelectKBest class in Sklearn.
    # Inputs: k=no. of features to select, args=(XTrain,yTrain)
    # returns: np array of k features.
    X, y = args
    skb = SelectKBest(k=k_features)
    return skb.fit_transform(X, y)


if __name__ == "__main__":
    data = fetchData()
    XTrain, XTest, yTrain, yTest = trainCvSplit(data)
    print(baseline(XTrain, XTest, yTrain, yTest))
    print(selectFeatures(5, XTrain, yTrain))
