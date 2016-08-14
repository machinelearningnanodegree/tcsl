# Import libraries
import numpy as np
import pandas as pd
import os
import time
from matplotlib import pyplot as plt
from sklearn import grid_search
from sklearn.metrics import f1_score
from sklearn import tree, svm, naive_bayes, ensemble, neighbors
from sklearn.cross_validation import train_test_split
import datetime


# Read student data
student_data = pd.read_csv("student-data.csv")
student_data.reindex(np.random.permutation(student_data.index))
print "Student data read successfully!"

dtc = tree.DecisionTreeClassifier()
svc = svm.SVC()
nbc = naive_bayes.GaussianNB()
knn = neighbors.KNeighborsClassifier()
rfc = ensemble.RandomForestClassifier()
adc = ensemble.AdaBoostClassifier()

models = [dtc,svc,nbc,knn,rfc,adc]

# Extract feature (X) and target (y) columns
feature_cols = list(student_data.columns[:-1])  # all columns but last are features
target_col = student_data.columns[-1]  # last column is the target/label

X_all = student_data[feature_cols]  # feature values for all students
y_all = student_data[target_col]  # corresponding targets/labels

# Preprocess feature columns
def preprocess_features(X):
    outX = pd.DataFrame(index=X.index)  # output dataframe, initially empty

    # Check each column
    for col, col_data in X.iteritems():
        # If data type is non-numeric, try to replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])
        # Note: This should change the data type for yes/no columns to int

        # If still non-numeric, convert to one or more dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix=col)  # e.g. 'school' => 'school_GP', 'school_MS'

        outX = outX.join(col_data)  # collect column(s) in output dataframe

    return outX

X_all = preprocess_features(X_all)
y_all = y_all == "yes"
y_all = y_all.astype(int)

def train_split(num_train):
    num_all = student_data.shape[0]  # same as len(student_data)
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all,
                                                        test_size=95, train_size=num_train,
                                                        random_state=1791,stratify=y_all)
    return X_train,y_train,X_test,y_test


class Model:


    def __init__(self,classifier,parameters=[]):
        self.classifier = classifier
        self.parameters = parameters

    def train_classifier(self,clf, X_train, y_train):
        print "Training {}...".format(clf.__class__.__name__)
        #start = np.datetime64(datetime.datetime.now(),"us")
        start = time.time()
        clf.fit(X_train, y_train)
        #end = np.datetime64(datetime.datetime.now(),"us")
        end = time.time()
        self.training_time = end-start
        print self.training_time

    def predict_labels(self,clf, features, target):
        #print "Predicting labels using {}...".format(clf.__class__.__name__)
        #start = np.datetime64(datetime.datetime.now(),"us")
        start = time.time()
        y_pred = clf.predict(features)
        #end = np.datetime64(datetime.datetime.now(),"us")
        end = time.time()
        self.prediction_time = end - start
        f1_score_output = f1_score(target.values, y_pred)
        return f1_score_output

    def train_predict(self,clf, X_train, y_train, X_test, y_test):
        print "------------------------------------------"
        print "Training set size: {}".format(len(X_train))
        self.train_classifier(clf, X_train, y_train)
        self.f1_train = self.predict_labels(clf, X_train, y_train)
        self.f1_test = self.predict_labels(clf, X_test, y_test)
        return [self.training_time,self.prediction_time,self.f1_train,self.f1_test]

dataframes = []

for x in [100,200,300]:
    classifiers = []
    train_times = []
    pred_times = []
    f1_trains = []
    f1_tests = []

    for model in models:
        clf = Model(model)
        X_train,y_train,X_test,y_test = train_split(x)
        output = clf.train_predict(model,X_train,y_train,X_test,y_test)
        classifiers.append(model.__class__.__name__)
        train_times.append(str(output[0]))
        pred_times.append(str(output[1]))
        f1_trains.append(output[2])
        f1_tests.append(output[3])

    df = pd.DataFrame({"Classifier":classifiers,
                       "Training Time":train_times,
                       "Prediction Time":pred_times,
                       "F1 Score on Training Set":f1_trains,
                       "F1 Score on Test Set":f1_tests})
    dataframes.append(df)

for frame in dataframes:
    filenumber = 100
    filename = "{} samples.csv".format(filenumber)
    while os.path.isfile(filename):
        filenumber+=100
        filename = "{} samples.csv".format(filenumber)
    frame.to_csv(filename)

X_train,y_train,X_test,y_test = train_split(300)

#def fit_model(clf,parameters,X,Y):
#    clfr = grid_search.GridSearchCV(clf,parameters,scoring="f1",cv=4)
#    return clfr.fit(X,Y)

#clf = fit_model(svc,
#                [{"kernel":["poly"],
#                  "degree":[1,2,3,4,5],
#                  "C":[1,10,100,1000],
#                  }],X_train,y_train)

#print clf.best_params_
#print predict_labels(clf,X_test,y_test)
