# Import libraries
import numpy as np
import pandas as pd
from sklearn import grid_search
from sklearn.cross_validation import train_test_split

from lib.classifiers import CLASSIFIERS
from lib.model import Model

data = np.load('./tmp/testTrainData.npz')
print(data.keys())
X_train, X_test, y_test, y_train = (data[item] for item in data.keys())

classifiers = []
dataframes = []
train_times = []
pred_times = []
f1_trains = []
f1_tests = []

for classifier, parameters in CLASSIFIERS:
    this_model = Model(classifier, parameters)

    this_model(X_train, y_train, X_test, y_test)
    classifiers.append(this_model.classifier.__class__.__name__)
    train_times.append(this_model.training_time)
    pred_times.append(this_model.train_prediction_time)
    f1_trains.append(this_model.f1_train)
    f1_tests.append(this_model.f1_test)

df = pd.DataFrame({"Classifier": classifiers,
                   "Training Time": train_times,
                   "Prediction Time": pred_times,
                   "F1 Score on Training Set": f1_trains,
                   "F1 Score on Test Set": f1_tests})
dataframes.append(df)

for i, frame in enumerate(dataframes):
    filenumber = i * 100
    filename = "results/{} samples.csv".format(filenumber)
    frame.to_csv(filename)


# def fit_model(clf,parameters,X,Y):
#    clfr = grid_search.GridSearchCV(clf,parameters,scoring="f1",cv=4)
#    return clfr.fit(X,Y)

# clf = fit_model(svc,
#                [{"kernel":["poly"],
#                  "degree":[1,2,3,4,5],
#                  "C":[1,10,100,1000],
#                  }],X_train,y_train)

# print(clf.best_params_)
 # print(predict_labels(clf,X_test,y_test))
