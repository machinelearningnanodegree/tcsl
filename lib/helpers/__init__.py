import datetime
import time
import os
import pandas as pd

from lib.model import Model
from lib.model.classifiers import CLASSIFIERS

def runTests(X_test, X_train, y_test, y_train):
    classifiers = []
    train_times = []
    gs_times = []
    pred_times = []
    f1_trains = []
    f1_tests = []

    for classifier, parameters, gs_params in CLASSIFIERS:
        print (classifier, parameters, gs_params)
        this_model = Model(classifier, parameters, gs_params)

        this_model(X_train, y_train, X_test, y_test)

        # Append vanilla values
        classifiers.append(this_model.classifier.__name__)
        train_times.append(this_model.training_time)
        pred_times.append(this_model.train_prediction_time)
        f1_trains.append(this_model.f1_train)
        f1_tests.append(this_model.f1_test)
        gs_times.append(0)

        # Append optimized classifier values
        classifiers.append(this_model.classifier.__name__ + ' (Optimized)')
        train_times.append(this_model.optimal_training_time)
        pred_times.append(this_model.optimal_train_prediction_time)
        f1_trains.append(this_model.f1_optimal_train)
        f1_tests.append(this_model.f1_optimal_test)
        gs_times.append(this_model.gs_time)


    return classifiers, train_times, pred_times, f1_trains, f1_tests, gs_times

def writeToCsv(classifiers, train_times, pred_times, f1_trains, f1_tests, gs_times):
    df = pd.DataFrame({"Classifier": classifiers,
                       "Training Time": train_times,
                       "Prediction Time": pred_times,
                       "F1 Score on Training Set": f1_trains,
                       "F1 Score on Test Set": f1_tests,
                       "Grid Search Time":gs_times})

    t = datetime.datetime(2011, 10, 21, 0, 0)
    t = time.mktime(t.timetuple())
    if not os.path.exists('results'):
        os.makedirs('results')
    filename = str(t)+'.csv'
    df.to_csv('results/'+filename)
