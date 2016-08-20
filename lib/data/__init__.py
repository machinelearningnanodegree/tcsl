import datetime
import time
import pandas as pd

from lib.model import Model
from lib.model.classifiers import CLASSIFIERS

def runTests(X_test, X_train, y_test, y_train):
    classifiers = []
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

    return classifiers, train_times, pred_times, f1_trains, f1_tests

def writeToCsv(classifiers, train_times, pred_times, f1_trains, f1_tests):
    df = pd.DataFrame({"Classifier": classifiers,
                       "Training Time": train_times,
                       "Prediction Time": pred_times,
                       "F1 Score on Training Set": f1_trains,
                       "F1 Score on Test Set": f1_tests})

    t = datetime.datetime(2011, 10, 21, 0, 0)
    t = time.mktime(t.timetuple())
    if not os.path.exists('results'):
        os.makedirs('results')
    filename = str(t)+'.csv'
    frame.to_csv('results/'+filename)
