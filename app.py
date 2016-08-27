# Import libraries
import numpy as np
import pandas as pd

from lib.data.wrangler import readData
from lib.helpers import runTests, writeToCsv

data = readData('./tmp/testTrainData.npz')
X_train = data['XTrain']
X_test = data['XTest']
y_train = data['yTrain']
y_test = data['yTest']

classifiers, \
    train_times, \
    pred_times, \
    f1_trains, \
    f1_tests, \
    gs_times = runTests(X_test, X_train, y_test, y_train)

writeToCsv(classifiers, train_times, pred_times, f1_trains, f1_tests, gs_times)
