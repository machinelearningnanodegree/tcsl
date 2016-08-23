# Import libraries
import numpy as np
import pandas as pd

from lib.data.wrangler import readData
from lib.helpers import runTests, writeToCsv

X_train, \
    X_test, \
    y_test, \
    y_train = readData('./tmp/testTrainData.npz')

classifiers, \
    train_times, \
    pred_times, \
    f1_trains, \
    f1_tests, \
    gs_times = runTests(X_test, X_train, y_test, y_train)

writeToCsv(classifiers, train_times, pred_times, f1_trains, f1_tests, gs_times)
