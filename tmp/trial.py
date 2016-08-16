from __future__ import absolute_import, print_function
import numpy as np

data = np.load('./tmp/testTrainData.npz')
print(data['XTrain'])
