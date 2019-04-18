import numpy as np

## Accuracy calc
def accuracy(y_test, y_pred):
     return np.sum(y_test == y_pred) / y_pred.shape[0]
