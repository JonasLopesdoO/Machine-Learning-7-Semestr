import numpy as np 

def accuracy(y, y_hat):
    return np.sum(y.values.transpose()[0] == y_hat) / y_hat.shape[0]