def accuracy(y, y_pred):
    return np.sum(y.values.transpose()[0] == y_pred) / y_pred.shape[0]