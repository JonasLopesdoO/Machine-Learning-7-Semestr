import numpy as np

## Cálculo do RMSE
def rmse(y, y_pred):
    return mse(y, y_pred) ** 0.5
# This make the sqrt for mse

## Calculo do MSE
def mse(y, y_pred):
    return np.sum(( y - y_pred ) ** 2 ) / y.shape[0]