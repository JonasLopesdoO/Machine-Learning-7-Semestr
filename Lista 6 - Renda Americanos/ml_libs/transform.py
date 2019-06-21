import numpy as np

class Normalize:
    def fit_transform(X):
        X_norm = np.copy(X)
        n_cols = X.shape[1]
        for i in range(n_cols):
            X_norm[:, i] = (X[:, i] - np.min(X[:, i])) / (np.max(X[:, i]) - np.min(X[:, i]))
        return X_norm;
    
class Standardize:
    def fit_transform(X):
        X_std = np.copy(X)
        n_cols = X.shape[1]
        for i in range(n_cols):
            X_std[:, i] = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])
        return X_std