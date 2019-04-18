import numpy as np

def split_stratified_train_test(y, perc_train, seed):
    r = np.random.RandomState(seed)
    perm_index = r.permutation(y.shape[0])
    
    n_train = int(np.ceil(y.shape[0] * perc_train))
    
    return perm_index[0:n_train], perm_index[n_train+1:]