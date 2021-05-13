import numpy as np

def dummie2multilabel(X):
    """Convert dummies to multilabel"""
    N = len(X)
    X_multi = np.zeros((N,1),dtype='int')
    for i in range(N):
        temp = np.where(X[i]==1)[0] # find 1 in the array
        if temp.size == 0: # is a empty array, there is no '1' in the X[i] array
            X_multi[i] = 0 # so we denote this class '0'
        else:
            X_multi[i] = temp[0] + 1
    return X_multi.T[0]