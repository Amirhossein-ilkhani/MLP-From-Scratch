import numpy as np


def MSE(y_true, y_pred):
    return np.square(np.subtract(y_true, y_pred))/2

def cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred))