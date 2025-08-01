import numpy as np

def cross_entropy(predictions, targets):
    predictions = np.clip(predictions, 1e-12, 1. - 1e-12)
    return -np.sum(targets * np.log(predictions))