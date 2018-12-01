import numpy as np

def normalize(data):
    return np.divide(data, 255.0)

def denormalize(data):
    return np.around(np.multiply(data, 255.0))