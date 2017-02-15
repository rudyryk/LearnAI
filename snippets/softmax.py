import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    z = np.exp(x)
    return z / np.sum(z, axis=0)
