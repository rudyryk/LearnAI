import numpy as np

def crossentropy(scores, labels):
    return -np.sum(labels * np.log(scores))
