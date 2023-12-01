
import numpy as np


def zero_one_loss(y_cat, prob, prob_max=None):
    if prob_max is None:
        prob_max = prob.max(axis=1)
    error = np.choose(y_cat, prob.T)
    error -= prob_max
    return - np.mean(error)
