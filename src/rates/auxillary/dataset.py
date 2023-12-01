
import numpy as np


def generate_Y(prob, one_hot=True):
    y_cat = (prob.cumsum(axis=1) < np.random.rand(len(prob))[:, np.newaxis]).sum(axis=1)
    if one_hot:
        return cat2one_hot(y_cat, prob.shape[1])
    return y_cat


def cat2one_hot(y, m=None):
    if m is None:
        m = y.max() + 1
    one_hot = np.zeros((len(y), m))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot


def one_hot2cat(y):
    return y.argmax(axis=1)
