
import numpy as np
from rates.auxillary import cat2one_hot


def mixture(n, d=4, noise_level=.1, labels=True, test=False):
    if test:
        y = np.floor(np.arange(n) * d / n).astype(int)
    else:
        y = np.random.randint(0, d, n)
    x = cat2one_hot(y, d)
    x += noise_level * np.random.randn(n, d)
    if labels:
        return x, y + 1
    return x
