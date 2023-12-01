
import numpy as np


# Dataset
def circles(n, d=4, noise_level=.1, labels=True, test=False):
    if test:
        r = np.floor(np.arange(n) * d / n)
        theta = np.arange(n) * d / n
    else:
        r = np.random.choice(d, n)
        theta = np.random.rand(n)
    r += 1
    theta *= 2 * np.pi
    x = np.empty((n, 2), dtype=float)
    x[:, 0] = np.cos(theta)
    x[:, 1] = np.sin(theta)
    x *= r[:, np.newaxis]
    x += noise_level * np.random.randn(n, 2)
    if labels:
        return x, r
    return x
