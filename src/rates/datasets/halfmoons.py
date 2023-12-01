
import numpy as np


# Dataset
def halfmoon(n, noise_level=.1):
    theta = np.random.rand(n)
    theta *= 2 * np.pi
    x = np.empty((n, 2), dtype=float)
    x[:, 0] = np.cos(theta)
    x[:, 1] = np.sin(theta)
    x[x[:, 0] > 0, 1] += 1
    x += noise_level * np.random.randn(n, 2)
    return x
