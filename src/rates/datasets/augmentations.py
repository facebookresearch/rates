
import numpy as np


# Augmentation
def noise_augmentations(x, y=None, m=2, eps=.1):
    xi = np.hstack((x,) * m).reshape(-1, x.shape[1])
    xi += eps * np.random.randn(*xi.shape)
    if y is None:
        return xi

    if len(y.shape) == 1:
        y = y[:, np.newaxis]
    y = np.hstack((y,) * m).reshape(-1, y.shape[1]).squeeze()
    return xi, y
