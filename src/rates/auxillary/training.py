import numpy as np

from .linalg import column_updated_inverse


def one_trial(K_reg, y_train, K_test, y_test, clip=None, weights=None):
    n_train = K_reg.shape[0]
    error = np.empty(n_train - 1)

    K_inv = np.linalg.inv(K_reg[0:1, 0:1])
    for i in range(1, n_train):
        # recursive inversion
        x = K_reg[:i, i]
        b = K_reg[i, i]
        K_inv = column_updated_inverse(K_inv, x, b)

        # prediction
        alpha = K_inv @ y_train[: i + 1]
        y_pred = K_test[:, : i + 1] @ alpha
        if clip is not None:
            np.clip(y_pred, clip[0], clip[1], out=y_pred)
        y_pred -= y_test
        y_pred **= 2

        # error
        if weights is not None:
            y_pred *= weights

        error[i - 1] = np.mean(y_pred)

    if weights is not None:
        error /= np.mean(weights)
    return error
