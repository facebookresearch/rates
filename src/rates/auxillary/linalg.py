import numba
import numpy as np


# Distance computation
def dist_computer(x1, x2):
    out = x1 @ x2.T
    out *= -2
    out += np.sum(x1 ** 2, axis=1)[:, np.newaxis]
    out += np.sum(x2 ** 2, axis=1)
    return out


# Online matrix inversion
def column_updated_inverse(A1, x, b):
    out = np.empty((len(x) + 1, len(x) + 1), dtype=np.float64)
    y = np.dot(A1, x)
    out[:-1, :-1] = np.dot(y[:, np.newaxis], y[np.newaxis, :])
    out[:-1, -1] = -y
    out[-1, :-1] = -y
    out[-1, -1] = 1
    out /= b - np.dot(x, y)
    out[:-1, :-1] += A1
    return out


def recursive_inverse(A):
    if len(A) == 1:
        return A ** (-1)
    else:
        A1 = recursive_inverse(A[:-1, :-1])
        x = A[:-1, -1]
        b = A[1, 1]
        return column_updated_inverse(A1, x, b)


@numba.jit("(f8[:, :], f8[:, :], i8[:], i8[:])")
def copy_matrix(source, dest, index1, index2):
    for i in index1:
        for j in index2:
            dest[i, j] = source[i, j]
