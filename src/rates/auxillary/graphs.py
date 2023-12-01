
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def connected_component(T, return_labels=False):
    return connected_components(csr_matrix(T), return_labels=return_labels)


def drop_entries(M, p):
    ind = np.triu(np.random.rand(*M.shape)) < p
    ind *= ind.T
    new_M = M.copy()
    new_M[ind] = 0
    return new_M


def label_graph(labels):
    T = labels[:, np.newaxis] @ labels[np.newaxis, :]
    T *= -2
    norm = labels ** 2
    T += norm[:, np.newaxis]
    T += norm[np.newaxis, :]
    T = (T == 0).astype(float)
    return T


class EntriesAdder:
    def __init__(self, T):
        self.T = T

    def __call__(self):
        new_T = np.eye(len(self.T), dtype=float)
        tmp = np.triu(self.T, k=1).flatten()
        ind = np.arange(len(tmp))[tmp.astype(bool)]
        np.random.shuffle(ind)

        count = 0
        while count < len(ind):
            yield new_T
            i, j = np.unravel_index(ind[count], self.T.shape)
            new_T[i, j] = self.T[i, j]
            new_T[j, i] = self.T[j, i]
            count += 1
        if count == len(ind):
            yield new_T
