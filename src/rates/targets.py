import numpy as np

from .auxillary import dist_computer


def monomial(x, legendre=True):
    if legendre:
        return (63 / 70) * x[:, 0] ** 5 - x[:, 0] ** 3 + (15 / 70) * x[:, 0]
    else:
        return x[:, 0].copy()


def Fourier_monomial(x, omega=1):
    return np.cos(2 * np.pi * omega * x[:, 0])


def clip_exp(x, clip=1):
    out = np.sum(x ** 2, axis=1)
    out[out > clip] = clip
    out *= -1
    np.exp(out, out=out)
    out -= np.exp(-clip)
    return out


def vanish_cos_2d(x):
    out = np.sum(x**2, axis=1)
    out *= -5
    np.exp(out, out=out)
    out *= np.sin(2 * np.pi * x[:, 0])
    out *= 5
    return out


def absolute_value(x):
    return np.sqrt(np.sum(x**2, axis=1))


def step_2d(x):
    return np.sign(x[:, 0])


def noisy_classif(x, p):
    out = np.abs(x)
    out **= 1 / p
    out *= np.sign(x)
    return out


def noisy_classif_modif(x, p, q, delta):
    out = np.abs(x)
    ind = out > delta
    out[ind] **= 1 / p
    np.invert(ind, out=ind)
    out[ind] **= 1 / q
    out[ind] *= delta ** (1 / p - 1 / q)
    out *= np.sign(x)
    return out


def two_cos(x, omega1=2, omega2=20):
    out = np.cos(np.pi * omega1 * x)
    if omega2 is not None:
        out += .5 * np.cos(np.pi * omega2 * x)
    out = np.sum(out, axis=1)
    return out


class HalfmoonFourClass:
    def __init__(self, n=10, sigma=8):
        self.x_set = self.init_x_set(n)
        self.sigma = sigma

    @staticmethod
    def init_x_set(n):
        theta = np.linspace(0, 1, num=n)
        theta *= np.pi / 2
        x_set = np.empty((4, n, 2), dtype=float)
        for i in range(4):
            x_set[i, :, 0] = np.cos(theta + i * np.pi / 2)
            x_set[i, :, 1] = np.sin(theta + i * np.pi / 2)
        x_set[0, :, 1] += 1
        x_set[-1, :, 1] += 1
        return x_set

    def __call__(self, x):
        dist = dist_computer(self.x_set.reshape(-1, 2), x).reshape(4, -1, len(x))
        out = dist.min(axis=1).T
        out *= - self.sigma
        np.exp(out, out=out)
        out /= out.sum(axis=1)[:, np.newaxis]
        return out


halfmoon_prob = HalfmoonFourClass()
