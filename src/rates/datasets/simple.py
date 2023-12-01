import numpy as np


class GeneratorGaussian1d:
    def x_gen(n_train, *args):
        return np.random.randn(n_train, 1)

    def x_test_gen(n_test, *args):
        return np.linspace(-3, 3, n_test)[:, np.newaxis]

    def weights_gen(x_test):
        return np.exp(-x_test.squeeze() ** 2 / 2)


class GeneratorGaussian2d:
    def x_gen(n_train, *args):
        return np.random.randn(n_train, 2)

    def x_test_gen(n_test, *args):
        lin = np.linspace(-2, 2, np.sqrt(n_test).astype(int))
        X1, X2 = np.meshgrid(lin, lin)
        return np.vstack((X1.reshape(-1), X2.reshape(-1))).T

    def weights_gen(x_test):
        return np.exp(-np.sum(x_test**2, axis=1) / 2)


class GeneratorUniform:
    def x_gen(n_train, d):
        return 2 * np.random.rand(n_train, d) - 1

    def x_test_gen(n_test, d):
        return 2 * np.random.rand(n_test, d) - 1

    def weights_gen(x_test):
        return


class GeneratorUniform1d:
    def x_gen(n_train, *args):
        return 2 * np.random.rand(n_train, 1) - 1

    def x_test_gen(n_test, *args):
        return np.linspace(-1, 1, n_test)[:, np.newaxis]

    def weights_gen(x_test):
        return


class GeneratorUniform2d:
    def x_gen(n_train, *args):
        return 2 * np.random.rand(n_train, 2) - 1

    def x_test_gen(n_test, *args):
        lin = np.linspace(-1, 1, np.sqrt(n_test).astype(int))
        X1, X2 = np.meshgrid(lin, lin)
        return np.vstack((X1.reshape(-1), X2.reshape(-1))).T

    def weights_gen(x_test):
        return
