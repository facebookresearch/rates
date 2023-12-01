
# Import
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh, pinvh

from rates.targets import halfmoon_prob
from rates.auxillary import generate_Y, zero_one_loss, write_numpy_file
from rates.datasets import halfmoon, noise_augmentations
from rates.architectures import exp_kernel


# Config
np.random.seed(0)
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{mathtools}')
plt.rc('font', size=10, family='serif', serif='cm')


sigma = .2
lambd = 1e-3

file_path_cl = "/private/vivc/cl_data.npy"
file_path_reg = "/private/vivc/reg_data.npy"

# Downstream performance
num_n_train = 5
min_n_train = 30
max_n_train = 1000
nb_down_trials = 10

num_n_up = 5
min_n_up = 30
max_n_up = 1000
nb_up_trials = 10

n_test = 500

ns_up = np.logspace(np.log10(min_n_up), np.log10(max_n_up), num=num_n_up).astype(int)
nu_max = ns_up[-1]
ns_down = np.logspace(np.log10(min_n_train), np.log10(max_n_train), num=num_n_train).astype(int)
nd_max = ns_down[-1]

T = np.kron(np.eye(nu_max), np.ones((2, 2)))

# error_reg = np.zeros((nb_up_trials, num_n_up, nb_down_trials, num_n_train))
# error_cl = np.zeros((nb_up_trials, num_n_up, nb_down_trials, num_n_train))
error_reg = np.zeros(num_n_train)
error_cl = np.zeros(num_n_train)

for j_up in range(nb_up_trials):
    xi = noise_augmentations(halfmoon(nu_max))
    x_test = halfmoon(n_test)
    K_test = exp_kernel(x_test, xi, sigma=sigma)
    p_test = halfmoon_prob(x_test)
    p_test_max = p_test.max(axis=1)

    K_up = exp_kernel(xi, xi, sigma=sigma)

    for i_up, n_up in enumerate(ns_up):
        K_inv = pinvh(K_up[:2 * n_up, :2 * n_up], atol=1e-5 / n_up)
        T_l = T[:2 * n_up, :2 * n_up] - lambd * K_inv
        w, v = eigh(T_l, subset_by_index=[len(T_l) - 5, len(T_l) - 1])

        # Test functions
        phi_test = K_test[:, :2 * n_up] @ (K_inv @ v)

        def zero_one(y_cat):
            return zero_one_loss(y_cat, p_test, prob_max=p_test_max)

        def l2(p):
            return np.sum((p - p_test) ** 2) / len(p)

        # Downstream performance

        for j in range(nb_down_trials):
            x_train = halfmoon(nd_max)
            y_train = generate_Y(halfmoon_prob(x_train))
            K_train = exp_kernel(x_train, xi[:2 * n_up], sigma=sigma)
            phi_train = K_train @ (K_inv @ v)

            for i, n_train in enumerate(ns_down):
                phi = phi_train[:n_train]

                # Linear probing on downstream
                w = np.linalg.solve(phi.T @ phi, phi.T @ y_train[:n_train])
                pred = phi_test @ w

                # Performance metrics
                # error_reg[j_up, i_up, j, i] = l2(pred)
                # error_cl[j_up, i_up, j, i] = zero_one(pred.argmax(axis=1))
                error_reg[i] = l2(pred)
                error_cl[i] = zero_one(pred.argmax(axis=1))

            write_numpy_file(error_reg, file_path_reg)
            write_numpy_file(error_cl, file_path_cl)
