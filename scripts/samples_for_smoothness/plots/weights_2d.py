import argparse

import matplotlib.pyplot as plt
import numpy as np

from rates.architectures import rbf_kernel, rbf_periodic
from rates.config import LOG_DIR


parser = argparse.ArgumentParser(
    prog="Plot setup",
    description="Setup visualization",
    epilog="@ 2022, Vivien Cabannes",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "-n",
    "--root_n",
    default=100,
    nargs="?",
    help="Root of the number of samples",
    type=int,
)
parser.add_argument(
    "-s",
    "--sigma",
    default=.001,
    nargs="?",
    help="Scale of regression kernel",
    type=float,
)
parser.add_argument(
    "-l",
    "--lambd",
    default=1e-5,
    nargs="?",
    help="Regularization parameter",
    type=float,
)
parser.add_argument(
    "-k",
    "--kernel",
    default="periodic",
    nargs="?",
    help="Kernel to compute weights",
    type=str,
)
config = parser.parse_args()

# testing set
lin = np.linspace(-1, 1, num=config.root_n)
X1, X2 = np.meshgrid(lin, lin)
x_test = np.vstack((X1.reshape(-1), X2.reshape(-1))).T

if config.kernel == "periodic":
    kernel = rbf_periodic
else:
    kernel = rbf_kernel

# weights computation
K = kernel(x_test, x_test, sigma=config.sigma)
K_test = kernel(x_test, np.array([.25, .25])[np.newaxis, :], sigma=config.sigma)
K += config.lambd * np.eye(K.shape[0])
w = np.linalg.solve(K, K_test.reshape(-1))
w /= np.mean(np.abs(w))

# visualization
fig, ax = plt.subplots(1, 1, figsize=(3, 3))
ax.contourf(
    X1,
    X2,
    w.reshape(config.root_n, config.root_n),
    cmap="RdBu",
    levels=20,
    vmin=-0,
    vmax=w.max(),
)
ax.set_axis_off()
fig.tight_layout()
fig.savefig(LOG_DIR / f"weight_{np.abs(np.log10(config.lambd)):.0f}.pdf")
