import argparse

import matplotlib.pyplot as plt
import numpy as np

from rates.auxillary.io import write_numpy_file
from rates.architectures.kernels import rbf_kernel
from rates.config import AUTHOR, SAVE_DIR
from rates.datasets import GeneratorUniform2d, GeneratorUniform1d

plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{mathtools}")
plt.rc("font", size=10, family="serif", serif="cm")
plt.rc("figure", figsize=(2, 1.5))

parser = argparse.ArgumentParser(
    prog="Rates",
    description="Plot effective dimension as a function of parameters",
    epilog=f"@ 2023, {AUTHOR}",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "dimension",
    help="Dimension of the input space",
    type=int,
)
parser.add_argument(
    "-v",
    "--savings",
    action="store_true",
    help="Whether to save computation results or not",
)
config = parser.parse_args()

d = config.dimension

sigmas = np.logspace(np.log10(0.05), np.log10(5), num=30)
lambdas = np.logspace(-8, 0, 30)

if d == 1:
    data_gen = GeneratorUniform1d.x_test_gen
else:
    data_gen = GeneratorUniform2d.x_test_gen

n_inf = 2500
x_inf = data_gen(n_inf)

eff_dim_1 = np.empty((len(sigmas), len(lambdas)))
eff_dim_2 = np.empty((len(sigmas), len(lambdas)))

for i, sigma in enumerate(sigmas):
    K_inf = rbf_kernel(x_inf, x_inf, sigma)
    K_inf /= n_inf
    eigvals = np.linalg.eigvalsh(K_inf)
    eff_dim_1[i] = np.sum(
        eigvals[:, np.newaxis] / (eigvals[:, np.newaxis] + lambdas[np.newaxis, :]),
        axis=0,
    )
    eff_dim_2[i] = np.sum(
        (eigvals[:, np.newaxis] / (eigvals[:, np.newaxis] + lambdas[np.newaxis, :]))**2,
        axis=0,
    )
    print(i)

if config.savings:
    write_numpy_file(eff_dim_1, SAVE_DIR / f"eff_dim1_{d}d.npy")
    write_numpy_file(eff_dim_2, SAVE_DIR / f"eff_dim2_{d}d.npy")


for i, eff_dim in zip([1, 2], [eff_dim_1, eff_dim_2]):
    # Plot a surface in 3d
    fig = plt.figure(figsize=(5, 4))
    ax = fig.gca(projection="3d")
    eff_dim[eff_dim < 1] = 1

    X, Y = np.meshgrid(np.log10(sigmas), np.log10(lambdas))
    surf = ax.plot_surface(X, Y, np.log10(eff_dim), cmap="viridis", vmin=0, vmax=3)
    ax.set_xlabel("sigma")
    ax.set_ylabel("lambda")
    ax.set_zlabel("effective dimension")
    ax.set_zlim(0, 3)

    ax.set_zticks([0, 1, 2, 3])
    ax.set_zticklabels([r"$1$", r"$10$", r"$10^2$", r"$10^3$"])
    ax.set_yticks([-8, -6, -4, -2, 0])
    ax.set_yticklabels([r"$10^{-8}$", r"$10^{-6}$", r"$10^{-4}$", r"$10^{-2}$", r"$10^0$"])
    ax.set_xticks([0.5, 0, -0.5, -1])
    ax.set_xticklabels([r"$3$", r"$1$", r"$.3$", r"$.1$"])

    ax.view_init(30, 30)
    fig.savefig(SAVE_DIR / f"eff_dim_{i}_{d}d.pdf", bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(1, 3))
    ax.axis("off")
    c = fig.colorbar(surf, ax=ax, ticks=[0, 1, 2, 3])
    c.ax.set_yticklabels([r"$1$", r"$10$", r"$10^2$", r"$10^3$"])
    fig.savefig(SAVE_DIR / f"eff_dim_colorbar_{d}d.pdf", bbox_inches="tight")
