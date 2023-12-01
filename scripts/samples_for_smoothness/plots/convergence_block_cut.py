import matplotlib.pyplot as plt
import numpy as np

from rates.architectures import rbf_kernel
from rates.auxillary import read_numpy_file
from rates.config import SAVE_DIR
from rates.datasets import GeneratorUniform1d, GeneratorUniform2d, GeneratorGaussian1d

plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{mathtools}")
plt.rc("font", size=10, family="serif", serif="cm")
plt.rc("figure", figsize=(2, 1.5))

n_trials = 100
n_train = 1000
# sigmas = np.logspace(np.log10(0.05), np.log10(5), num=30)
# lambdas = np.logspace(-8, 0, 30)
# data_gen = GeneratorUniform2d.x_test_gen
# save_file = SAVE_DIR / "step_2d.npy"

# sigmas = np.logspace(-3, -1, 20)
# lambdas = np.logspace(-8, 0, 20)
# data_gen = GeneratorUniform1d.x_test_gen
# save_file = SAVE_DIR / "slow_fast_1d.npy"

sigmas = np.logspace(-1, 1, 20)
lambdas = np.logspace(-10, 0, 20)
data_gen = GeneratorGaussian1d.x_test_gen
save_file = SAVE_DIR / "fast_slow_1d.npy"

shape = [n_trials, len(sigmas), len(lambdas), n_train - 1]
# errors = read_numpy_file(save_file, shape=shape, order="C").squeeze()
errors = read_numpy_file(save_file, order="C")
num = shape[1] * shape[2] * shape[3]
shape[0] = errors.size // num
num = shape[0] * num
errors = errors[:num].reshape(shape)

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

ns = np.arange(n_train - 1) + 1
S, L, N = np.log10(sigmas), np.log10(lambdas), np.log10(ns)
data = np.log10(errors.mean(axis=0))
XS, YS = np.meshgrid(N, L)
XL, YL = np.meshgrid(N, S)
XN, YN = np.meshgrid(S, L)

for i in range(len(sigmas)):
    fig, ax = plt.subplots(1, 1)
    s = ax.contourf(XS, YS, data[i, ...], cmap="RdBu_r", levels=20, extend="both")
    # ax.plot(np.log10(eff_dim_1[i]), np.log10(lambdas), c="C0")
    # ax.plot(np.log10(eff_dim_2[i]), np.log10(lambdas), c='C0')

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=8)

    ax.set_xlim([0, 3])
    ax.set_xticks([np.log10(3), np.log10(30), np.log10(300)])
    ax.set_xticklabels([r"$3$", r"$30$", r"$300$"], fontsize=6)

    ax.set_ylim([-8, 0])
    ax.set_yticks([-1, -4, -7])
    ax.set_yticklabels([r"$10^{-1}$", r"$10^{-4}$", r"$10^{-7}$"], fontsize=6)

    # c = fig.colorbar(s, ticks=[-1, 0])
    # c.ax.set_yticklabels([r"$10^{-1}$", r"$10^{0}$"])
    c = fig.colorbar(s, ticks=[-4, -2])
    c.ax.set_yticklabels([r"$10^{-4}$", r"$10^{-2}$"])
    c.ax.tick_params(labelsize=6)

    ax.set_xlabel("Number of samples", fontsize=8)
    ax.set_ylabel("Lambdas", fontsize=8)
    ax.set_title(f"Sigma: {sigmas[i]:.2f}", fontsize=10)
    fig.savefig(SAVE_DIR / f"sigma_{i}.pdf", bbox_inches="tight")
    fig.savefig(SAVE_DIR / f"sigma_{i}.jpg", bbox_inches="tight")


for i in range(len(lambdas)):
    fig, ax = plt.subplots(1, 1)
    s = ax.contourf(XL, YL, data[:, i, :], levels=20, cmap="RdBu_r", extend="both")
    ax.plot(np.log10(eff_dim_1[:, i]), np.log10(sigmas), c="C0")
    ax.plot(np.log10(eff_dim_2[:, i]), np.log10(sigmas), c="C0")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=8)

    ax.set_xlim([0, 3])
    ax.set_xticks([np.log10(3), np.log10(30), np.log10(300)])
    ax.set_xticklabels([r"$3$", r"$30$", r"$300$"], fontsize=6)

    # ax.set_yticks([0, -1])
    # ax.set_yticklabels([r"$1$", r"$0.1$"], fontsize=6)
    # ax.set_yticks([-2.5, -1.5])
    # ax.set_yticklabels([r"$.003$", r"$0.03$"], fontsize=6)
    ax.set_yticks([-1, 1])
    ax.set_yticklabels([r"$.1$", r"$10$"], fontsize=6)

    # c = fig.colorbar(s, ticks=[-1, 0])
    # c.ax.set_yticklabels([r"$10^{-1}$", r"$10^{0}$"])
    c = fig.colorbar(s, ticks=[-4, -2, 0])
    c.ax.set_yticklabels([r"$10^{-4}$", r"$10^{-2}$", r"$10^{0}$"])
    c.ax.tick_params(labelsize=6)

    ax.set_xlabel("Number of samples", fontsize=8)
    ax.set_ylabel("Sigmas", fontsize=8)
    ax.set_title(f"Lambda: {lambdas[i]:.2e}", fontsize=10)

    fig.savefig(SAVE_DIR / f"lambda_{i}.pdf", bbox_inches="tight")
    fig.savefig(SAVE_DIR / f"lambda_{i}.jpg", bbox_inches="tight")
    break


S_, L_ = np.meshgrid(np.log10(sigmas), np.log10(lambdas))
for i, n in enumerate([30, 100, 300, 1000]):
    fig, ax = plt.subplots(1, 1)
    s = ax.contourf(XN, YN, data[..., n - 2], levels=20, cmap="RdBu_r", extend="both")
    # ax.contour(
        # S_, L_, np.log10(eff_dim_1), cmap="tab10_r", alpha=1, leels=[-1, np.log10(n)]
    # )
    ax.contour(
        S_, L_, np.log10(eff_dim_2), cmap="tab10_r", alpha=1, levels=[-100, np.log10(n)]
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=8)

    # ax.set_xticks([0, -1])
    # ax.set_xticklabels([r"$1$", r"$0.1$"], fontsize=6)
    # ax.set_xticks([-2.5, -1.5])
    # ax.set_xticklabels([r"$.003$", r"$0.03$"], fontsize=6)
    ax.set_xticks([-1, 1])
    ax.set_xticklabels([r"$.1$", r"$10$"], fontsize=6)

    ax.set_yticks([-1, -4, -7])
    ax.set_yticklabels([r"$10^{-1}$", r"$10^{-4}$", r"$10^{-7}$"], fontsize=6)

    c = fig.colorbar(s, ticks=[-4, -2, 0])
    c.ax.set_yticklabels([r"$10^{-4}$", r"$10^{-2}$", r"$10^{0}$"])
    # c = fig.colorbar(s, ticks=[-1, 0])
    # c.ax.set_yticklabels([r"$10^{-1}$", r"$10^{0}$"])
    c.ax.tick_params(labelsize=6)

    ax.set_xlabel("Sigmas", fontsize=8)
    ax.set_ylabel("Lambdas", fontsize=8)
    ax.set_title(f"With {n} samples", fontsize=10)
    fig.savefig(SAVE_DIR / f"n_{i}.pdf", bbox_inches="tight")
    fig.savefig(SAVE_DIR / f"n_{i}.jpg", bbox_inches="tight")
