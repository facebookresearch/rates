import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import binom

from rates.auxillary import read_numpy_file
from rates.config import SAVE_DIR, AUTHOR

plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{mathtools}")
plt.rc("font", size=10, family="serif", serif="cm")
plt.rc("figure", figsize=(2, 1.5))

parser = argparse.ArgumentParser(
    prog="Rates Plot",
    description="Plotting empirical convergence rates",
    epilog=f"@ 2023, {AUTHOR}",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "name",
    choices=["poly", "cos"],
    help="Name of the target function",
    type=str,
)
parser.add_argument(
    "-n",
    "--n-train",
    default=1000,
    nargs="?",
    help="Number of traning samples used",
    type=int,
)
parser.add_argument(
    "-t",
    "--nb-trials",
    default=100,
    nargs="?",
    help="Number of trials run",
    type=int,
)
parser.add_argument(
    "-d",
    "--dimension",
    default=10,
    nargs="?",
    help="Input dimensions",
    type=int,
)
config = parser.parse_args()

name = config.name

ds = config.dimension
nb_trials = config.nb_trials
n_train = config.n_train

errors = np.zeros((ds, n_train - 1, nb_trials))

for d in range(10):
    save_file = SAVE_DIR / f"single_{name}_{d+1}d.npy"
    errors[d] = read_numpy_file(save_file, shape=(-1, nb_trials), order="F")

d, n = np.meshgrid(np.arange(1, ds + 1), np.arange(2, n_train + 1))

fig, ax = plt.subplots(1, 1)
tmp = np.linspace(-5, 0.2, 20)
s = ax.contourf(
    n, d, np.log10(errors.mean(axis=-1)).T, cmap="RdBu_r", levels=tmp, extend="both"
)
ax.set_xscale("log")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="both", which="major", labelsize=6)
ax.set_yticks([1, 5, 9])

ax.set_xlabel("Number of samples", fontsize=8)
ax.set_ylabel("Number of dimension", fontsize=8)
c = fig.colorbar(s, ticks=[-4, -2, 0])
c.ax.set_yticklabels([r"$10^{-4}$", r"$10^{-2}$", r"$10^{0}$"])
c.ax.tick_params(labelsize=6)
fig.savefig(SAVE_DIR / f"single_{name}.pdf", bbox_inches="tight", pad_inches=0.05)


# Plotting lower bound
errors_lower = np.zeros((ds, n_train - 1))
for i in range(ds):
    errors_lower[i] = binom(i + 6, 5) / (2 + np.arange(n_train - 1))
errors_lower *= 1e-4

fig, ax = plt.subplots(1, 1)
s = ax.contourf(
    n, d, np.log10(errors_lower).T, levels=tmp, cmap="RdBu_r", extend="both"
)
ax.set_xscale("log")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="both", which="major", labelsize=6)
ax.set_yticks([1, 5, 9])

ax.set_xlabel("Number of samples", fontsize=8)
ax.set_ylabel("Number of dimension", fontsize=8)
c = fig.colorbar(s, ticks=[-2, -4, -6])
c.ax.set_yticklabels([r"$10^{-2}$", r"$10^{-4}$", r"$10^{-6}$"])
c.ax.tick_params(labelsize=6)
fig.savefig(SAVE_DIR / f"single_lower_{name}.pdf", bbox_inches="tight", pad_inches=0.05)
