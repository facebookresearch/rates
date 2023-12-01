import matplotlib.pyplot as plt
import numpy as np

from rates.architectures import rbf_kernel
from rates.auxillary import read_numpy_file
from rates.config import SAVE_DIR
from rates.datasets import GeneratorUniform2d, GeneratorUniform1d

plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{mathtools}")
plt.rc("font", size=10, family="serif", serif="cm")
plt.rc("figure", figsize=(2, 1.5))

n_trials = 100
n_train = 1000
# sigmas = np.logspace(-3, -1, 10)
# lambdas = np.logspace(-8, 0, 30)
sigmas = np.logspace(-1, 1, 20)
lambdas = np.logspace(-10, 0, 20)
# sigmas = np.logspace(-3, -1, 20)
# lambdas = np.logspace(-8, 0, 20)

save_file = SAVE_DIR / "fast_slow_1d.npy"
# save_file = SAVE_DIR / "slow_fast_1d.npy"
shape = (n_trials, len(sigmas), len(lambdas), n_train - 1)
errors = read_numpy_file(save_file, shape=shape, order="C").squeeze()
# errors = read_numpy_file(save_file, order="C")
# num = shape[0] * shape[1] * shape[2] * shape[3]
# errors = errors[:num].reshape(shape)

ns = np.arange(n_train - 1) + 1

S, L, N = np.log10(sigmas), np.log10(lambdas), np.log10(ns)
data_errors = np.log10(errors.mean(axis=0))

l, s, n = 10, -10, -1
# l, s, n = 9, -1, 30
X, Y, Z = np.meshgrid(S[:s], L[l:], N[:n])
data = data_errors[:s, l:, :n].transpose((1, 0, 2))

min_err = np.nanmin(np.log10(data_errors))
max_err = np.nanmax(np.log10(data_errors))
kw = {
    "vmin": -5,
    "vmax": -2,
    "levels": 100,
    "cmap": "RdBu_r",
    "rasterized": True,
}

# Create a figure with 3D ax
fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(111, projection="3d")

ax.set_xlabel(r"Bandwidth $\sigma$", fontsize=10)
ax.set_ylabel(r"Regularizer $\lambda$", fontsize=10)
ax.set_zlabel(r"Samples $n$", fontsize=10)

# Plot contour
xmin, xmax = X.min(), X.max()
ymin, ymax = Y.min(), Y.max()
zmin, zmax = Z.min(), Z.max()
ax.set(xlim=[xmin, xmax], ylim=[ymax, ymin], zlim=[zmin, zmax])
edges_kw = dict(color="0.4", linewidth=1, zorder=1e3, linestyle="--")
ax.plot([xmax, xmax], [ymax, ymin], zmax, **edges_kw)
ax.plot([xmin, xmax], [ymin, ymin], zmax, **edges_kw)
ax.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)

# Cube surfaces
C = ax.contourf(
    X[:, :, -1], Y[:, :, -1], data[:, :, -1], zdir="z", offset=Z.max(), **kw
)
for c in C.collections:
    c.set_rasterized(True)
C = ax.contourf(X[0, :, :], data[0, :, :], Z[0, :, :], zdir="y", offset=Y.min(), **kw)
for c in C.collections:
    c.set_rasterized(True)
Cs = ax.contourf(data[:, -1, :], Y[:, -1, :], Z[:, -1, :], zdir="x", offset=X.max(), **kw)
for c in Cs.collections:
    c.set_rasterized(True)

# Effective dimension
data_gen = GeneratorUniform2d.x_test_gen
data_gen = GeneratorUniform1d.x_test_gen
n_inf = 2500
x_inf = data_gen(n_inf)

# eff_dim_1 = np.empty((len(sigmas), len(lambdas)))
# eff_dim_2 = np.empty((len(sigmas), len(lambdas)))
# for i, sigma in enumerate(sigmas):
#     K_inf = rbf_kernel(x_inf, x_inf, sigma)
#     K_inf /= n_inf
#     eigvals = np.linalg.eigvalsh(K_inf)
#     eff_dim_1[i] = np.sum(
#         eigvals[:, np.newaxis] / (eigvals[:, np.newaxis] + lambdas[np.newaxis, :]),
#         axis=0,
#     )
#     eff_dim_2[i] = np.sum(
#         (eigvals[:, np.newaxis] / (eigvals[:, np.newaxis] + lambdas[np.newaxis, :])) ** 2,
#         axis=0,
#     )
# for eff_dim in [eff_dim_1, eff_dim_2]:
#     E = np.log10(eff_dim)
#     x, y, z = np.full_like(L[l:], S[s - 1]), L[l:], E[s - 1, l:]
#     ind = (z > 0) & (z < N[n])
#     ax.plot(x[ind], y[ind], z[ind], c="C0", zorder=1e3)
#     x, y, z = S[:s], np.full_like(sigmas[:s], L[l]), E[:s, l]
#     ind = (z > 0) & (z < N[n])
#     ax.plot(x[ind], y[ind], z[ind], c="C0", zorder=1e3)
#     S_, L_ = np.meshgrid(S[:s], L[l:])
#     ax.contour(
#         S_, L_, E[:s, l:].T, cmap="tab10_r", levels=[-100, N[n]], zdir="z", offset=Z.max(), zorder=1e5
#     )

n1, n2 = 3, 30
n1, n2 = 30, 300
ax.set_zticks([np.log10(n1), np.log10(n2)])
ax.set_zticklabels([rf"${n1}$", rf"${n2}$"])

s1, s2, s3 = .8, .5, .2
ax.set_xticks([np.log10(s1), np.log10(s2), np.log10(s3)])
ax.set_xticklabels([rf"${s1}$", rf"${s2}$", rf"${s3}$"])

l1, l2, l3 = -1, -2, -3
ax.set_yticks([l1, l2, l3])
ax.set_yticklabels(
    [r"$10^{" + str(l1) + "}$", r"$10^{" + str(l2) + "}$", r"$10^{" + str(l3) + "}$"]
)

ax.view_init(elev=30, azim=40)  # , vertical_axis='x')
# fig.savefig(SAVE_DIR / "block_edge.pdf", bbox_inches='tight')
fig.savefig(SAVE_DIR / "block_edge_zoom.pdf", dpi=300, bbox_inches="tight")

fig, ax = plt.subplots(figsize=(1, 3))
ax.axis("off")
c = fig.colorbar(C, ax=ax, ticks=[-4, -3])
c.ax.set_yticklabels([r"$10^{-4}$", r"$10^{-3}$"])
fig.savefig(SAVE_DIR / "block_step_colorbar.pdf", bbox_inches="tight")
