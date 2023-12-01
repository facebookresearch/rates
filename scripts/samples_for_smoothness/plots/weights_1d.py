import matplotlib.pyplot as plt
import numpy as np

from rates.architectures.kernels import rbf_kernel
from rates.config import SAVE_DIR
from rates.datasets import GeneratorUniform1d

plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{mathtools}")
plt.rc("font", size=10, family="serif", serif="cm")
plt.rc("figure", figsize=(2, 1.5))

np.random.seed(100)

d = 1
n_train = 1000
nb_trials = 100
lambd = 1e-6
sigma = 5e-2
ns = [27, 45, 90, 180, 360, 720]
num = 1000
x = np.linspace(-1, 1, num=5)[:, np.newaxis]
X = np.linspace(-1, 1, num=num)[:, np.newaxis]

alphas = {}
for n in ns:
    alphas[n] = np.empty((nb_trials, num, len(x)))

for i in range(nb_trials):
    data = GeneratorUniform1d.x_gen(n_train)
    K = rbf_kernel(data, data, sigma)
    # K += lambd * np.eye(n_train)

    K_X = rbf_kernel(data, X, sigma)
    K_x = rbf_kernel(data, x, sigma)
    K_Xx = rbf_kernel(X, x, sigma)

    for n in ns:
        w, v = np.linalg.eigh(K[: n - 1, : n - 1])
        K_rootinv = (((w + n * lambd) ** (-0.5)) * v) @ v.T
        Z_X = K_rootinv @ K_X[: n - 1]
        Z_x = K_rootinv @ K_x[: n - 1]

        denominator = np.sum(Z_X**2, axis=0)
        denominator *= -1
        denominator += 1 + n * lambd
        denominator **= -1
        alphas[n][i] = Z_X.T @ Z_x
        alphas[n][i] *= -1
        alphas[n][i] += K_Xx
        alphas[n][i] /= denominator[:, np.newaxis]

n_inf = 2500
x_inf = GeneratorUniform1d.x_test_gen(n_inf)
K_inf = rbf_kernel(x_inf, x_inf, sigma)
K_inf /= n_inf
eigvals = np.linalg.eigvalsh(K_inf)
eff_dim = np.sum(eigvals / (eigvals + lambd))
print(eff_dim)

alpha = {}
for n in ns:
    alpha[n] = np.mean(alphas[n], axis=0).squeeze()

fig, axes = plt.subplots(2, 3, figsize=(8, 5))
for i, n in enumerate(ns):
    axes[i // 3, i % 3].plot(X[10:-10], n * alpha[n][10:-10, 1:-1])
    axes[i // 3, i % 3].set_xticks([-0.5, 0, 0.5])
    axes[i // 3, i % 3].set_yticks([0])
    axes[i // 3, i % 3].set_title(f"High-sample ratio: {n / eff_dim:.2f}")
    axes[i // 3, i % 3].grid(True)
fig.tight_layout()
fig.savefig(SAVE_DIR / "weights.pdf", bbox_inches="tight")
