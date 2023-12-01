import argparse
from functools import partial
import logging

import numpy as np

from rates.targets import (
    Fourier_monomial,
    absolute_value,
    clip_exp,
    monomial,
    step_2d,
    two_cos,
    vanish_cos_2d,
)
from rates.architectures.kernels import (
    polynomial_kernel,
    rbf_kernel,
    rbf_periodic,
)
from rates.auxillary.io import write_numpy_file
from rates.auxillary.training import one_trial
from rates.config import SAVE_DIR, AUTHOR
from rates.datasets import (
    GeneratorGaussian1d,
    GeneratorUniform,
    GeneratorUniform1d,
    GeneratorUniform2d,
)

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    prog="Rates",
    description="Experimental rates computation",
    epilog=f"@ 2023, {AUTHOR}",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "name",
    choices=[
        "absolute_value",
        "fast_slow",
        "single_cos",
        "single_poly",
        "slow_fast",
        "step",
        "two_cos",
        "vanish_cos",
    ],
    help="Name of the experiment",
    type=str,
)
parser.add_argument(
    "-e",
    "--nb-exp",
    default=100,
    nargs="?",
    help="Number of experiments to estimate risk expectation",
    type=int,
)
parser.add_argument(
    "--n-train",
    default=1000,
    nargs="?",
    help="Number of traning samples",
    type=int,
)
parser.add_argument(
    "--n-test",
    default=2500,
    nargs="?",
    help="Number of testing samples",
    type=int,
)
parser.add_argument(
    "-d",
    "--dimension",
    default=1,
    nargs="?",
    help="Input dimension of the problem",
    type=int,
)
parser.add_argument(
    "--seed",
    default=0,
    nargs="?",
    help="Random seed",
    type=int,
)
parser.add_argument(
    "--noise",
    default=0.01,
    nargs="?",
    help="Level of noise in labels",
    type=float,
)


parser.add_argument(
    "--omega",
    default=20,
    nargs="?",
    help="Cosine frequency",
    type=float,
)
parser.add_argument(
    "--clip",
    default=0.25,
    nargs="?",
    help="Clipping for the target function",
    type=float,
)
config = parser.parse_args()


np.random.seed(config.seed)

d = config.dimension
n_train = config.n_train
noise_level = config.noise
n_test = config.n_test
nb_trials = config.nb_exp

# Target function in 1d
if config.name == "slow_fast":
    d = 1
    sigmas = np.logspace(-3, -1, 20)
    lambdas = np.logspace(-8, 0, 20)

    kernel = rbf_periodic
    func = partial(Fourier_monomial, omega=config.omega)
    pred_clip = (-2, 2)

    x_gen = GeneratorUniform1d.x_gen
    x_test_gen = GeneratorUniform1d.x_test_gen
    weights_gen = GeneratorUniform1d.weights_gen

elif config.name == "fast_slow":
    d = 1
    sigmas = np.logspace(-1, 1, 20)
    lambdas = np.logspace(-10, 0, 20)

    kernel = rbf_kernel
    func = partial(clip_exp, clip=config.clip)
    pred_clip = (-1 + np.exp(-config.clip), 1 - np.exp(-config.clip))

    x_gen = GeneratorGaussian1d.x_gen
    x_test_gen = GeneratorGaussian1d.x_test_gen
    weights_gen = GeneratorGaussian1d.weights_gen

# Target function in 2d
elif config.name == "absolute_value":
    d = 2
    sigmas = np.logspace(np.log10(0.05), np.log10(5), num=30)
    lambdas = np.logspace(-8, 0, 30)

    kernel = rbf_kernel
    func = absolute_value
    pred_clip = (-1, 3)

    x_gen = GeneratorUniform2d.x_gen
    x_test_gen = GeneratorUniform2d.x_test_gen
    weights_gen = GeneratorUniform2d.weights_gen

elif config.name == "step":
    d = 2
    sigmas = np.logspace(np.log10(0.05), np.log10(5), num=30)
    lambdas = np.logspace(-8, 0, 30)

    kernel = rbf_kernel
    func = step_2d
    pred_clip = (-2, 2)

    x_gen = GeneratorUniform2d.x_gen
    x_test_gen = GeneratorUniform2d.x_test_gen
    weights_gen = GeneratorUniform2d.weights_gen

elif config.name == "two_cos":
    d = 2
    sigmas = np.logspace(-1, 1, 3)
    lambdas = np.logspace(-8, 0, 3)

    kernel = rbf_periodic
    func = two_cos
    pred_clip = None

    x_gen = GeneratorUniform2d.x_gen
    x_test_gen = GeneratorUniform2d.x_test_gen
    weights_gen = GeneratorUniform2d.weights_gen

elif config.name == "vanish_cos":
    d = 2
    sigmas = np.logspace(np.log10(0.05), np.log10(5), num=30)
    lambdas = np.logspace(-8, 0, 3)

    kernel = rbf_kernel
    func = vanish_cos_2d
    pred_clip = None

    x_gen = GeneratorUniform2d.x_gen
    x_test_gen = GeneratorUniform2d.x_test_gen
    weights_gen = GeneratorUniform2d.weights_gen

# Target function in arbitrary dimension
elif config.name == "single_cos":
    sigmas = [0.1 * np.sqrt(d)]
    lambdas = [1e-6]

    kernel = rbf_kernel
    func = Fourier_monomial
    pred_clip = (-1, 1)

    x_gen = GeneratorUniform.x_gen
    x_test_gen = GeneratorUniform.x_test_gen
    weights_gen = GeneratorUniform.weights_gen

elif config.name == "single_poly":
    sigmas = [5]  # degree of the polynomial kernel
    lambdas = [1e-6]

    kernel = polynomial_kernel
    func = partial(monomial, legendre=False)
    pred_clip = (-1, 1)

    x_gen = GeneratorUniform.x_gen
    x_test_gen = GeneratorUniform.x_test_gen
    weights_gen = GeneratorUniform.weights_gen

else:
    raise NotImplementedError


logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        # logging.FileHandler(LOG_DIR / f"{config.name}_{d}d.log"),
    ],
)
save_file = SAVE_DIR / f"{config.name}_{d}d.npy"

for trial in range(nb_trials):
    x_train = x_gen(n_train, d)
    y_train = func(x_train)
    y_train += noise_level * np.random.randn(n_train)

    x_test = x_test_gen(n_test, d)
    weights = weights_gen(x_test)
    y_test = func(x_test)

    for i_sigma, sigma in enumerate(sigmas):
        K = kernel(x_train, x_train, sigma)
        K_test = kernel(x_test, x_train, sigma)

        for i_lambda, lambd in enumerate(lambdas):
            K_reg = K + lambd * np.eye(n_train)
            error = one_trial(
                K_reg, y_train, K_test, y_test, clip=pred_clip, weights=weights
            )
            write_numpy_file(error, save_file)
            logger.info(f"d: {d}, trail: {trial}, sigma: {i_sigma}, lambda: {i_lambda}")
