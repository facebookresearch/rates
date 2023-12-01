import argparse

import matplotlib.pyplot as plt
import numpy as np

from rates.config import LOG_DIR, AUTHOR
from rates.targets import (
    absolute_value,
    step_2d,
    two_cos,
    vanish_cos_2d,
)


parser = argparse.ArgumentParser(
    prog="Plot setup",
    description="Setup visualization",
    epilog=f"@ 2022, {AUTHOR}",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "-f",
    "--name",
    default="smooth",
    nargs="?",
    help="Target function",
    type=str,
)
parser.add_argument(
    "-t",
    "--root_n_test",
    default=100,
    nargs="?",
    help="Root of the number of testing samples",
    type=int,
)
config = parser.parse_args()


if config.name == "smooth":
    func = vanish_cos_2d
    offset = 1
elif config.name == "non_smooth":
    func = absolute_value
    offset = 1
elif config.name == "heaviside":
    func = step_2d
    offset = 1
elif config.name == "two_cos":
    func = two_cos
    offset = 1.5
else:
    raise ValueError(f"Function not implemented for name={config.name}")


save_file = LOG_DIR / (config.name + ".bitfile")

# testing set
lin = np.linspace(-1, 1, num=config.root_n_test)
X1, X2 = np.meshgrid(lin, lin)
x_test = np.vstack((X1.reshape(-1), X2.reshape(-1))).T
y_test = func(x_test)

# problem visualization
fig, ax = plt.subplots(1, 1, figsize=(3, 3))
if config.name == "smooth":
    begin = y_test.min() - 1
    end = y_test.max() + 1
    levels = [begin] + list(np.linspace(-offset, offset, num=20)) + [end]
else:
    levels = 10
ax.contourf(
    X1,
    X2,
    y_test.reshape(config.root_n_test, config.root_n_test),
    cmap="RdBu",
    levels=levels,
    vmin=-offset,
    vmax=offset,
)
ax.set_axis_off()
fig.tight_layout()
fig.savefig(LOG_DIR / (config.name + ".pdf"))
