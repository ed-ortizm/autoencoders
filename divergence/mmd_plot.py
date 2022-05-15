"""Visuals of MMD divergence"""
import matplotlib.pyplot as plt
import numpy as np

import autoencoders
##############################################################################
mmd = autoencoders.divergence.mmd.MMD(number_prior_samples=1000)

fig, axs = plt.subplots(
    nrows=2,
    ncols=3,
    figsize=(15, 8),
    sharex=True,
    sharey=False,
    tight_layout=True,
)

stds = np.array([0.25, 0.5, 1, 4, 9, 16]).reshape(2, 3)

for idx_row, ax_row in enumerate(axs):

    for idx_col, ax in enumerate(ax_row):

        ax.plot(mmd.prior_samples[:, 0], color="black", label="N(0,1)")

        std = stds[idx_row, idx_col]

        divergence, g = mmd.to_gaussian(1000, mu=0, std=std)

        ax.plot(g[:, 0], label=f"N(1, {std})\nMMD={divergence:.2f}", alpha=0.4)
        ax.tick_params(labelleft=True, labelbottom=True)
        ax.legend()
##############################################################################
mmd = autoencoders.divergence.mmd.MMD(number_prior_samples=1000)

fig, axs = plt.subplots(
    nrows=2,
    ncols=3,
    figsize=(15, 8),
    sharex=True,
    sharey=True,
    tight_layout=True,
)

mus = np.arange(-2, 4, 1).reshape(2, 3)

for idx_row, ax_row in enumerate(axs):

    for idx_col, ax in enumerate(ax_row):

        ax.plot(mmd.prior_samples[:, 0], color="black", label="N(0,1)")

        mu = mus[idx_row, idx_col]

        divergence, g = mmd.to_gaussian(1000, mu=mu, std=1)

        ax.plot(g[:, 0], label=f"N({mu}, 1)\nMMD={divergence:.2f}", alpha=0.4)
        ax.tick_params(labelleft=False, labelbottom=True)
        ax.legend()
##############################################################################
