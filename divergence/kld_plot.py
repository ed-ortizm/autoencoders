"""Visuals for KLD divergence between a normal and Gaussian distribution"""
import matplotlib.pyplot as plt
import numpy as np

from autoencoders.divergence.kld import KLD
##############################################################################
stds = np.array([0.25, 0.5, 1, 4, 9, 16]).reshape(2, 3)

fig, axs = plt.subplots(
    nrows=2,
    ncols=3,
    figsize=(15, 8),
    sharex=True,
    sharey=True,
    tight_layout=True,
)

kld = KLD()

for idx_row, ax_row in enumerate(axs):

    for idx_col, ax in enumerate(ax_row):

        ax.plot(kld.x, kld.prior, color="black", label="N(0,1)")

        std = stds[idx_row, idx_col]

        divergence, g = kld.to_gaussian(mu=0, std=std)

        ax.plot(kld.x, g, label=f"N(0,{std})\nKLD={divergence:.2f}")
        ax.tick_params(labelleft=False, labelbottom=False)
        ax.legend()
##############################################################################
mus = np.arange(-2, 4, 1).reshape(2, 3)

fig, axs = plt.subplots(
    nrows=2,
    ncols=3,
    figsize=(15, 8),
    sharex=True,
    sharey=True,
    tight_layout=True,
)

for idx_row, ax_row in enumerate(axs):
    for idx_col, ax in enumerate(ax_row):

        ax.plot(kld.x, kld.prior, color="black", label="N(0,1)")

        mu = mus[idx_row, idx_col]

        divergence, g = kld.to_gaussian(mu=mu, std=1)

        ax.plot(kld.x, g, label=f"N({mu}, 1)\nKLD={divergence:.2f}")
        ax.tick_params(labelleft=False, labelbottom=True)
        ax.legend()
