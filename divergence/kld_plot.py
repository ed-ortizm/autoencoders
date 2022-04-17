import matplotlib.pyplot as plt
##############################################################################
fig, axs = plt.subplots(
    nrows= 2,ncols=3, figsize=(15, 8), sharex=True, sharey=True,
    tight_layout=True
)
stds = np.arange(1, 7).reshape(2,3)
for idx_row, ax_row in enumerate(axs):

    for idx_col, ax in enumerate(ax_row):

        ax.plot(kld.x, kld.prior, color="black", label="N(0,1)")

        std=stds[idx_row, idx_col]

        divergence , g = kld.to_gaussian(mu=0, std=std)

        ax.plot(kld.x, g, label=f"N(0,{std})\nKLD={divergence:.2f}")
        ax.tick_params(labelleft=False, labelbottom=False)
        ax.legend()
##############################################################################