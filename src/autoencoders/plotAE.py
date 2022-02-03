import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np

from sdss.superclasses import FileDirectory
###############################################################################

###############################################################################
def ax_tex(ax, x, y, text):

    ax.text(
        x, y,
        text,
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes
    )
    # return ax
###############################################################################
def visual_train_history(
    train_history: dict,
    hyperparameters: dict,
    # m: int = 0,
    # n: int = -1,
    save_to: str,
    save_format: str="png"
)-> None:

    ###########################################################################
    [
    loss, validation_loss,
    mse, validation_mse,
    kld, validation_kld,
    mmd, validation_mmd
    ] = [
    train_history["loss"], train_history["val_loss"],
    train_history["mse"], train_history["val_mse"],
    train_history["KLD"], train_history["val_KLD"],
    train_history["MMD"], train_history["val_MMD"]
    ]

    epochs = [i+1 for i in range(len(loss))]

    ###########################################################################
    fig, axs = plt.subplots(
        nrows=4,
        ncols=2,
        figsize=(20, 10),
        sharex=True,
        sharey="row",
        gridspec_kw={
            # "wspace":.5,
            "hspace":.5
        }
    )


    ###########################################################################
    axs[0, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    ax_tex(axs[0, 0], x=0.5, y=0.8, text="loss")
    axs[0, 0].plot(epochs, loss, label="loss")

    ax_tex(axs[0, 1], x=0.5, y=0.8, text="val loss")
    axs[0, 1].plot(epochs, validation_loss, label="val loss")

    ###########################################################################
    axs[1, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    ax_tex(axs[1, 0], x=0.5, y=0.8,
        text=f"MSE: {hyperparameters['reconstruction_weight']:2d}")
    axs[1, 0].plot(epochs, mse, label="MSE")

    ax_tex(axs[1, 1], x=0.5, y=0.8, text="val MSE")
    axs[1, 1].plot(epochs, validation_mse, label="val MSE")

    ###########################################################################
    axs[2, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    ax_tex(axs[2, 0], x=0.5, y=0.8, text=f"KLD")
    axs[2, 0].plot(epochs, kld, label="KLD")

    ax_tex(axs[2, 1], x=0.5, y=0.8, text="val KLD")
    axs[2, 1].plot(epochs, validation_kld, label="val KLD")

    ###########################################################################
    axs[3, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    ax_tex(axs[3, 0], x=0.5, y=0.8,
        text=f"MMD: {hyperparameters['lambda'] -1:1.0f}")
    axs[3, 0].plot(epochs, mmd, label="MMD")

    ax_tex(axs[3, 1], x=0.5, y=0.8, text="val MMD")
    axs[3, 1].plot(epochs, validation_mmd, label="val MMD")

    ###########################################################################
    axs[3, 0].set_xlabel("Epochs")
    axs[3, 1].set_xlabel("Epochs")
    ###########################################################################
    FileDirectory().check_directory(save_to, exit=False)

    file_name = (
        f"ae_MSE_{hyperparameters['reconstruction_weight']:1d}"
        f"MMD_{hyperparameters['lambda'] -1:1.0f}"
    )

    fig.savefig(f"{save_to}/{file_name}.{save_format}")
###############################################################################
