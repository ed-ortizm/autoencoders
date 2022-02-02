import matplotlib.pyplot as plt
import numpy as np

###############################################################################

###############################################################################
def plotHistory(
    train_history: dict,
    hyperparameters: dict,
    m: int = 0,
    n: int = -1,
    loss: bool = False,
    validation_loss: bool = False,
    mse: bool = False,
    validation_mse: bool = False,
    kld: bool = False,
    validation_kld: bool = False,
    mmd: bool = False,
    validation_mmd: bool = False,

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
    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(8, 5), sharex=True)


    axs[0, 0].text(
        .8, .8,
        "loss",
        horizontalalignment="center",
        verticalalignment="center",
        transform=axs[0, 0].transAxes
    )
    axs[0, 0].plot(epochs, loss, label="loss")

    axs[0, 1].text(
        .8, .8,
        "val loss",
        horizontalalignment="center",
        verticalalignment="center",
        transform=axs[0, 1].transAxes
    )
    axs[0, 1].plot(epochs, validation_loss, label="val loss")

    axs[1, 0].plot(epochs, mse, label="mse")
    axs[1, 1].plot(epochs, validation_mse, label="val MSE")

    axs[2, 0].plot(epochs, kld, label="kld")
    axs[2, 1].plot(epochs, validation_kld, label="val KLD")

    axs[3, 0].plot(epochs, mmd, label="MMD")
    axs[3, 1].plot(epochs, validation_mmd, label="val MMD")

    # for ax in axs.reshape(-1):
        # ax.legend()

    # axs[:::].set_xlabel(f"epochs")

    ###########################################################################
    plot_title = (
    f"{hyperparameters['reconstruction_weight']:02.1f}"
    f"{hyperparameters['lambda']:02.1f}"
    # f"L = $\alpha$ * MSE + KLD + ($\lambda -1$) MMD"
    )

    ###########################################################################
    # fig.savefig(f"{save_to}.pdf")

    # plt.close()

    # if loss is True:
    #     loss = train_history["loss"]
    #
    # if validation_loss is True:
    #     validation_loss = train_history["validation_loss"]
    #
    # if mse is True:
    #     mse = train_history["mse"]
    #
    # if validation_mse is True:
    #     validation_mse = train_history["validation_mse"]
    #
    # if kld is True:
    #     kld = train_history["kld"]
    #
    # if validation_kld is True:
    #     validation_kld = train_history["validation_kld"]
    #
    # if mmd is True:
    #     mmd = train_history["mmd"]
    #
    # if validation_mmd is True:
    #     validation_mmd = train_history["validation_mmd"]

###############################################################################
