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
    train_history["loss"], train_history["validation_loss"],
    train_history["mse"], train_history["val_mse"],
    train_history["KLD"], train_history["val_KLD"],
    train_history["MMD"], train_history["valMMD"]
    ]

    epochs = [i+1 for i in range(len(loss))]

    ###########################################################################
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 5), sharex=True)

    axs[0, 0].plot(epochs, loss, label="loss")
    axs[0, 0].plot(epoch, validation_loss, label="val loss")

    axs[0, 1].plot(epoch, mse, label="mse")
    axs[0, 1].plot(epoch, validation_mse, label="val MSE")

    axs[1, 0].plot(epoch, kld, label="kld")
    axs[1, 0].plot(epoch, validation_kld, label="val KLD")
    
    axs[1, 1].plot(epoch, mmd, label="MMD")
    axs[1, 1].plot(epoch, validation_mmd, label="val MMD")

    axs[:::].set_xlabel(f"epochs")

    ###########################################################################
    plot_title = (
    f"$\alpha$: {hyperparameters['']}, \t \t"
    f"$\lambda$: {hyperparameters['']} \n"
    f"L = $\alpha$ * MSE + KLD + ($\lamda -1$) MMD"
    )
    ###########################################################################
    fig.savefig(f"{save_to}.pdf")

    plt.close()

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



    reconstruction = np.array(train_history["history"]["mse"])
    kl_divergence = (loss - reconstruction) / kl_weight
###############################################################################
