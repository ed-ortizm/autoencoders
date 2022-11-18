import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (
    MultipleLocator,
    FormatStrFormatter,
    AutoMinorLocator,
)

from sdss.utils.managefiles import FileDirectory

###############################################################################
def ax_tex(ax: plt.Axes, x: float, y: float, text: str) -> None:
    """
    Transfrom axes to position text in sub-plop

    INPUTS
        ax: the axes to transform
        x, y: where to locate the figure
            0, 0 is bottom left
            1, 1 is upper right
        text: text
    """

    assert 0 <= x <= 1
    assert 0 <= y <= 1

    ax.text(
        x,
        y,
        text,
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
        size="x-large",
        weight="bold",
        # backgroundcolor=None
    )


###############################################################################
def visual_history(
    model_id: str,
    history: dict,
    hyperparameters: dict,
    figsize: tuple = (10, 10),
    slice_from: int = 0,
    save_to: str = ".",
    save_format: str = "png",
) -> None:

    """
    Plots loss and its elements, as well as the validations counterpart

    INPUTS
        history: contains values of loss and metrics
            {
                "loss":[values...], "val_loss":[values...],
                "metric":[metric...], "val_metric":[bodies...],
                ...
            }
        hyperparameters: contains AE hyperparameters
            { "lambda": value, "reconstruction_weight": value}
        figsize:
        slice_from: indicate epoch number to start checking plot
        save_to:
        save_format:
    """
    [
        loss,
        validation_loss,
        mse,
        validation_mse,
        kld,
        validation_kld,
        mmd,
        validation_mmd,
    ] = [
        history["loss"][slice_from:],
        history["val_loss"][slice_from:],
        history["mse"][slice_from:],
        history["val_mse"][slice_from:],
        history["KLD"][slice_from:],
        history["val_KLD"][slice_from:],
        history["MMD"][slice_from:],
        history["val_MMD"][slice_from:],
    ]

    epochs = [i + 1 for i in range(len(loss) + slice_from)][slice_from:]

    ###########################################################################
    fig, [(ax_loss, ax_mse), (ax_kld, ax_mmd)] = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=figsize,
        sharex=True,
        # sharey="row",
        gridspec_kw={"wspace": None, "hspace": 0.2},
        constrained_layout=True,
    )

    linewidth = 3
    val_linewidth = 2
    validation_alpha = 0.4
    ###########################################################################
    # loss
    ax_loss.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax_tex(ax_loss, x=0.5, y=0.8, text="loss")

    ax_loss.plot(epochs, loss, "--o", color="black", linewidth=linewidth)

    ax_loss.plot(
        epochs,
        validation_loss,
        "--*",
        label="validation",
        color="red",
        linewidth=val_linewidth,
        alpha=validation_alpha,
    )

    ###########################################################################
    # mse

    if "." in str(hyperparameters["reconstruction_weight"]):

        mse_text = str(hyperparameters["reconstruction_weight"]).split(".")
        mse_text = [int(x) for x in mse_text[-2:]]
        mse_text = f"{mse_text[0]:02d}_{mse_text[0]}"

    else:

        mse_text = f"{hyperparameters['reconstruction_weight']:02d}"

    ax_mse.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax_tex(ax_mse, x=0.5, y=0.8, text=f"MSE: {mse_text}")

    ax_mse.plot(epochs, mse, "--o", color="black", linewidth=linewidth)
    ax_mse.plot(
        epochs,
        validation_mse,
        "--*",
        label="validation",
        color="red",
        linewidth=val_linewidth,
        alpha=validation_alpha,
    )

    ###########################################################################
    # kld
    ax_kld.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax_tex(ax_kld, x=0.5, y=0.8, text="KLD")
    ax_kld.plot(epochs, kld, "--o", color="black", linewidth=linewidth)
    ax_kld.plot(
        epochs,
        validation_kld,
        "--*",
        label="validation",
        color="red",
        linewidth=val_linewidth,
        alpha=validation_alpha,
    )
    ###########################################################################
    # mmd
    ax_mmd.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    ax_tex(ax_mmd, x=0.5, y=0.8, text=f"MMD: {hyperparameters['lambda']:1.0f}")

    ax_mmd.plot(epochs, mmd, "--o", color="black", linewidth=linewidth)
    (mmd_line,) = ax_mmd.plot(
        epochs,
        validation_mmd,
        "--*",
        label="validation",
        color="red",
        linewidth=val_linewidth,
        alpha=validation_alpha,
    )
    ###########################################################################
    ax_kld.set_xlabel("Epochs")
    ax_kld.xaxis.set_major_formatter(FormatStrFormatter("% 1.0f"))
    ax_mmd.set_xlabel("Epochs")
    ax_mmd.xaxis.set_major_formatter(FormatStrFormatter("% 1.0f"))
    fig.legend([mmd_line], ["validation"], loc="center")
    ###########################################################################
    FileDirectory().check_directory(save_to, exit_program=False)

    file_name = (
        f"{model_id}-"
        f"ae_MSE_{mse_text}_"
        f"MMD_{hyperparameters['lambda']:1.0f}"
    )

    fig.savefig(f"{save_to}/{file_name}.{save_format}")

    plt.close()


###############################################################################
