import matplotlib.pyplot as plt
import numpy as np
###############################################################################

def plot_train_history(train_history: dict, kl_weight: np.float, save_to: str):

    """
    Plot loss, regularization and reconstruction

    PARAMETERS
        train_history: {

        "parameters": {
                'batch_size': 256,
                'epochs': 10,
                'steps': None,
                'samples': 328479,
                'verbose': 1,
                'do_validation': False,
                'metrics': ['loss', 'mse']
            }

        "history": {
            'loss': [0.39, ..., 0.12339567896077754],
            'mse': [0.38579124, ..., 0.11851482]
            }
            this key contains the loss function during training,
            the other metrics shown, correspond to the ones used
            to monitor the training
            # Add custom metrics to show the rest of parameters

        }

    """

    loss = np.array(train_history["history"]["loss"])
    reconstruction = np.array(train_history["history"]["mse"])
    kl_divergence = (loss - reconstruction) / kl_weight

    fig, axs = plt.subplots(
        nrows=2 , ncols=1,
        figsize=(8, 5),
        sharex=True
    )

    axs[0].plot(loss, label="loss")
    axs[0].plot(reconstruction, label="MSE")
    axs[0].legend()

    axs[1].plot(kl_divergence, label="KL-divergence")
    axs[1].legend()

    axs[1].set_xlabel(f'epochs')

    fig.savefig(f'{save_to}.pdf')

    plt.close()

###############################################################################
