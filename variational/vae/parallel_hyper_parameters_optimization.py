import multiprocessing as mp
from multiprocessing.sharedctypes import RawArray
###############################################################################
import numpy as np

from autoencoders.ae import AutoEncoder
###############################################################################
def to_numpy_array(array: RawArray, array_shape: tuple) -> np.array:
    """Create a numpy array backed by a shared memory Array."""

    array = np.ctypeslib.as_array(array)

    return array.reshape(array_shape)
#######################################################################
def init_worker(
    share_counter: mp.Value,
    share_data: RawArray,
    data_shape: tuple,
    share_architecture: dict,
    share_hyperparameters: dict,
    share_model_directory: str,
) -> None:
    """
    Initialize worker to train different AEs
    PARAMETERS

    """
    global counter
    global data
    global architecture
    global hyperparameters
    global model_directory

    counter = share_counter
    data = to_numpy_array(share_data, data_shape)
    architecture = share_architecture
    hyperparameters = share_hyperparameters
    model_directory = share_model_directory

#######################################################################
def worker(lambda_: float, reconstruction_weight: float):

    hyperparameters["lambda"] = lambda_
    hyperparameters["reconstruction_weight"] = reconstruction_weight

    vae = AutoEncoder(architecture, hyperparameters)
    vae.train(data)
    vae.save_model(model_directory)

    with counter.get_lock():

        print(f"Finish train model {counter.value}", end="\r")
        counter.value += 1
