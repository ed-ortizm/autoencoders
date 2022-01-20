# process base parallelism to train models in a grid of hyperparameters
import itertools
import multiprocessing as mp
from multiprocessing.sharedctypes import RawArray


import numpy as np

# from autoencoders.ae import AutoEncoder
###############################################################################
def to_numpy_array(array: RawArray, array_shape: tuple) -> np.array:
    """Create a numpy array backed by a shared memory Array."""

    array = np.ctypeslib.as_array(array)

    return array.reshape(array_shape)
###############################################################################
def init_shared_data(
    share_counter: mp.Value,
    share_data: RawArray,
    data_shape: tuple,
    data_location: str,
    share_architecture: dict,
    share_hyperparameters: dict,
    share_model_directory: str,
) -> None:
    """
    Initialize worker to train different AEs

    PARAMETERS

        share_counter:
        share_data:
        data_shape:
        data_location:
        share_architecture:
        share_hyperparameters:
        share_model_directory:

    """
    global counter
    global data
    global architecture
    global hyperparameters
    global model_directory

    counter = share_counter
    data = to_numpy_array(share_data, data_shape)
    data[...] = np.load(data_location)
    architecture = share_architecture
    hyperparameters = share_hyperparameters
    model_directory = share_model_directory

###############################################################################
def build_and_train_model(
    rec_weight: float,
    mmd_weight: float,
    kld_weight: float,
    alpha: float,
    lambda_: float,
)-> None:
    """
    Define the AutoEncoder instance based on hyperparameters from the grid
    PARAMETERS
        rec_weight:
        mmd_weight:
        kld_weight:
        alpha:
        lambda_:

    """
    ###########################################################################
    from tensorflow import keras.backed as  K
    import tensorflow as tf
    from autoencoders.ae import AutoEncoder

    # set the number of cores to use per model in each worker
    jobs = 4
    config = tf.ConfigProto(
        intra_op_parallelism_threads=jobs,
        inter_op_parallelism_threads=jobs,
        allow_soft_placement=True,
        device_count={'CPU': jobs}
    )
    session = tf.Session(config=config)
    K.set_session(session)
    ###########################################################################
    hyperparameters["reconstruction_weight"] = rec_weight
    hyperparameters["mmd_weight"] = mmd_weight
    hyperparameters["kld_weight"] = kld_weight
    hyperparameters["alpha"] = kld_weight
    hyperparameters["lambda"] = lambda_


    with counter.get_lock():

        print(f"Train up to model N:{counter.value:05d}", end="\r")

        counter.value += 1

    vae = AutoEncoder(architecture, hyperparameters)
    vae.train(data)
    vae.save_model(model_directory)
###############################################################################
def get_parameters_grid(hyperparameters: dict) -> itertools.product:
    """
    Returns cartesian product of hyperparameters: reconstruction_weight,
        mmd_weights, kld_weights, alpha and lambda

    PARAMETERS
        hyperparameters:

    OUTPUT
        parameters_grid: iterable with the cartesian product
            of input parameters
    """
    for key, value in hyperparameters.items():

        if type(value) != type([]):
            hyperparameters[key] = [value]

    grid = itertools.product(
        hyperparameters["reconstruction_weight"],
        hyperparameters["mmd_weight"],
        hyperparameters["kld_weight"],
        hyperparameters["alpha"],
        hyperparameters["lambda"],
    )

    return grid
