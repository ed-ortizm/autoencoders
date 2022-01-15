#!/usr/bin/env python3.8
import os
# Set environment variables to disable multithreading as users will probably
# want to set the number of cores to the max of their computer.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
###############################################################################
import ctypes
from configparser import ConfigParser, ExtendedInterpolation
import itertools
import multiprocessing as mp
from multiprocessing.sharedctypes import RawArray
import sys
import time

###############################################################################
import numpy as np

from autoencoders.ae import AutoEncoder, SamplingLayer
from autoencoders.customObjects import MyCustomLoss
from sdss.superclasses import ConfigurationFile, FileDirectory

###############################################################################
def to_numpy_array(array, array_shape):
    """Create a numpy array backed by a shared memory Array."""

    array = np.ctypeslib.as_array(array)

    return array.reshape(array_shape)
#######################################################################
def init_worker(
    counter: mp.Value,
    share_data: np.array,
    data_shape: tuple,
    share_architecture: dict,
    share_hyperparameters: dict,
    share_model_directory: str,
) -> None:
    """
    Initialize worker to train different AEs
    PARAMETERS

    """
    global data
    global architecture
    global hyperparameters
    global model_directory

    data = to_numpy_array(share_data, data_shape)
    architecture = share_architecture
    hyperparameters = share_hyperparameters
    model_directory = share_model_directory

#######################################################################
def worker(lambda_: float, reconstruction_weight: float):

    hyperparameters["lambda"] = lambda_
    hyperparameters["reconstruction_weight"] = reconstruction_weight

    vae = AutoEncoder(architecture, hyperparameters)
    print("working")
    # vae.train(data)
    vae.save_model(model_directory)

###############################################################################
if __name__ == "__main__":

    mp.set_start_method("spawn")

    ###########################################################################
    ti = time.time()
    ###########################################################################
    config_handler = ConfigurationFile()
    parser = ConfigParser(interpolation=ExtendedInterpolation())
    parser.read("lambda_search.ini")
    ###########################################################################
    # load data
    print(f"Load data")
    data_directory = parser.get("directories", "train")
    data_name = parser.get("files", "train")
    data = np.load(f"{data_directory}/{data_name}")
    input_dimensions = data.shape[1]
    ###########################################################################
    architecture = config_handler.section_to_dictionary(
        parser.items("architecture"), value_separators=["_"]
    )

    architecture["input_dimensions"] = input_dimensions

    hyperparameters = config_handler.section_to_dictionary(
        parser.items("hyperparameters"), value_separators=["_"]
    )
    print(hyperparameters)
    ###########################################################################
    # set grid for hyperparameters

    # portillo2021:
    # Dimensionality Reduction of SDSS Spectra with Variational Autoencoders

    lambdas = np.exp(
        np.random.uniform(
            low=1,
            high=np.log(1e3),
            size=(hyperparameters["search_lambda"])
        )
    )

    reconstruction_weights = np.array(
        hyperparameters["reconstruction_weights"]
    )

    lambda_reconstruction_weight_grid = itertools.product(
        lambdas, reconstruction_weights
    )
    ###########################################################################
    array_shape = data.shape
    counter = mp.Value("i", 0)
    # RawArray since I just need to read the array
    print("Raw array")
    share_data = RawArray(
        np.ctypeslib.as_ctypes_type(data.dtype),
        data.flatten()
    )

    model_directory = parser.get("directories", "output")
    ###########################################################################
    with mp.Pool(
        processes=48,
        initializer=init_worker,
        initargs=(
            counter,
            share_data,
            # data,
            array_shape,
            architecture,
            hyperparameters,
            model_directory,
        ),
    ) as pool:

        pool.starmap(worker, lambda_reconstruction_weight_grid)

        #
        # model_directory = f"{model_directory}/{architecture_str}"
        #
        #
        # FileDirectory().check_directory(
        #     f"{model_directory}/{model_name}",
        #     exit=False
        # )
        #
    ###########################################################################
    tf = time.time()
    print(f"Running time: {tf-ti:.2f}")
