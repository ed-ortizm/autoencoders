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

import parallel_hyper_parameters_optimization as hyper_optimize
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
    #######################################################################
    if parser.getboolean("code", "test") is True:
        data = data[:10_000]
        array_shape = data.shape
    #######################################################################
    print("Raw array")
    # share_data = RawArray(
    raw_data = RawArray(
        np.ctypeslib.as_ctypes_type(data.dtype),
        data.size
        # data.flatten()
    )

    share_data = np.frombuffer(raw_data, dtype=data.dtype).reshape()
    share_data[...] = np.load(f"{data_directory}/{data_name}")

    del data

    model_directory = parser.get("directories", "output")
    ###########################################################################
    with mp.Pool(
        processes=10,
        initializer=hyper_optimize.init_worker,
        initargs=(
            counter,
            share_data,
            array_shape,
            architecture,
            hyperparameters,
            model_directory,
        ),
    ) as pool:

        pool.starmap(hyper_optimize.worker, lambda_reconstruction_weight_grid)

    ###########################################################################
    tf = time.time()
    print(f"Running time: {tf-ti:.2f}")
