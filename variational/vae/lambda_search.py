#!/usr/bin/env python3.8
import os
import tensorflow as tf
# Set environment variables to disable multithreading as users will probably
# want to set the number of cores to the max of their computer.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
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
    data = np.load(f"{data_directory}/{data_name}", mmap_mode='r')

    input_dimensions = data.shape[1]
    array_shape = data.shape
    array_size = data.size
    array_dtype = data.dtype

    del data
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

    hyperparameters["reconstruction_weight"] = input_dimensions
    ###########################################################################
    counter = mp.Value("i", 0)

    share_data = RawArray(
        np.ctypeslib.as_ctypes_type(array_dtype),
        array_size
    )

    #######################################################################
    model_directory = parser.get("directories", "output")
    ###########################################################################
    # 100 models with 80 k to train and 20 k to validate
    # 20: 302 [s] ~ 70% of each thread and load of ~70
    # 25: 267 [s] ~ 80% of each thread and load of ~100
    # 30: 235 [s] ~ 90% of each thread and load of ~120
    # 35: 235 [s] ~ 90% of each thread and load of ~130
    # 48: 260 [s] ~ 100% of each thread and load of ~ 160
    with mp.Pool(
        processes=30,
        initializer=hyper_optimize.init_worker,
        initargs=(
            counter,
            share_data,
            array_shape,
            f"{data_directory}/{data_name}",
            architecture,
            hyperparameters,
            model_directory,
        ),
    ) as pool:

        pool.map(hyper_optimize.worker, lambdas)

    ###########################################################################
    tf = time.time()
    print(f"Running time: {tf-ti:.2f}")