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
# Set TensorFlow print of log information
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
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

from sdss.superclasses import ConfigurationFile, FileDirectory
from autoencoders import hyperSearch

###############################################################################
if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)

    ###########################################################################
    ti = time.time()
    ###########################################################################
    config_handler = ConfigurationFile()
    parser = ConfigParser(interpolation=ExtendedInterpolation())
    parser.read("search_hyperparameters.ini")
    ###########################################################################
    # load data
    print(f"Load data")
    data_directory = parser.get("directories", "train")
    data_name = parser.get("files", "train")
    data = np.load(f"{data_directory}/{data_name}", mmap_mode="r")

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
    ###########################################################################
    print(f"Get hyperparameters grid", end="\n")
    grid = config_handler.section_to_dictionary(
        parser.items("param-search"), value_separators=["_"]
    )

    grid["reconstruction_weight"].append(input_dimensions)

    if grid["lambda"] == "random":

        lambdas = np.exp(
            np.random.uniform(
                low=0, high=np.log(2e2), size=(grid["number_lambdas"])
            )
        )

        grid["lambda"] = lambdas.tolist()


    grid = hyperSearch.get_parameters_grid(grid)
    ###########################################################################
    counter = mp.Value("i", 0)

    share_data = RawArray(np.ctypeslib.as_ctypes_type(array_dtype), array_size)

    #######################################################################
    model_directory = parser.get("directories", "output")
    ###########################################################################
    number_processes = parser.getint("configuration", "number_processes")
    cores_per_worker = parser.getint("configuration", "cores_per_worker")
    with mp.Pool(
        processes=number_processes,
        initializer=hyperSearch.init_shared_data,
        initargs=(
            counter,
            share_data,
            array_shape,
            f"{data_directory}/{data_name}",
            architecture,
            hyperparameters,
            model_directory,
            cores_per_worker,
        ),
    ) as pool:

        pool.starmap(hyperSearch.build_and_train_model, grid)

    ###########################################################################
    time_f = time.time()
    print(f"Running time: {time_f-ti:.2f}")
