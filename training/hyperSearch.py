"""Train a grid of autoencoders"""
from configparser import ConfigParser, ExtendedInterpolation
import multiprocessing as mp
from multiprocessing.sharedctypes import RawArray
import os
import time

import numpy as np

from sdss.utils.configfile import ConfigurationFile
from autoencoders import hyperSearch

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
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
###############################################################################
if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)

    ###########################################################################
    ti = time.time()
    ###########################################################################
    config_handler = ConfigurationFile()
    parser = ConfigParser(interpolation=ExtendedInterpolation())
    parser.read("hyperSearch.ini")
    ###########################################################################
    # load data
    print("Load data")
    data_directory = parser.get("directory", "train")
    data_name = parser.get("file", "train")
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
    print("Get hyperparameters grid", end="\n")
    grid = config_handler.section_to_dictionary(
        parser.items("param-search"), value_separators=["_", ","]
    )
    ###########################################################################
    # Set reconstruction weight as list

    if isinstance(grid["reconstruction_weight"], list) is False:

        grid["reconstruction_weight"] = [
            grid["reconstruction_weight"]
        ]
    ###########################################################################
    # Set lambdas
    if grid["lambda"] == "random":

        lambdas = np.exp(
            np.random.uniform(
                low=0.1, high=np.log(1e3), size=(grid["number_lambdas"])
            )
        )

        grid["lambda"] = lambdas.tolist()

    if grid["lambda"] == "uniform":

        lambdas = np.arange(
            0, grid["lambda_top_uniform"], grid["lambda_step_uniform"]
        )
        # lambda must be larger than 1
        lambdas[0] = 2

        grid["lambda"] = lambdas.tolist()

    grid = hyperSearch.get_parameters_grid(grid)
    ###########################################################################
    counter = mp.Value("i", 0)

    share_data = RawArray(np.ctypeslib.as_ctypes_type(array_dtype), array_size)

    #######################################################################
    model_directory = parser.get("directory", "models")
    latent_dimensions = parser.getint("architecture", "latent_dimensions")
    model_directory = f"{model_directory}/latent_{latent_dimensions:02d}"
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
    # Save configuration file
    with open(
        f"{model_directory}/hyperSearch.ini", "w", encoding="utf8"
    ) as configfile:
        parser.write(configfile)
    ###########################################################################
    time_f = time.time()
    print(f"Running time: {time_f-ti:.2f}")
