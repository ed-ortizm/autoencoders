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
from configparser import ConfigParser, ExtendedInterpolation
import itertools
import sys
import time

###############################################################################
import numpy as np

from autoencoders.ae import AutoEncoder, SamplingLayer
from autoencoders.customObjects import MyCustomLoss
from sdss.superclasses import ConfigurationFile, FileDirectory

###############################################################################
ti = time.time()
###############################################################################
config_handler = ConfigurationFile()
parser = ConfigParser(interpolation=ExtendedInterpolation())
parser.read("lambda_search.ini")
###############################################################################
# load data
print(f"Load data")
data_directory = parser.get("directories", "train")
data_name = parser.get("files", "train")
data = np.load(f"{data_directory}/{data_name}")
input_dimensions = data.shape[1]
###############################################################################
architecture = config_handler.section_to_dictionary(
    parser.items("architecture"), value_separators=["_"]
)

architecture["input_dimensions"] = input_dimensions

hyperparameters = config_handler.section_to_dictionary(
    parser.items("hyperparameters"), value_separators=["_"]
)
print(hyperparameters)
###############################################################################
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

reconstruction_weights = np.array(hyperparameters["reconstruction_weights"])
print(reconstruction_weights)
hyperparameters_grid = itertools.product(reconstruction_weights, lambdas)
###############################################################################

    # vae = AutoEncoder(architecture, hyperparameters)
    #
    # number_params = vae.model.count_params()
    # print(f"\nThe model has {number_params} parameters", end="\n")
    # ###########################################################################
    # print("Train the model")
    # vae.train(data)
    # del data
    # ###########################################################################
    # # save model
    # architecture_str = architecture["encoder"]
    # architecture_str += [architecture["latent_dimensions"]]
    # architecture_str += architecture["decoder"]
    # architecture_str = "_".join(str(unit) for unit in architecture_str)
    #
    # model_directory = parser.get("directories", "output")
    # model_directory = f"{model_directory}/{architecture_str}"
    #
    # model_name = f"{architecture['model_name']}"
    # model_name += f"_alpha_{hyperparameters['alpha']}"
    # model_name += f"_lambda_{hyperparameters['lambda']}"
    #
    # FileDirectory().check_directory(
    #     f"{model_directory}/{model_name}",
    #     exit=False
    # )
    #
    # vae.save_model(f"{model_directory}/{model_name}")
###############################################################################
tf = time.time()
print(f"Running time: {tf-ti:.2f}")
