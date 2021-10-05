#!/usr/bin/env python3.8
from configparser import ConfigParser, ExtendedInterpolation
import os
import sys
import time

###############################################################################
import numpy as np

from autoencoders.variational.autoencoder import VAE, Encoder
###############################################################################
ti = time.time()
###############################################################################
parser = ConfigParser(interpolation=ExtendedInterpolation())
parser.read("vae.ini")
################################################################################
# load data
data_directory = parser.get("directories", "train")
data_name = parser.get("files", "train")
data = np.load(f"{data_directory}/{data_name}")
input_dimensions = data.shape[1]
###############################################################################
architecture = dict(parser.items("architecture"))
architecture["input_dimensions"] = input_dimensions

hyperparameters = dict(parser.items("hyperparameters"))
encoder = Encoder(architecture)
###############################################################################
# vae = VAE(
#     architecture,
#     hyperparameters,
# )
#
# vae.summary()
# ################################################################################
# # Training the model
# history = vae.train(data)
# # save model
# models_directory = parser.get("directories", "models")
#
# if not os.path.exists(f"{models_directory}/{vae.architecture_str}"):
#     os.makedirs(f"{models_directory}/{vae.architecture_str}")
#
# vae.save_model(models_directory)
###############################################################################
###############################################################################
tf = time.time()
print(f"Running time: {tf-ti:.2f}")
