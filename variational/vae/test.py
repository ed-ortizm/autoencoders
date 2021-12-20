#!/usr/bin/env python3.8
from configparser import ConfigParser, ExtendedInterpolation
import os
import sys
import time

###############################################################################
import numpy as np

from autoencoders.variational.tfae import VAEtf, MyCustomLoss
from sdss.superclasses import ConfigurationFile, FileDirectory
config_handler = ConfigurationFile()
###############################################################################
ti = time.time()
###############################################################################
parser = ConfigParser(interpolation=ExtendedInterpolation())
parser.read("test.ini")
###############################################################################
# load data
print(f"Load data")
data = np.random.normal(size=(1000, 100))
input_dimensions = data.shape[1]
###############################################################################
architecture = config_handler.section_to_dictionary(
    parser.items("architecture"),
    value_separators = ["_"]
    )

architecture["input_dimensions"] = input_dimensions

hyperparameters = config_handler.section_to_dictionary(
    parser.items("hyperparameters"),
    value_separators = []
    )

# print(architecture)
# print(hyperparameters)
###############################################################################
print(f"Build VAE")

vae = VAEtf(architecture, hyperparameters, is_variational=True)
save_to = "/home/edgar/Download/test_model"
vae.model.save(save_to)
# print(vae.model.summary())
# print(vae.encoder.summary())
# print(vae.decoder.summary())
del vae
# from tensorflow import keras
# keras.models.load_model(
#     save_to,
#     custom_objects={"MyCustomLoss":MyCustomLoss}
# )
# vae.summary()
# #############################################################################
# # Training the model
# vae.train(data)
# print(vae.train_history)
# # print(vae.train_history.history)
# # save model
# model_directory = parser.get("directories", "output")
# model_directory = f"{model_directory}/{vae.architecture_str}"
# FileDirectory().check_directory(model_directory, exit=False)
#
# vae.save_model(model_directory)
# ###############################################################################
# tf = time.time()
# print(f"Running time: {tf-ti:.2f}")
