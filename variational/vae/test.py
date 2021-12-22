#!/usr/bin/env python3.8
from configparser import ConfigParser, ExtendedInterpolation
import os
import sys
import time

###############################################################################
import numpy as np

from autoencoders.ae import AutoEncoder, SamplingLayer
from autoencoders.customObjects import MyCustomLoss
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
data = np.ones(shape=(10_000, 100))
input_dimensions = data.shape[1]
###############################################################################
architecture = config_handler.section_to_dictionary(
    parser.items("architecture"), value_separators=["_"]
)

architecture["input_dimensions"] = input_dimensions

hyperparameters = config_handler.section_to_dictionary(
    parser.items("hyperparameters"), value_separators=[]
)

# print(architecture)
# print(hyperparameters)
###############################################################################
print(f"Build VAE")

import tensorflow as tf
from tensorflow import keras

vae = AutoEncoder(architecture, hyperparameters, is_variational=True)
# #############################################################################
# Training the model
vae.train(data)
save_to = "/home/edgar/Downloads"
vae.save_model(save_to)
da = keras.models.load_model(
    save_to,
    custom_objects={"MyCustomLoss":MyCustomLoss, "SamplingLayer":SamplingLayer}
)
###############################################################################
tf = time.time()
print(f"Running time: {tf-ti:.2f}")
