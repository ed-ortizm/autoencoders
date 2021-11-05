#!/usr/bin/env python3.8
from configparser import ConfigParser, ExtendedInterpolation
import os
import sys
import time

import numpy as np

from autoencoders.deterministic.autoencoder import AE
from sdss.superclasses import FileDirectory
###############################################################################
ti = time.time()
###############################################################################
parser = ConfigParser(interpolation=ExtendedInterpolation())
parser.read("ae.ini")
###############################################################################
# load data
print(f"Load data")
data_directory = parser.get("directories", "train")
data_name = parser.get("files", "train")
data = np.load(f"{data_directory}/{data_name}")
input_dimensions = data.shape[1]
###############################################################################
architecture = dict(parser.items("architecture"))
architecture["input_dimensions"] = input_dimensions

hyperparameters = dict(parser.items("hyperparameters"))
###############################################################################
print(f"Build AE")

ae = AE(
    architecture,
    hyperparameters,
    reload=False
)

ae.summary()
#############################################################################
# Training the model
history = ae.train(data)
# save model
model_directory = parser.get("directories", "output")
# model_directory = f"{model_directory}/{ae.architecture_str}"
FileDirectory().check_directory(model_directory, exit=False)

ae.save_model(model_directory)
###############################################################################
tf = time.time()
print(f"Running time: {tf-ti:.2f}")
