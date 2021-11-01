#!/usr/bin/env python3.8
from configparser import ConfigParser, ExtendedInterpolation
import os
import sys
import time

###############################################################################
import numpy as np

from autoencoders.deterministic.autoencoder import AE
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

#nan_values = np.count_nonzero(np.isnan(data))
#indefinite_values = np.count_nonzero(~np.isfinite(data))
#print(f"Nans and indefinte: {nan_values, indefinite_values}")

# Shuffle for better performance in SGD
#print(f"Shuffle in place train data")
# np.random.shuffle(data)
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
models_directory = parser.get("directories", "models")

if not os.path.exists(f"{models_directory}/{ae.architecture_str}"):
    os.makedirs(f"{models_directory}/{ae.architecture_str}")

ae.save_model(models_directory)
###############################################################################
tf = time.time()
print(f"Running time: {tf-ti:.2f}")
