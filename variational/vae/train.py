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

###############################################################################
ti = time.time()
###############################################################################
parser = ConfigParser(interpolation=ExtendedInterpolation())
parser.read("vae.ini")
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
    parser.items("hyperparameters"), value_separators=[]
)
###############################################################################
print(f"Build AutoEncoder")

vae = AutoEncoder(architecture, hyperparameters, is_variational=True)

number_params = vae.model.count_params()
print(f"\nThe model has {number_params} parameters", end="\n")
#############################################################################
# Training the model
vae.train(data)
# save model
model_directory = parser.get("directories", "output")
model_directory = f"{model_directory}/{vae.architecture_str}"
FileDirectory().check_directory(model_directory, exit=False)

vae.save_model(model_directory)
###############################################################################
tf = time.time()
print(f"Running time: {tf-ti:.2f}")
