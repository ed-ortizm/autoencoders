#!/usr/bin/env python3.8
import os

# Set environment variables to disable multithreading as users will probably
# want to set the number of cores to the max of their computer.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
###############################################################################
from configparser import ConfigParser, ExtendedInterpolation
import sys
import time

###############################################################################
import numpy as np
import tensorflow as tf
from tensorflow import keras

from autoencoders.ae import AutoEncoder
from sdss.superclasses import ConfigurationFile, FileDirectory
###############################################################################
ti = time.time()
###############################################################################
config_handler = ConfigurationFile()
parser = ConfigParser(interpolation=ExtendedInterpolation())
parser.read("train.ini")
###############################################################################
# set the number of cores to use during training
cores_per_worker = parser.getint("tensorflow-session", "cores")
jobs = cores_per_worker
config = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=jobs,
    inter_op_parallelism_threads=jobs,
    allow_soft_placement=True,
    device_count={"CPU": jobs},
)
session = tf.compat.v1.Session(config=config)
###############################################################################
# load data
print(f"Load data", end="\n")
data_directory = parser.get("directories", "train")
data_name = parser.get("files", "train")
data = np.load(f"{data_directory}/{data_name}")
input_dimensions = data.shape[1]
###############################################################################
print(f"Build AutoEncoder", end="\n")

architecture = config_handler.section_to_dictionary(
    parser.items("architecture"), value_separators=["_"]
)

architecture["input_dimensions"] = input_dimensions
# architecture["input_dimensions"] = 3000

hyperparameters = config_handler.section_to_dictionary(
    parser.items("hyperparameters"), value_separators=[]
)
###########################################################################
vae = AutoEncoder(architecture, hyperparameters)
number_params = vae.model.count_params()
print(f"\nThe model has {number_params} parameters", end="\n")
#############################################################################
# Training the model
print("Train the model", end="\n")
vae.train(data)
del data

save_model_to = parser.get("directories", "save_model_to")
print(f"Save model to: {save_model_to}", end="\n")
vae.save_model(f"{save_model_to}")
###############################################################################
# Save reconstructed data
save_reconstruction = parser.getboolean("files", "save_reconstruction")

if save_reconstruction is True:

    print("Save reconstructed spectra", end="\n")
    _, long_model_name = vae.get_architecture_and_model_str()

    observation_name = parser.get("files", "observation")
    observation = np.load(f"{data_directory}/{observation_name}")
    reconstruction = vae.reconstruct(observation)
    np.save(
        f"{data_directory}/reconstructions_{long_model_name}.npy", reconstruction
    )

# close tf session to free resources
session.close()
###############################################################################
tf = time.time()
print(f"Running time: {tf-ti:.2f}")
