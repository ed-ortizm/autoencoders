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
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
###############################################################################
from configparser import ConfigParser, ExtendedInterpolation
import time

import numpy as np
import tensorflow as tf

from autoencoders.ae import AutoEncoder
from sdss.superclasses import FileDirectory, ConfigurationFile

###############################################################################
start_time = time.time()
###############################################################################
parser = ConfigParser(interpolation=ExtendedInterpolation())
config_file_name = "latent.ini"
parser.read(f"{config_file_name}")
###############################################################################
# Check files and directory
check = FileDirectory()
# Handle configuration file
configuration = ConfigurationFile()
###############################################################################
# # set the number of cores to use per model in each worker
# jobs = parser.getint("configuration", "cores_per_worker")
# config = tf.compat.v1.ConfigProto(
#     intra_op_parallelism_threads=jobs,
#     inter_op_parallelism_threads=jobs,
#     allow_soft_placement=True,
#     device_count={"CPU": jobs},
# )
# session = tf.compat.v1.Session(config=config)
###############################################################################
# load data
meta = parser.get("common", "meta")
bin_id = parser.get("common", "bin")

model_id = parser.get("file", "model")
print(f"Load model {model_id}")
print(f"Model train on: \n {meta} \n {bin_id}", end="\n")

model_directory = parser.get("directory", "model")
model_directory = f"{model_directory}/{model_id}"

model = AutoEncoder(reload=True, reload_from=model_directory)
###############################################################################
print(f"Load data", end="\n")

data_directory = parser.get("directory", "data")
fluxes_name = parser.get("file","fluxes")

fluxes = np.load(f"{data_directory}/{fluxes_name}")
print(fluxes.shape)
###############################################################################
###############################################################################
# session.close()
###############################################################################
# save_config_to = f"{data_directory}/"
# with open(f"{save_config_to}/{config_file_name}", "w") as config_file:
#     parser.write(config_file)
###############################################################################
finish_time = time.time()
print(f"\nRun time: {finish_time - start_time:.2f}")
