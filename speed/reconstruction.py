"""Get latent representation of a set of models"""
from configparser import ConfigParser, ExtendedInterpolation
import glob
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import time

import numpy as np
import tensorflow as tf

from autoencoders.ae import AutoEncoder
from sdss.utils.managefiles import FileDirectory

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
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
###############################################################################
start_time = time.time()
###############################################################################
parser = ConfigParser(interpolation=ExtendedInterpolation())
config_file_name = "reconstruction.ini"
parser.read(f"{config_file_name}")
###############################################################################
# set the number of cores to use per model in each worker
jobs = parser.getint("configuration", "cores_per_worker")
config = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=jobs,
    inter_op_parallelism_threads=jobs,
    allow_soft_placement=True,
    device_count={"CPU": jobs},
)
session = tf.compat.v1.Session(config=config)
###############################################################################
meta = parser.get("common", "meta")
bin_id = parser.get("common", "bin")

print(f"Load {bin_id} from {meta}", end="\n")

data_directory = parser.get("directory", "data")
FileDirectory().check_directory(data_directory, exit_program=True)

fluxes_name = parser.get("file", "fluxes")
fluxes = np.load(f"{data_directory}/{fluxes_name}", mmap_mode="r")

number_spectra = parser.getint("configuration", "number_spectra")
fluxes = fluxes[:number_spectra].copy()

models_directory = parser.get("directory", "model")
model_id = parser.get("file", "model_id")

print(f"Load model: {models_directory}/{model_id}", end="\n")
# load model config file to save in output dir
model_parser = ConfigParser(interpolation=ExtendedInterpolation())
model_config_file_name = glob.glob(f"{models_directory}/*.ini")[0]
model_parser.read(model_config_file_name)

model = AutoEncoder(reload=True, reload_from=f"{models_directory}/{model_id}")

# (batch_size, time)
reconstruction_time = np.empty((number_spectra, 2))

for idx, batch_size in enumerate(range(1, number_spectra + 1)):

    print(f"N spectra: {batch_size}", end="\r")

    data = fluxes[:batch_size]
    if batch_size == number_spectra:
        data = fluxes

    start = time.perf_counter()
    model.reconstruct(data)
    finish = time.perf_counter()

    reconstruction_time[idx, 0] = batch_size
    reconstruction_time[idx, 1] = finish - start

save_data_to = parser.get("directory", "speed")
save_data_to = f"{save_data_to}/{model_id}"
FileDirectory().check_directory(save_data_to, exit_program=False)

np.save(f"{save_data_to}/speed_reconstruction.npy", reconstruction_time)

print("Save configuration files of run", end="\n")

with open(
    f"{save_data_to}/{config_file_name}", "w", encoding="utf8"
) as config_file:
    parser.write(config_file)

print("Save configuration files of model", end="\n")

with open(f"{save_data_to}/model.ini", "w", encoding="utf8") as config_file:
    model_parser.write(config_file)
###############################################################################
session.close()
finish_time = time.time()
print(f"\nRun time: {finish_time - start_time:.2f}")
