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
import glob
import time

import numpy as np
import tensorflow as tf

from autoencoders.ae import AutoEncoder
from sdss.utils.managefiles import FileDirectory
from sdss.utils.configfile import ConfigurationFile

###############################################################################
start_time = time.time()
###############################################################################
parser = ConfigParser(interpolation=ExtendedInterpolation())
config_file_name = "latent.ini"
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

fluxes_name = parser.get("file","fluxes")
fluxes = np.load(f"{data_directory}/{fluxes_name}")

models_directory = parser.get("directory", "model")
model_locations = glob.glob(f"{models_directory}/*/")

# load model config file to save in output dir
model_parser = ConfigParser(interpolation=ExtendedInterpolation())
model_config_file_name = glob.glob(f"{models_directory}/*.ini")[0]
model_parser.read(model_config_file_name)

models_id = [model_id.split("/")[-2] for model_id in model_locations]

for idx, model_location in enumerate(model_locations):

    print(f"{bin_id}: Latent space of model {models_id[idx]}", end="\n")

    model = AutoEncoder(reload=True, reload_from=model_location)
    ###########################################################################

    latent_representation = model.encode(fluxes)

    save_data_to = parser.get("directory", "latent")
    save_data_to = f"{save_data_to}/{models_id[idx]}"
    FileDirectory().check_directory(save_data_to, exit_program=False)

    np.save(f"{save_data_to}/latent_{bin_id}.npy", latent_representation)
    ###########################################################################
    print(f"Save configuration files of run", end="\n")

    with open(f"{save_data_to}/{config_file_name}", "w") as config_file:
        parser.write(config_file)
    ###########################################################################
    print(f"Save configuration files of models", end="\n")

    with open(f"{save_data_to}/model.ini", "w") as config_file:
        model_parser.write(config_file)
###############################################################################
session.close()
finish_time = time.time()
print(f"\nRun time: {finish_time - start_time:.2f}")
