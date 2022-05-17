"""Get TSNE embedding of latent representation"""
from configparser import ConfigParser, ExtendedInterpolation
import glob
import os
import time

import numpy as np
from sklearn.manifold import TSNE

from sdss.utils.managefiles import FileDirectory
from sdss.utils.configfile import ConfigurationFile

# Set environment variables to disable multithreading as users will probably
# want to set the number of cores to the max of their computer.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
###############################################################################
start_time = time.time()
###############################################################################
parser = ConfigParser(interpolation=ExtendedInterpolation())
config_file_name = "tsne_embedding.ini"
parser.read(f"{config_file_name}")
config = ConfigurationFile()
###############################################################################
print(f"Load data", end="\n")

latent_directory = parser.get("directory", "latent")
latent_directories = glob.glob(f"{latent_directory}/*/")

models_id = [model_id.split("/")[-2] for model_id in latent_directories]

latent_name = parser.get("file", "latent")

_ = [
    FileDirectory().file_exists(
        f"{latent_location}/{latent_name}", exit_program=True
    )
    for latent_location in latent_directories
]

metrics = config.entry_to_list(parser.get("tsne", "metrics"), str, ",")
jobs = parser.getint("tsne", "jobs")

bin_id = parser.get("common", "bin")


for idx, latent_directory in enumerate(latent_directories):

    latent = np.load(f"{latent_directory}/{latent_name}")

    for metric in metrics:

        print(f"model {models_id[idx]}: TSNE {metric} metric", end="\n")

        reducer = TSNE(metric=metric, n_jobs=jobs)
        embedding = reducer.fit_transform(latent)

        np.save(f"{latent_directory}/tsne_{metric}_{bin_id}.npy", embedding)
    ###########################################################################
    print(f"Save configuration files of run", end="\n")

    with open(f"{latent_directory}/{config_file_name}", "w") as config_file:
        parser.write(config_file)
###############################################################################
finish_time = time.time()
print(f"\nRun time: {finish_time - start_time:.2f}")
