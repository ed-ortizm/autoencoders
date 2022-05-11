"""Get UMAP visualization of latent representation"""
from configparser import ConfigParser, ExtendedInterpolation
import os
import time

import numpy as np
import umap

from sdss.utils.managefiles import FileDirectory
from sdss.utils.configfile import ConfigurationFile

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
###############################################################################
start_time = time.time()
###############################################################################
parser = ConfigParser(interpolation=ExtendedInterpolation())
config_file_name = "umapvisual.ini"
parser.read(f"{config_file_name}")
config = ConfigurationFile()
###############################################################################
print(f"Load data", end="\n")

latent_directory = parser.get("directory", "latent")
latent_name = parser.get("file", "latent")
FileDirectory().file_exists(
    f"{latent_directory}/{latent_name}",
    exit_program=True
)

latent = np.load(f"{latent_directory}/{latent_name}")
###############################################################################
metrics = config.entry_to_list(parser.get("umap", "metrics"), str, ",")

bin_id = parser.get("common", "bin")

for metric in metrics:

    print(f"UMAP with {metric} metric", end="\n")

    reducer = umap.UMAP(metric=metric)
    embedding = reducer.fit_transform(latent)

    np.save(f"{latent_directory}/umap_{metric}_{bin_id}.npy", embedding)

###############################################################################
print(f"Save configuration file", end="\n")

with open(f"{latent_directory}/{config_file_name}", "w") as config_file:
    parser.write(config_file)
###############################################################################
finish_time = time.time()
print(f"\nRun time: {finish_time - start_time:.2f}")
