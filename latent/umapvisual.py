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
print(f"{latent_directory}/{latent_name}")
FileDirectory().file_exists(
    f"{latent_directory}/{latent_name}",
    exit_program=True
)

###############################################################################
print(f"Define UMAP reducers", end="\n")
umap_paramenters = config.section_to_dictionary(
    parser.items("umap"), [","]
)
print(umap_paramenters)
reducer = umap.UMAP()
###############################################################################
# print(f"Save configuration files of run", end="\n")
# save_to = "."
# with open(f"{save_to}/{config_file_name}", "w") as config_file:
#     parser.write(config_file)
###############################################################################
finish_time = time.time()
print(f"\nRun time: {finish_time - start_time:.2f}")
