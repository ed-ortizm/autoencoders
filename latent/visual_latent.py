"""Do scatter plots of latent variables, including umap representation"""
from configparser import ConfigParser, ExtendedInterpolation
import time

import numpy as np
import pandas as pd

from sdss.utils.managefiles import FileDirectory
from sdss.utils.configfile import ConfigurationFile

###############################################################################
start_time = time.time()
###############################################################################
parser = ConfigParser(interpolation=ExtendedInterpolation())
config_file_name = "visual_latent.ini"
parser.read(f"{config_file_name}")
config = ConfigurationFile()
###############################################################################
print(f"Load metadata", end="\n")
data_directory = parser.get("directory", "data")
science_df = parser.get("file", "science")

science_df = pd.read_csv(
    f"{data_directory}/{science_df}", index_col="specobjid"
)

bin_directory = parser.get("directory", "bin_data")
specobjid_name = parser.get("file", "specobjid")

specobjid = np.load(f"{bin_directory}/{specobjid_name}")
###############################################################################
print(f"Load latent representation", end="\n")

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
###############################################################################
print(f"Save configuration file", end="\n")

with open(f"{latent_directory}/{config_file_name}", "w") as config_file:
    parser.write(config_file)
###############################################################################
finish_time = time.time()
print(f"\nRun time: {finish_time - start_time:.2f}")
