"""UMAP 2D embedding visualization"""
from configparser import ConfigParser, ExtendedInterpolation
import glob
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sdss.utils.managefiles import FileDirectory
from sdss.utils.configfile import ConfigurationFile

###############################################################################
start_time = time.time()
###############################################################################
parser = ConfigParser(interpolation=ExtendedInterpolation())
config_file_name = "umap_visual.ini"
parser.read(f"{config_file_name}")

config = ConfigurationFile()
manage_files = FileDirectory()
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

bin_df = science_df.loc[specobjid[:, 1]]
del science_df
###############################################################################
print(f"Load embedding and latent representations", end="\n")

latent_directory = parser.get("directory", "latent")
latent_directories = glob.glob(f"{latent_directory}/*/")

latent_name = parser.get("file", "latent")

_ = [
    manage_files.file_exists(
        f"{latent_location}/{latent_name}", exit_program=True
    )
    for latent_location in latent_directories
]

bin_id = parser.get("common", "bin")
metrics = config.entry_to_list(parser.get("umap", "metrics"), str, ",")

# set plot parameters

parameters_of_plot = config.section_to_dictionary(
    parser.items("plot"), value_separators=[","]
)

size = config.entry_to_list(parser.get("plot", "size"), float, ",")
size = tuple(size)
parameters_of_plot["size"] = size

fig, ax = plt.subplots(figsize=size, tight_layout=True)

# flags
number_latent_variables = None
bin_df_of_plot = None


models_ids = [model_id.split("/")[-2] for model_id in latent_directories]

for model_idx, latent_directory in enumerate(latent_directories):

    print(f"model {models_ids[model_idx]}: UMAP", end="\n")

    # load umap embedding
    for metric in metrics:

        embedding = np.load(f"{latent_directory}/umap_{metric}_{bin_id}.npy")

        bin_df[f"{metric}_01"] = embedding[:, 0]
        bin_df[f"{metric}_02"] = embedding[:, 1]

    for hue in parameters_of_plot["hues"]:

        bin_df_of_plot = bin_df[bin_df[hue] != "undefined"]

        for metric in metrics:

            print(f"UMAP metric: {metric}. Hue: {hue}", end="\r")

            pair_plot = sns.scatterplot(
                x=f"{metric}_01",
                y=f"{metric}_02",
                ax=ax,
                data=bin_df_of_plot,
                hue=hue,
                alpha=parameters_of_plot["alpha"],
                s=parameters_of_plot["marker_size"],
                edgecolors=parameters_of_plot["edgecolors"],
            )

            save_to = f"{latent_directory}/umap"
            manage_files.check_directory(save_to, exit_program=False)

            fig.savefig(
                f"{save_to}/"
                f"umap_{metric}_{hue}.{parameters_of_plot['format']}"
            )

            ax.clear()

    ###########################################################################
    print(f"Save configuration file", end="\n")

    with open(f"{latent_directory}/{config_file_name}", "w") as config_file:
        parser.write(config_file)
###############################################################################
finish_time = time.time()
print(f"\nRun time: {finish_time - start_time:.2f}")
