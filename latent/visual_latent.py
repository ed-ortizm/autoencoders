"""Do scatter plots of latent variables, including umap representation"""
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

bin_df = science_df.loc[specobjid[:, 1]]
del science_df
###############################################################################
print(f"Load embedding and latent representations", end="\n")

latent_directory = parser.get("directory", "latent")
latent_directories = glob.glob(f"{latent_directory}/*/")

models_id = [model_id.split("/")[-2] for model_id in latent_directories]

latent_name = parser.get("file", "latent")

_ = [
    FileDirectory().file_exists(
        f"{latent_location}/{latent_name}",
        exit_program=True
    ) for latent_location in latent_directories
]

bin_id = parser.get("common", "bin")
metrics = config.entry_to_list(parser.get("umap", "metrics"), str, ",")

# set plot parameters

size = ConfigurationFile().entry_to_list(
    parser.get("plot", "size"), float, ","
)
size = tuple(size)

fig, ax = plt.subplots(figsize=size, tight_layout=True)

alpha = parser.getfloat("plot", "alpha")
plot_format = parser.get("plot", "format")

number_latent_variables = None
hues = ConfigurationFile().entry_to_list(
    parser.get("plot", "hues"), str, ","
)

show_undefined = parser.getboolean("plot", "show_undefined")

for idx, latent_directory in enumerate(latent_directories):

    latent = np.load(f"{latent_directory}/{latent_name}")
    number_latent_variables = latent.shape[1]

    # load latent representation to data frame
    for idx in range(number_latent_variables):
        bin_df[f"{idx:02d}Latent"] = latent[:, idx]

    # load umap embedding
    for metric in metrics:

        embedding = np.load(f"{latent_directory}/umap_{metric}_{bin_id}.npy")

        bin_df[f"{metric}_01"] = embedding[:, 0]
        bin_df[f"{metric}_02"] = embedding[:, 1]

    print(f"Save pair plots of latent representation", end="\n")

    for hue in hues:

        if show_undefined is True:
            plot_df = bin_df[bin_df[hue]!="undefined"]
        else:
            plot_df = bin_df


        for latent_x in range(number_latent_variables):

            for latent_y in range(latent_x, number_latent_variables):

                if latent_x == latent_y:
                    continue

                print(f"Pair plots: {latent_x:02d} vs {latent_y:02d}", end="\r")

                # pair_plot = sns.scatterplot(
                sns.scatterplot(
                    x=f"{latent_x:02d}Latent", y=f"{latent_y:02d}Latent",
                    ax=ax, data=plot_df, hue=hue, alpha=alpha
                )

                fig.savefig(
                    f"{latent_directory}/"
                    f"pair_{latent_x:02d}_{latent_y:02d}_"
                    f"{hue}_{show_undefined}.{plot_format}"
                )

                ax.clear()
        ###########################################################################
        for metric in metrics:

            print(f"Umap visualization: {metric}", end="\r")

            pair_plot = sns.scatterplot(
                x=f"{metric}_01", y=f"{metric}_02",
                ax=ax, data=plot_df, hue=hue, alpha=alpha
            )

            fig.savefig(
                f"{latent_directory}/umap_{metric}"
                f"{hue}_{show_undefined}.{plot_format}"
            )

            ax.clear()

    ###########################################################################
    print(f"Save configuration file", end="\n")

    with open(f"{latent_directory}/{config_file_name}", "w") as config_file:
        parser.write(config_file)
###############################################################################
finish_time = time.time()
print(f"\nRun time: {finish_time - start_time:.2f}")
