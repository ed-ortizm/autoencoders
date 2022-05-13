"""Do scatter plots of latent variables, including umap representation"""
from configparser import ConfigParser, ExtendedInterpolation
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
print(f"Load latent representation", end="\n")

latent_directory = parser.get("directory", "latent")

latent_name = parser.get("file", "latent")
FileDirectory().file_exists(
    f"{latent_directory}/{latent_name}",
    exit_program=True
)

latent = np.load(f"{latent_directory}/{latent_name}")
number_variables = latent.shape[1]

for idx in range(number_variables):
    bin_df[f"{idx:02d}Latent"] = latent[:, idx]

###############################################################################
print(f"Load umap embedding", end="\n")
metrics = config.entry_to_list(parser.get("umap", "metrics"), str, ",")
bin_id = parser.get("common", "bin")

for metric in metrics:

    embedding = np.load(f"{latent_directory}/umap_{metric}_{bin_id}.npy")

    bin_df[f"{metric}_01"] = embedding[:, 0]
    bin_df[f"{metric}_02"] = embedding[:, 1]


bin_df.dropna(inplace=True)
# print(bin_df.columns, bin_df.shape)
print(f"Save pair plots of latent representation", end="\n")

size = ConfigurationFile().entry_to_list(
    parser.get("plot", "size"), float, ","
)
size = tuple(size)

fig, ax = plt.subplots(figsize=size, tight_layout=True)

hue = parser.get("plot", "hue")
alpha = parser.getfloat("plot", "alpha")
plot_format = parser.get("plot", "format")

# create new set of classes
bin_df["ABSSB"] = bin_df["subClass"]

for idx in bin_df.index:

    if "STARF" in bin_df.loc[idx, "ABSSB"]:
        bin_df.loc[idx, "ABSSB"] = "STARFORMING"

    elif "STARB" in bin_df.loc[idx, "ABSSB"]:
        bin_df.loc[idx, "ABSSB"] = "STARBURST"

    elif "AGN" in bin_df.loc[idx, "ABSSB"]:
        bin_df.loc[idx, "ABSSB"] = "AGN"

for latent_x in range(number_variables):

    for latent_y in range(latent_x, number_variables):

        if latent_x == latent_y:
            continue

        print(f"Pair plots: {latent_x:02d} vs {latent_y:02d}", end="\r")

        # pair_plot = sns.scatterplot(
        sns.scatterplot(
            x=f"{latent_x:02d}Latent", y=f"{latent_y:02d}Latent",
            ax=ax, data=bin_df, hue=hue, alpha=alpha
        )

        fig.savefig(
            f"{latent_directory}/"
            f"pair_{latent_x:02d}_{latent_y:02d}.{plot_format}"
        )

        ax.clear()
###############################################################################
for metric in metrics:

    print(f"Umap visualization: {metric}", end="\r")

    pair_plot = sns.scatterplot(
        x=f"{metric}_01", y=f"{metric}_02",
        ax=ax, data=bin_df, hue=hue, alpha=alpha
    )

    fig.savefig(f"{latent_directory}/umap_{metric}.{plot_format}")

    ax.clear()

###############################################################################
print(f"Save configuration file", end="\n")

with open(f"{latent_directory}/{config_file_name}", "w") as config_file:
    parser.write(config_file)
###############################################################################
finish_time = time.time()
print(f"\nRun time: {finish_time - start_time:.2f}")
