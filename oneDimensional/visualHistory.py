#!/usr/bin/env python3.8

from configparser import ConfigParser, ExtendedInterpolation
import glob
import time
import pickle

from autoencoders.plotAE import visual_history
from sdss.superclasses import FileDirectory

###############################################################################
time_start = time.time()

parser = ConfigParser(interpolation=ExtendedInterpolation())
parser.read("visualHistory.ini")
###############################################################################
# get relevant directories
models_directory = parser.get("directory", "models")

models_directories = glob.glob(f"{models_directory}/*")
###############################################################################
slice_from = parser.getint("configuration", "slice_from")
save_to = parser.get("directory", "save_to")
save_format = parser.get("file", "save_format")

for idx, location in enumerate(models_directories):

    ###########################################################################
    print(f"Plot history of model {idx+1}", end="\r")

    file_location = f"{location}/train_history.pkl"

    with open(file_location, "rb") as file:
        parameters = pickle.load(file)

    [_, hyperparameters, history] = parameters

    visual_history(
        history=history,
        hyperparameters=hyperparameters,
        save_to=f"{save_to}",
        save_format=save_format,
        slice_from=slice_from,
    )

###############################################################################
time_finish = time.time()
print(f" Run time: {time_finish - time_start: 1.0f}[s]")
