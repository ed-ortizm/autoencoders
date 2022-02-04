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
input_directory = parser.get("directory", "input")
model_type = parser.get("common", "model")
model_architecture = parser.get("common", "architecture")

model_locations = glob.glob(
    f"{input_directory}/bin_*/bin_*/{model_type}/{model_architecture}/*"
)
###############################################################################
slice_from = parser.getint("configuration", "slice_from")
save_to = parser.get("directory", "save_to")
save_format = parser.get("file", "save_format")

for idx, location in enumerate(model_locations):

    bin_number = location.split("/")[-4]
    ###########################################################################
    print(f"Model {idx}, bin: {bin_number}", end="\r")
    file_location = f"{location}/train_history.pkl"

    with open(file_location, "rb") as file:
        parameters = pickle.load(file)

    [architecture, hyperparameters, history] = parameters

    visual_history(
        history=history,
        hyperparameters=hyperparameters,
        save_to=f"{save_to}/{bin_number}",
        save_format=save_format,
        slice_from=slice_from,
    )

###############################################################################
time_finish = time.time()
print(f" Run time: {time_finish - time_start: 1.0f}[s]")
