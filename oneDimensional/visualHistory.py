#!/usr/bin/env python3.8

from configparser import ConfigParser, ExtendedInterpolation
import glob
import time

from autoencoders.ae import AutoEncoder
from autoencoders.plotAE import visual_train_history
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

    print(f"Model {idx}, bin: {bin_number}", end="\r")

    ae = AutoEncoder(reload=True, reload_from=location)

    visual_train_history(
        train_history=ae.history,
        hyperparameters=ae.hyperparameters,
        save_to = f"{save_to}/{bin_number}",
        save_format = save_format,
        slice_from=slice_from
    )

    del ae

###############################################################################
time_finish = time.time()
print(f" Run time: {time_finish - time_start: 1.0f}[s]")
