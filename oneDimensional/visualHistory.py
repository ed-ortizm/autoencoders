from configparser import ConfigParser, ExtendedInterpolation
import glob
import time
import pickle

from autoencoders.plotAE import visual_history
from sdss.superclasses import FileDirectory

###############################################################################
time_start = time.time()

parser = ConfigParser(interpolation=ExtendedInterpolation())
parser_name = "visualHistory.ini"
parser.read(f"{parser_name}")
###############################################################################
# get relevant directories
models_directory = parser.get("directory", "models")

models_directories = glob.glob(f"{models_directory}/*/")
###############################################################################
slice_from = parser.getint("configuration", "slice_from")

save_to = parser.get("directory", "save_to")
save_format = parser.get("file", "save_format")

for idx, location in enumerate(models_directories):
    ###########################################################################
    model_id = location.split("/")[-2]
    print(f"[{model_id}] Plot history of model {idx+1}", end="\r")

    file_location = f"{location}/train_history.pkl"

    with open(file_location, "rb") as file:
        parameters = pickle.load(file)

    [_, hyperparameters, history] = parameters

    visual_history(
        model_id,
        history=history,
        hyperparameters=hyperparameters,
        save_to=f"{save_to}",
        save_format=save_format,
        slice_from=slice_from,
    )

##############################################################################
with open(f"{save_to}/{parser_name}", "w") as file:

    parser.write(file)

models_config = glob.glob(f"{models_directory}/*.ini")[0]
configuration_file = ConfigParser(interpolation=ExtendedInterpolation())
configuration_file.read(models_config)
config_name = models_config.split("/")[-1]

with open(f"{save_to}/{config_name}", "w") as file:

    configuration_file.write(file)

###############################################################################
time_finish = time.time()
print(f"\nRun time: {time_finish - time_start: 1.0f}[s]")
