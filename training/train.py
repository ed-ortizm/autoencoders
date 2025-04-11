"""Train a single AutoEncoder"""
import argparse
import os
from configparser import ConfigParser, ExtendedInterpolation
import time

import numpy as np
import tensorflow as tf

from sdss.utils.configfile import ConfigurationFile
from autoencoders.ae import AutoEncoder


def main():
    """Run AE training"""
    # Set environment variables to disable multithreading
    # as users will probably want to set the number of cores
    # to the max of their computer.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    parser = argparse.ArgumentParser(
        description="Train a VAE using config file."
        )
    parser.add_argument(
        "--config",
        type=str,
        default="train.ini",
        help="Path to config file"
        )
    args = parser.parse_args()

    config_path = args.config
    parser = ConfigParser(interpolation=ExtendedInterpolation())
    parser.read(config_path)
    seed = parser.getint("hyperparaneters", "seed", fallback=0)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    #########################################################################
    start_time = time.perf_counter()
    ########################################################################
    config_handler = ConfigurationFile()
    ########################################################################
    # set the number of cores to use during training
    cores_per_worker = parser.getint("tensorflow-session", "cores")
    tf.config.threading.set_intra_op_parallelism_threads(cores_per_worker)
    tf.config.threading.set_inter_op_parallelism_threads(cores_per_worker)
    ########################################################################
    # load data
    print("Load data")
    data_directory = parser.get("directories", "train")
    data_name = parser.get("files", "train")
    data = np.load(f"{data_directory}/{data_name}")
    # shuffle data to break any bias in spectra order if present
    np.random.shuffle(data)
    input_dimensions = data.shape[1]
    ########################################################################
    print("Build AutoEncoder")

    architecture = config_handler.section_to_dictionary(
        parser.items("architecture"), value_separators=["_"]
    )

    architecture["input_dimensions"] = input_dimensions
    # architecture["input_dimensions"] = 3000

    hyperparameters = config_handler.section_to_dictionary(
        parser.items("hyperparameters"), value_separators=[]
    )
    ########################################################################
    vae = AutoEncoder(architecture, hyperparameters)
    number_params = vae.model.count_params()
    print(f"\nThe model has {number_params} parameters")
    ########################################################################
    # Training the model
    print("Train the model")
    vae.train(data)
    del data

    save_model_to = parser.get("directories", "save_model_to")
    print(f"Save model to: {save_model_to}")
    vae.save_model(f"{save_model_to}")
    ########################################################################
    # Save reconstructed data
    save_reconstruction = parser.getboolean("files", "save_reconstruction")

    if save_reconstruction is True:

        print("Save reconstructed spectra")
        _, long_model_name = vae.get_architecture_and_model_str()

        observation_name = parser.get("files", "observation")
        observation = np.load(f"{data_directory}/{observation_name}")
        reconstruction = vae.reconstruct(observation)
        np.save(
            f"{data_directory}/reconstructions_{long_model_name}.npy",
            reconstruction,
        )

    ########################################################################
    finish_time = time.perf_counter()
    print(f"Running time: {finish_time-start_time:.2f}")
if __name__ == "__main__":

    main()
