"""Explode MMD divergence"""
from configparser import ConfigParser, ExtendedInterpolation
import time

###############################################################################

from sdss.utils.configfile import ConfigurationFile
from autoencoders.divergence.mmd import MMD

###############################################################################
start_time = time.time()

parser = ConfigParser(interpolation=ExtendedInterpolation())
parser.read("mmd.ini")
###############################################################################
prior_samples = parser.getint("common", "prior_samples")

mmd = MMD(prior_samples)

number_samples = parser.getint("common", "samples")

for distribution in ["gaussian", "uniform", "gamma", "exponential"]:

    parameters = ConfigurationFile().section_to_dictionary(
        parser.items(distribution), value_separators=[","]
    )

    if distribution == "gaussian":

        divergence = mmd.to_gaussian(number_samples, parameters)

        print(divergence)

    elif distribution == "uniform":

        divergence = mmd.to_uniform(number_samples, parameters)

        print(divergence)

    elif distribution == "gamma":

        divergence = mmd.to_gamma(number_samples, parameters)

        print(divergence)

    elif distribution == "exponential":

        divergence = mmd.to_exponential(number_samples, parameters)

        print(divergence)

###############################################################################
finish_time = time.time()
print(f"Run time: {finish_time-start_time:.2f}")
