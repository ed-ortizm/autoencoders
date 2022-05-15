###############################################################################
from configparser import ConfigParser, ExtendedInterpolation
import time

###############################################################################
import numpy as np

from sdss.superclasses import ConfigurationFile
from autoencoders.divergence.kld import KLD

###############################################################################
start_time = time.time()

parser = ConfigParser(interpolation=ExtendedInterpolation())
parser.read("kld.ini")
###############################################################################
start = parser.getfloat("prior", "start")
end = parser.getfloat("prior", "end")
grid_size = parser.getint("prior", "grid_size")

kld = KLD(start, end, grid_size)

for distribution in ["gaussian", "uniform", "gamma", "exponential"]:

    parameters = ConfigurationFile().section_to_dictionary(
        parser.items(distribution), value_separators=[","]
    )

    if distribution == "gaussian":

        divergence = kld.to_gaussian(
            mu=parameters["mean"], std=parameters["standard_deviation"]
        )

        print(divergence)

    elif distribution == "uniform":

        divergence = kld.to_uniform(parameters)

        print(divergence)

    elif distribution == "gamma":

        divergence = kld.to_gamma(parameters)

        print(divergence)

    elif distribution == "exponential":

        divergence = kld.to_exponential(parameters)

        print(divergence)

###############################################################################
finish_time = time.time()
print(f"Run time: {finish_time-start_time:.2f}")
