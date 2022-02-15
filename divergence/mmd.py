###############################################################################
from configparser import ConfigParser, ExtendedInterpolation
import time

###############################################################################
import numpy as np

from sdss.superclasses import ConfigurationFile
from autoencoders.divergence import MMDtoNormal

###############################################################################
start_time = time.time()

parser = ConfigParser(interpolation=ExtendedInterpolation())
parser.read("mmd.ini")
###############################################################################
prior_samples = parser.getint("common", "prior_samples")

mmd = MMDtoNormal(prior_samples)

number_samples = parser.getint("common", "samples")

distribution = parser.get("common", "distribution")

parameters = ConfigurationFile().section_to_dictionary(
    parser.items(distribution), value_separators = [","]
)
if distribution == "gaussian":

    divergence = mmd.gaussian(number_samples,parameters)

    print(divergence)

elif distribution == "uniform":

    divergence = mmd.uniform(number_samples,parameters)

    print(divergence)

elif distribution == "gamma":

    divergence = mmd.gamma(number_samples,parameters)

    print(divergence)

elif distribution == "exponential":

    divergence = mmd.gamma(number_samples,parameters)

    print(divergence)
###############################################################################
finish_time = time.time()
print(f"Run time: {finish_time-start_time:.2f}")
