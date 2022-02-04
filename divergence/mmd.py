###############################################################################
from configparser import ConfigParser, ExtendedInterpolation
import time

###############################################################################
import numpy as np

from autoencoders.divergence import MMD

###############################################################################
start_time = time.time()

config_handler = ConfigurationFile()
parser = ConfigParser(interpolation=ExtendedInterpolation())
parser.read("mmd.ini")
###############################################################################

mmd = MMD()
# compute_mmd(true_samples, z)

###############################################################################
finish_time = time.time()
print(f"Run time: {finish_time-start_time:.2f}")
