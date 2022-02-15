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

if distribution == "multivariate-normal":

    parameters = parser.items("multivariate-normal")
    parameters = ConfigurationFile().section_to_dictionary(
        parameters, value_separators = [","]
    )
    mean = np.array(parameters["mean"])
    assert mean.size == parameters["dimension"]

    covariance = np.diag(parameters["covariance"])

    in_samples = np.random.multivariate_normal(
        mean, covariance, size=number_samples
    )

    print(in_samples.shape)

# compute_mmd(true_samples, z)
# cov=np.diag([1,1])
# In [17]: tata = np.random.multivariate_normal(mean=[0, 1], cov=[[1,0],[0,1]], size=(2, 10_000))
#
# In [18]: tata.shape
# Out[18]: (2, 10000, 2)
#
# In [19]: x, y = np.random.multivariate_normal(mean=[0, 1], cov=[[1,0],[0,1]], size=10_000).T
#
# In [20]: x.shape
# Out[20]: (10000,)
###############################################################################
finish_time = time.time()
print(f"Run time: {finish_time-start_time:.2f}")
