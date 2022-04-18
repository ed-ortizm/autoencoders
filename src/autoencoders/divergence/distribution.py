import numpy as np
from scipy.stats import norm, gamma, uniform, expon, entropy

###############################################################################
class Distribution:
    """
    Class with distributions to use with MMD and KLD to Normal distribution
    """

    def __init__(self):
        pass

    ###########################################################################
    def exponential(self, number_samples: int, parameters: dict) -> np.array:
        """
        Sample from exponential distribution

        INPUT
            number_samples: number of samples to draw
            parameters: parameters of exponential distribution
                {"scale": [1, 1]}

        OUTPUT
            array with samples from distribution
        """

        # make sure dimension is properly set
        dimension = len(parameters["scale"])

        samples = np.random.exponential(
            parameters["scale"], size=(number_samples, dimension)
        )

        return samples

    ###########################################################################
    def gamma(self, number_samples: int, parameters: dict) -> np.array:
        """
        Sample from gamma distribution

        INPUT
            number_samples: number of samples to draw
            parameters: parameters of gamma distribution
                {"shape": [1, 1], "scale": [1, 1]}

        OUTPUT
            array with samples from distribution
        """

        # make sure dimension is properly set
        dimension = len(parameters["shape"])

        assert len(parameters["scale"]) == dimension

        samples = np.random.gamma(
            parameters["shape"],
            parameters["scale"],
            size=(number_samples, dimension),
        )

        return samples

    ###########################################################################
    def uniform(self, number_samples: int, parameters: dict) -> np.array:
        """
        Sample from uniform distribution

        INPUT
            number_samples: number of samples to draw
            parameters: parameters of uniform distribution
                {"low": [0, 6], "high": [1 , 20]}

        OUTPUT
            array with samples from distribution
        """

        # make sure dimension is properly set
        dimension = len(parameters["high"])

        assert len(parameters["low"]) == dimension

        samples = np.random.uniform(
            parameters["low"],
            parameters["high"],
            size=(number_samples, dimension),
        )

        return samples

    ###########################################################################
    def gaussian(self, number_samples: int, mu:float=0., std:float=1.) -> np.array:
        """
        Sample from gaussian distribution

        INPUT
            number_samples: number of samples to draw
            mu: mean value of gaussian
            std: standard deviation of gaussian

        OUTPUT
            array with samples from distribution
        """

        # samples = np.random.multivariate_normal(
        #     mean, covariance, size=number_samples
        # )
        
        samples = np.random.normal(
            loc=mu, scale=std, size=(number_samples,1)
        )

        return samples

    ###########################################################################
    def normal(self, number_samples: int) -> np.array:
        """
        Samples from normal distribution

        INPUT
            number_samples: number of samples to draw

        OUTPUT
            array with samples from distribution
        """

        return np.random.normal(size=(number_samples, 1))
