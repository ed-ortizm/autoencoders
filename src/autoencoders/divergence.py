import numpy as np

# Compute KLD
# KLD = -0.5 * np.mean(z_log_var - z_mean**2 - np.exp(z_log_var) + 1)


###############################################################################
class Distribution:
    """
    Class with distributions to use with MMD and KLD to Normal distribution
    """

    def __init__(self):
        pass
    ###########################################################################
    def poisson(self, number_samples: int, parameters: dict) -> np.array:
        """
        Sample from poisson distribution

        INPUT
            number_samples: number of samples to draw
            parameters: parameters of poisson distribution
            {"lambda": [2, 5, 45]}

        OUTPUT
            array with samples from distribution
        """


        # make sure dimension is properly set
        dimension = len(parameters["lambda"])

        samples = np.random.poisson(
            parameters["lambda"],
            size=(number_samples, dimension)
        )

        return samples
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
            parameters["scale"],
            size=(number_samples, dimension)
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
            parameters["shape"], parameters["scale"],
            size=(number_samples, dimension)
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
            parameters["low"], parameters["high"],
            size=(number_samples, dimension)
        )

        return samples
    ###########################################################################
    def gaussian(self, number_samples: int, parameters: dict) -> np.array:
        """
        Sample from gaussian distribution

        INPUT
            number_samples: number of samples to draw
            parameters: parameters of gaussian
                {"mean": [1, 2], "covariance": [4, 2]}

        OUTPUT
            array with samples from distribution
        """

        mean = np.array(parameters["mean"])

        covariance = np.diag(parameters["covariance"])

        # make sure dimensions are properly set
        assert mean.size == covariance.shape[0]

        samples = np.random.multivariate_normal(
            mean, covariance, size=number_samples
        )

        return samples
    ###########################################################################
    def normal(self, number_samples: int, dimension: int) -> np.array:
        """
        Samples from normal distribution

        INPUT
            number_samples: number of samples to draw
            dimension: dimension of the sampled vector

        OUTPUT
            array with samples from distribution
        """

        return np.random.normal(size=(number_samples, dimension))

###############################################################################
class MMD(Distribution):
    """
    Compute Maximun Mean Discrepancy between samples of a distribution and a
    multivariate normal distribution
    """

    def __init__(self, prior_samples: int = 200):
        """
        INPUT
            prior_samples: samples to draw from the multivariate normal
        """
        self.prior_samples = prior_samples

    ###########################################################################
    def compute_mmd(
        self, in_samples: np.array, sigma_sqr: np.array = None
    ) -> np.array:
        """
        INPUT
            in_samples: samples from a distirubution used to compute its
                divergence with a multivariate Normal
            sigma_sqrt: dispersion factor when computing kernels
        OUTPUTS
        """

        dim = in_samples.shape[1]

        if sigma_sqr == None:
            sigma_sqr = 2 / dim

        prior_samples = self._prior(self.prior_samples, dim)

        prior_kernel = self.compute_kernel(
            prior_samples, prior_samples, sigma_sqr
        )

        in_kernel = self.compute_kernel(in_samples, in_samples, sigma_sqr)

        mix_kernel = self.compute_kernel(prior_samples, in_samples, sigma_sqr)

        mmd = (
            np.mean(prior_kernel)
            + np.mean(in_kernel)
            - 2 * np.mean(mix_kernel)
        )

        return mmd

    ###########################################################################
    def compute_kernel(self, x, y, sigma_sqr):

        x_size = x.shape[0]
        y_size = y.shape[0]
        dim = x.shape[1]

        tiled_x = np.tile(x.reshape(x_size, 1, dim), (1, y_size, 1))

        tiled_y = np.tile(y.reshape(1, y_size, dim), (x_size, 1, 1))

        z_diff = tiled_x - tiled_y
        kernel = np.exp(-np.mean(z_diff ** 2, axis=2) / (2 * sigma_sqr))

        return kernel

    ###########################################################################
