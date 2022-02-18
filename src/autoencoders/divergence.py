import numpy as np
import scipy.stats.entropy

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
# E(z) = mean + logvar * e(z^2)
# P = e^(z^2)
# KLD(P, E) = E*log(P/E)
# (mean + logvar * e(z^2))*log(e^(z^2))
#  (mean + logvar * e(z^2)) * log ((mean + logvar * e(z^2)))
# (mean + logvar * e(z^2)) (1-log ((mean + logvar * e(z^2))))
# Compute KLD
# KLD = -0.5 * np.mean(z_log_var - z_mean**2 - np.exp(z_log_var) + 1)
class MMD(Distribution):
    """
    Compute Maximun Mean Discrepancy between samples of a distribution and a
    multivariate normal distribution
    """

    def __init__(self, prior_samples: int = 200, sigma_sqr: float=None):
        """
        INPUT
            prior_samples: samples to draw from the multivariate normal
            sigma_sqr: kernel width
        """
        self.prior_samples = prior_samples
        self.sigma_sqr = sigma_sqr

    ###########################################################################
    def to_gaussian(self, number_samples: int, parameters: dict) -> float:
        """
        Compute MMD between Normal and gaussian distribution

        INPUT
            number_samples: number of samples to draw from gaussian
                distribution
            parameters: parameters of gaussian distribution

        OUTPUT
            Maximun Mean Discrepancy to gaussian
        """

        in_samples = super().gaussian(number_samples, parameters)

        mmd = self.compute_mmd(in_samples)

        return mmd
    ###########################################################################
    def to_exponential(self, number_samples: int, parameters: dict) -> float:
        """
        Compute MMD between Normal and exponential distribution

        INPUT
            number_samples: number of samples to draw from exponential
                distribution
            parameters: parameters of exponential distribution

        OUTPUT
            Maximun Mean Discrepancy to exponential
        """

        in_samples = super().exponential(number_samples, parameters)

        mmd = self.compute_mmd(in_samples)

        return mmd
    ###########################################################################
    def to_gamma(self, number_samples: int, parameters: dict) -> float:
        """
        Compute MMD between Normal and gamma distribution

        INPUT
            number_samples: number of samples to draw from gamma
                distribution
            parameters: parameters of gamma distribution

        OUTPUT
            Maximun Mean Discrepancy to gamma
        """

        in_samples = super().gamma(number_samples, parameters)

        mmd = self.compute_mmd(in_samples)

        return mmd
    ###########################################################################
    def to_poisson(self, number_samples: int, parameters: dict) -> float:
        """
        Compute MMD between Normal and poisson distribution

        INPUT
            number_samples: number of samples to draw from poisson
                distribution
            parameters: parameters of poisson distribution

        OUTPUT
            Maximun Mean Discrepancy to poisson
        """

        in_samples = super().poisson(number_samples, parameters)

        mmd = self.compute_mmd(in_samples)

        return mmd
    ###########################################################################
    def to_uniform(self, number_samples: int, parameters: dict) -> float:
        """
        Compute MMD between Normal and uniform distribution

        INPUT
            number_samples: number of samples to draw from uniform
                distribution
            parameters: parameters of uniform distribution

        OUTPUT
            Maximun Mean Discrepancy to uniform
        """

        in_samples = super().uniform(number_samples, parameters)

        mmd = self.compute_mmd(in_samples)

        return mmd
    ###########################################################################
    def compute_kld(self, in_samples: np.array) -> float:
        """
        INPUT
            in_samples: samples from a distirubution used to compute its
                divergence with a multivariate Normal
        OUTPUTS
            KL divergence of in_samples to normal distribution
        """

        dim = in_samples.shape[1]

        prior_samples = super().normal(self.prior_samples, dim)

        kld = scipy.stats.entropy(in_samples, prior_samples)

        return kld

###############################################################################
class MMD(Distribution):
    """
    Compute Maximun Mean Discrepancy between samples of a distribution and a
    multivariate normal distribution
    """

    def __init__(self, prior_samples: int = 200, sigma_sqr: float=None):
        """
        INPUT
            prior_samples: samples to draw from the multivariate normal
            sigma_sqr: kernel width
        """
        self.prior_samples = prior_samples
        self.sigma_sqr = sigma_sqr

    ###########################################################################
    def to_gaussian(self, number_samples: int, parameters: dict) -> float:
        """
        Compute MMD between Normal and gaussian distribution

        INPUT
            number_samples: number of samples to draw from gaussian
                distribution
            parameters: parameters of gaussian distribution

        OUTPUT
            Maximun Mean Discrepancy to gaussian
        """

        in_samples = super().gaussian(number_samples, parameters)

        mmd = self.compute_mmd(in_samples)

        return mmd
    ###########################################################################
    def to_exponential(self, number_samples: int, parameters: dict) -> float:
        """
        Compute MMD between Normal and exponential distribution

        INPUT
            number_samples: number of samples to draw from exponential
                distribution
            parameters: parameters of exponential distribution

        OUTPUT
            Maximun Mean Discrepancy to exponential
        """

        in_samples = super().exponential(number_samples, parameters)

        mmd = self.compute_mmd(in_samples)

        return mmd
    ###########################################################################
    def to_gamma(self, number_samples: int, parameters: dict) -> float:
        """
        Compute MMD between Normal and gamma distribution

        INPUT
            number_samples: number of samples to draw from gamma
                distribution
            parameters: parameters of gamma distribution

        OUTPUT
            Maximun Mean Discrepancy to gamma
        """

        in_samples = super().gamma(number_samples, parameters)

        mmd = self.compute_mmd(in_samples)

        return mmd
    ###########################################################################
    def to_poisson(self, number_samples: int, parameters: dict) -> float:
        """
        Compute MMD between Normal and poisson distribution

        INPUT
            number_samples: number of samples to draw from poisson
                distribution
            parameters: parameters of poisson distribution

        OUTPUT
            Maximun Mean Discrepancy to poisson
        """

        in_samples = super().poisson(number_samples, parameters)

        mmd = self.compute_mmd(in_samples)

        return mmd
    ###########################################################################
    def to_uniform(self, number_samples: int, parameters: dict) -> float:
        """
        Compute MMD between Normal and uniform distribution

        INPUT
            number_samples: number of samples to draw from uniform
                distribution
            parameters: parameters of uniform distribution

        OUTPUT
            Maximun Mean Discrepancy to uniform
        """

        in_samples = super().uniform(number_samples, parameters)

        mmd = self.compute_mmd(in_samples)

        return mmd
    ###########################################################################
    def compute_mmd(self, in_samples: np.array) -> float:
        """
        INPUT
            in_samples: samples from a distirubution used to compute its
                divergence with a multivariate Normal
        OUTPUTS
            Maximun Mean Discrepancy of in_samples to normal distribution
        """

        dim = in_samples.shape[1]

        if self.sigma_sqr == None:
            sigma_sqr = 2 / dim

        prior_samples = super().normal(self.prior_samples, dim)

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
