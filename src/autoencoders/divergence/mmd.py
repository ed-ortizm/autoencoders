import numpy as np
from scipy.stats import norm, gamma, uniform, expon, entropy

from autoencoders.divergence.distribution import Distribution

###############################################################################
class MMD(Distribution):
    """
    Compute Maximun Mean Discrepancy between samples of a distribution and a
    multivariate normal distribution
    """

    def __init__(self, number_prior_samples: int = 1000):
        """
        INPUT
            number_prior_samples: samples to draw from the multivariate normal
        """

        self.prior_samples = super().normal(number_prior_samples)

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
    def to_gaussian(
        self,
        number_samples: int,
        mu: float = 0.0,
        std: float = 1.0,
        sigma_sqrt: float = None,
    ) -> float:
        """
        Compute MMD between Normal and gaussian distribution

        INPUT
            number_samples: number of samples to draw from gaussian
                distribution
            mu: mean value of gaussian
            std: standard deviation of gaussian
            sigma_sqrt: kernel width

        OUTPUT
            Maximun Mean Discrepancy to gaussian
        """

        if sigma_sqrt == None:
            sigma_sqrt = 2.0

        in_samples = super().gaussian(
            number_samples=number_samples, mu=mu, std=std
        )

        mmd = self.compute_mmd(in_samples=in_samples, sigma_sqrt=sigma_sqrt)

        return mmd, in_samples

    ###########################################################################
    def compute_mmd(
        self, in_samples: np.array, sigma_sqrt: float = None
    ) -> float:
        """
        INPUT
            in_samples: samples from a distirubution used to compute its
                divergence with a multivariate Normal
        OUTPUTS
            Maximun Mean Discrepancy of in_samples to normal distribution
        """

        if sigma_sqrt == None:
            sigma_sqrt = 2.0

        prior_kernel = self.compute_kernel(
            self.prior_samples, self.prior_samples, sigma_sqrt
        )

        in_kernel = self.compute_kernel(in_samples, in_samples, sigma_sqrt)

        mix_kernel = self.compute_kernel(
            self.prior_samples, in_samples, sigma_sqrt
        )

        mmd = (
            np.mean(prior_kernel)
            + np.mean(in_kernel)
            - 2 * np.mean(mix_kernel)
        )

        return mmd

    ###########################################################################
    def compute_kernel(self, x, y, sigma_sqrt):

        if sigma_sqrt == None:
            sigma_sqrt = 2.0

        x_size = x.shape[0]
        y_size = y.shape[0]
        dim = 1

        tiled_x = np.tile(x.reshape(x_size, 1, dim), (1, y_size, 1))

        tiled_y = np.tile(y.reshape(1, y_size, dim), (x_size, 1, 1))

        z_diff = tiled_x - tiled_y
        kernel = np.exp(-np.mean(z_diff ** 2, axis=2) / (2 * sigma_sqrt))

        return kernel

    ###########################################################################
