import numpy as np
from scipy.stats import norm, gamma, uniform, expon, entropy

###############################################################################
class KLD:
    """
    Compute Kullback-Leibler Divergence (KLD) between samples of a
    distribution and a normal distribution
    """

    def __init__(
        self,
        x_min: float = -10.0,
        x_max: float = 10.0,
        grid_size: int = 1000,
    ):
        """
        INPUT
            x_min: lower bound of domain for normal distribution
            x_max: upper bound of domain for normal distribution
            grid_size: number of points in domain of normal distribution
        """

        self.grid_size = grid_size
        self.x = np.linspace(x_min, x_max, grid_size)
        self.prior = norm.pdf(self.x)

    ###########################################################################
    def to_gaussian(self, mu: float = 0.0, std: float = 1.0) -> float:
        """
        Compute KLD between Normal and gaussian distribution

        INPUT
            mu: mena value of gaussian
            std: standard deviation of gaussian

        OUTPUT
            kld: KLD to gaussian
        """
        Q = norm.pdf(self.x, loc=mu, scale=std)

        kld = entropy(self.prior, Q)

        return kld, Q

    ###########################################################################
    def to_exponential(self, parameters: dict) -> float:
        """
        Compute KLD between Normal and exponential distribution

        INPUT
            parameters: parameters of exponential distribution

        OUTPUT
            KLD to exponential
        """
        Q = expon.pdf(
            self.x, loc=parameters["location"], scale=parameters["scale"]
        )

        kld = entropy(self.prior, Q)

        return kld

    ###########################################################################
    def to_gamma(self, parameters: dict) -> float:
        """
        Compute KLD between Normal and gamma distribution

        INPUT
            parameters: parameters of gamma distribution

        OUTPUT
            KLD to gamma
        """
        Q = gamma.pdf(
            self.x,
            loc=parameters["location"],
            scale=parameters["scale"],
            a=parameters["a"],
        )

        kld = entropy(self.prior, Q)

        return kld

    ###########################################################################
    def to_uniform(self, parameters: dict) -> float:
        """
        Compute KLD between Normal and uniform distribution

        INPUT
            parameters: parameters of uniform distribution

        OUTPUT
            KLD to uniform
        """

        Q = uniform.pdf(
            self.x, loc=parameters["low"], scale=parameters["scale"]
        )

        kld = entropy(self.prior, Q)

        return kld
