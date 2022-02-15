import numpy as np

from sdss.superclasses import ConfigurationFile
# Compute KLD
# KLD = -0.5 * np.mean(z_log_var - z_mean**2 - np.exp(z_log_var) + 1)


###############################################################################
class MMDtoNormal(ConfigurationFile):
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
    def uniform(self, number_samples: int, parameters: tuple) -> float:
        """
        Compute mmd between an uniform and a Normal
        distribution

        INPUT
            number_samples: number of samples for the multivarite
                gaussian
            parameters: parameters of uniform distribution
                ("low" = "0", "high" = "1", "dimension" = "4")

        OUTPUT
            MMD between the distributions
        """

        parameters = super().section_to_dictionary(
            parameters, value_separators = [","]
        )

        in_samples = np.random.uniform(
            parameters["low"], parameters["high"],
            size=(number_samples, parameters["dimension"])
        )

        return self.compute_mmd(in_samples)
    ###########################################################################
    def gaussian(self, number_samples: int, parameters: tuple) -> float:
        """
        Compute mmd between a multivariate gaussian and a Normal
        distribution

        INPUT
            number_samples: number of samples for the multivarite
                gaussian
            parameters: parameters of multivariate gaussian
                ("dimension" = "6", "mean" = "1, 2", "covariance" = "4, 2")

        OUTPUT
            MMD between the distributions
        """

        parameters = super().section_to_dictionary(
            parameters, value_separators = [","]
        )

        mean = np.array(parameters["mean"])
        assert mean.size == parameters["dimension"]

        covariance = np.diag(parameters["covariance"])

        in_samples = np.random.multivariate_normal(
            mean, covariance, size=number_samples
        )

        return self.compute_mmd(in_samples)
    ###########################################################################
    def compute_mmd(
        self, in_samples: np.array, sigma_sqr: float = None
    ) -> float:
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
    def _prior(self, number_samples: int, dimension: int) -> np.array:

        return np.random.normal(size=(number_samples, dimension))

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
