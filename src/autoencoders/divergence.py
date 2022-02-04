import numpy as np
# Compute KLD
KLD = -0.5 * np.mean(z_log_var - z_mean**2 - np.exp(z_log_var) + 1)


###############################################################################
class MMD:

    def __init__(self,):
        pass
    ###########################################################################
    def prior(self, number_samples, dim):

        return np.random.normal(200, dim)
    ###########################################################################

    ###########################################################################
    def compute_kernel(self, x, y, sigma_sqr):

        x_size = x.shape[0]
        y_size = y.shape[0]
        dim = x.shape[1]

        tiled_x = np.tile(x.reshape(x_size, 1, dim), (1, y_size, 1))

        tiled_y = np.tile(y.reshape(1, y_size, dim), (x_size, 1, 1))

        z_diff = tiled_x - tiled_y
        kernel = np.exp(-np.mean(z_diff**2, axis=2) / (2 * sigma_sqr))

        return kernel

    ###########################################################################
    def compute_mmd(self,
        in_samples: np.array,
        sigma_sqr: float, # 2/dim
        n_samples_from_prior: int = 200,
        ) -> float:

        dim = in_samples.sahpe[1]
        prior_samples = self.prior(n_samples_from_prior, dim)

        prior_kernel = self.compute_kernel(
            prior_samples, prior_samples, sigma_sqr
        )

        in_kernel = self.compute_kernel(in_samples, in_samples, sigma_sqr)

        prior_in_kernel = self.compute_kernel(
            prior_samples, in_samples, sigma_sqr
        )

        mmd = (
            np.mean(prior_kernel) + np.mean(in_kernel)
            - 2 * np.mean(prior_in_kernel)
        )

        return mmd
