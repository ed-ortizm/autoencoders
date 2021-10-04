from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chisquare
from sklearn.decomposition import PCA

################################################################################
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Lambda, Dropout
from tensorflow.keras.losses import mse
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

################################################################################
class Outlier:
    """
    Class for dealing with the outliers based on a generative model trained with
    tensorflow.keras
    """

    def __init__(
        self,
        model_path,
        o_scores_path=".",
        metric="mse",
        p="p",
        custom=False,
        custom_metric=None,
    ):
        """
        Init fucntion

        Args:
            model_path: path where the trained generative model is located

            o_scores_path: (str) path where the numpy arrays with the outlier scores
                is located. Its functionality is for the cases where the class
                will be implemented several times, offering therefore the
                possibility to load the scores instead of computing them over
                and over.

            metric: (str) the name of the metric used to compute the outlier score
                using the observed spectrum and its reconstruction. Possible

            p: (float > 0) in case the metric is the lp metric, p needs to be a non null
                possitive float [Aggarwal 2001]
        """

        self.model_path = model_path
        self.o_scores_path = o_scores_path
        self.metric = metric
        self.p = p
        self.custom = custom
        if self.custom:
            self.custom_metric = custom_metric

    def _get_OR(self, O, model):

        if len(O.shape) == 1:
            O = O.reshape(1, -1)

        R = model.predict(O)

        return O, R

    def score(self, O):
        """
        Computes the outlier score according to the metric used to instantiate
        the class.

        Args:
            O: (2D np.array) with the original objects where index 0 indicates
            the object and index 1 the features of the object.

        Returns:
            A one dimensional numpy array with the outlier scores for objects
            present in O
        """

        model_name = self.model_path.split("/")[-1]
        print(f"Loading model: {model_name}")
        model = load_model(f"{self.model_path}")

        O, R = self._get_OR(O, model)

        if self.custom:
            print(f"Computing the predictions of {model_name}")
            return self.user_metric(O=O, R=R)

        elif self.metric == "mse":
            print(f"Computing the predictions of {model_name}")
            return self._mse(O=O, R=R)

        elif self.metric == "chi2":
            print(f"Computing the predictions of {model_name}")
            return self._chi2(O=O, R=R)

        elif self.metric == "mad":
            print(f"Computing the predictions of {model_name}")
            return self._mad(O=O, R=R)

        elif self.metric == "lp":

            if self.p == "p" or self.p <= 0:
                print(f"For the {self.metric} metric you need p")
                return None

            print(f"Computing the predictions of {model_name}")
            return self._lp(O=O, R=R)

        else:
            print(f"The provided metric: {self.metric} is not implemented yet")
            return None

    def _coscine_similarity(self, O, R):
        """
        Computes the coscine similarity between the reconstruction of the input
        objects

        Args:
            O: (2D np.array) with the original objects where index 0 denotes
                indicates the objec and index 1 the features of the object.

            R: Reconstruction of O by (tensorflow.keras model) the generative
            model

        Returns:
            A one dimensional numpy array with the cosine similarity between
            objects O and their reconstructiob
        """

        pass

    def _jaccard_index(self, O, R):
        """
        Computes the mean square error for the reconstruction of the input
        objects

        Args:
            O: (2D np.array) with the original objects where index 0 denotes
                indicates the objec and index 1 the features of the object.

            R: Reconstruction of O by (tensorflow.keras model) the generative
            model

        Returns:
            A one dimensional numpy array with the mean square error for objects
            present in O
        """

        pass

    def _sorensen_dice_index(self, O, R):
        """
        Computes the mean square error for the reconstruction of the input
        objects

        Args:
            O: (2D np.array) with the original objects where index 0 denotes
                indicates the objec and index 1 the features of the object.

            R: Reconstruction of O by (tensorflow.keras model) the generative
            model

        Returns:
            A one dimensional numpy array with the mean square error for objects
            present in O
        """
        pass

    # Mahalanobis, Canberra, Braycurtis, and KL-divergence
    def _mse(self, O, R):
        """
        Computes the mean square error for the reconstruction of the input
        objects

        Args:
            O: (2D np.array) with the original objects where index 0 denotes
                indicates the objec and index 1 the features of the object.

            R: Reconstruction of O by (tensorflow.keras model) the generative
            model

        Returns:
            A one dimensional numpy array with the mean square error for objects
            present in O
        """

        return np.square(R - O).mean(axis=1)

    def _chi2(self, O, R):
        """
        Computes the chi square error for the reconstruction of the input
        objects

        Args:
            O: (2D np.array) with the original objects where index 0 denotes
                indicates the objec and index 1 the features of the object.

            R: Reconstruction of O by (tensorflow.keras model) the generative
            model

        Returns:
            A one dimensional numpy array with the chi square error for objects
            present in O
        """

        return (np.square(R - O) * (1 / np.abs(R))).mean(axis=1)

    def _mad(self, O, R):
        """
        Computes the maximum absolute deviation from the reconstruction of the
        input objects

        Args:
            O: (2D np.array) with the original objects where index 0 denotes
                indicates the objec and index 1 the features of the object.

            R: Reconstruction of O by (tensorflow.keras model) the generative
            model

        Returns:
            A one dimensional numpy array with the maximum absolute deviation
            from the objects present in O
        """

        return np.abs(R - O).mean(axis=1)

    def _lp(self, O, R):
        """
        Computes the lp distance from the reconstruction of the input objects

        Args:
            O: (2D np.array) with the original objects where index 0 denotes
                indicates the objec and index 1 the features of the object.

            R: Reconstruction of O by (tensorflow.keras model) the generative
            model

        Returns:
            A one dimensional numpy array with the lp distance from the objects
            present in O
        """

        return (np.sum((np.abs(R - O)) ** self.p, axis=1)) ** (1 / self.p)

    # gotta code conditionals to make sure that the user inputs a "good one"
    def user_metric(self, custom_metric, O, R):
        """
        Computes the custom metric for the reconstruction of the input objects
        as defined by the user

        Args:
            O: (2D np.array) with the original objects where index 0 denotes
                indicates the objec and index 1 the features of the object.

            R: Reconstruction of O by (tensorflow.keras model) the generative
            model

        Returns:
            A one dimensional numpy array with the score produced by the user
            defiend metric of objects present in O
        """

        return self.custom_metric(O, R)

    def metadata(self, spec_idx, training_data_files):
        """
        Generates the names and paths of the individual objects used to create
        the training data set.
        Note: this work according to the way the training data set was created

        Args:
            spec_idx: (int > 0) the location index of the spectrum in the
                training data set.

            training_data_files: (list of strs) a list with the paths of the
                individual objects used to create the training data set.

        Returns:
            sdss_name, sdss_name_path: (str, str) the sdss name of the objec,
                the path of the object in the files system
        """

        # print('Gathering name of data points used for training')

        sdss_names = [
            name.split("/")[-1].split(".")[0] for name in training_data_files
        ]

        # print('Retrieving the sdss name of the desired spectrum')

        sdss_name = sdss_names[spec_idx]
        sdss_name_path = training_data_files[spec_idx]

        return sdss_name, sdss_name_path

    def top_reconstructions(self, O, n_top_spectra):
        """
        Selects the most normal and outlying objecs

        Args:
            O: (2D np.array) with the original objects where index 0 denotes
                indicates the objec and index 1 the features of the object.

            n_top_spectra: (int > 0) this parameter controls the number of
                objects identifiers to return for the top reconstruction,
                that is, the idices for the most oulying and the most normal
                objects.

        Returns:
            most_normal, most_oulying: (1D np.array, 1D np.array) numpy arrays
                with the location indexes of the most normal and outlying
                object in the training (and pred) set.
        """

        if os.path.exists(f"{self.o_scores_path}/{self.metric}_o_score.npy"):
            scores = np.load(f"{self.o_scores_path}/{self.metric}_o_score.npy")
        else:
            scores = self.score(O)

        spec_idxs = np.argpartition(
            scores, [n_top_spectra, -1 * n_top_spectra]
        )

        most_normal_ids = spec_idxs[:n_top_spectra]
        most_oulying_ids = spec_idxs[-1 * n_top_spectra :]

        return most_normal_ids, most_oulying_ids


###############################################################################
class AEpca:
    def __init__(self, in_dim, lat_dim=2, batch_size=32, epochs=10, lr=1e-4):
        self.in_dim = in_dim
        self.batch_size = batch_size
        self.lat_dim = lat_dim
        self.epochs = epochs
        self.lr = lr
        self.encoder = None
        self.decoder = None
        self.AE = None
        self._init_AE()

    def _init_AE(self):

        # Build Encoder
        inputs = Input(shape=(self.in_dim,), name="encoder_input")
        latent = Dense(self.lat_dim, name="latent_vector")(inputs)
        self.encoder = Model(inputs, latent, name="encoder")
        self.encoder.summary()
        #        plot_model(self.encoder, to_file='encoder.png', show_shapes='True')

        # Build Decoder
        latent_in = Input(shape=(self.lat_dim,), name="decoder_input")
        outputs = Dense(self.in_dim, name="decoder_output")(latent_in)
        self.decoder = Model(latent_in, outputs, name="decoder")
        self.decoder.summary()
        #        plot_model(self.decoder, to_file='decoder.png', show_shapes='True')

        # AE = Encoder + Decoder
        autoencoder = Model(
            inputs, self.decoder(self.encoder(inputs)), name="autoencoder"
        )
        autoencoder.summary()
        #        plot_model(autoencoder, to_file='autoencoder.png', show_shapes=True)

        # Mean square error loss function with Adam optimizer
        autoencoder.compile(loss="mse", optimizer="adam")  # , lr = self.lr)

        self.AE = autoencoder

    def fit(self, spectra):
        self.AE.fit(
            spectra, spectra, epochs=self.epochs, batch_size=self.batch_size
        )

    def predict(self, test_spec):
        if test_spec.ndim == 1:
            test_spec = test_spec.reshape(1, -1)
        return self.AE.predict(test_spec)

    def encode(self, spec):
        if spec.ndim == 1:
            spec = spec.reshape(1, -1)
        return self.encoder(spec)

    def decode(self, lat_val):
        return self.decoder(lat_val)

    def save(self):
        self.encoder.save("encoder")
        self.decoder.save("decoder")
        self.AE.save("AutoEncoder")


###############################################################################
class PcA:
    def __init__(self, n_comps=False):
        if not (n_comps):
            self.PCA = PCA()
        else:
            self.n_comps = n_comps
            self.PCA = PCA(self.n_comps)

    def fit(self, spec):
        return self.PCA.fit(spec)

    def components(self):
        return self.PCA.components_

    def expvar(self):
        return self.PCA.explained_variance_ratio_

    def inverse(self, trf_spec):
        if trf_spec.ndim == 1:
            trf_spec = trf_spec.reshape(1, -1)

        return self.PCA.inverse_transform(trf_spec)

    def predict(self, test_spec):
        if test_spec.ndim == 1:
            test_spec = test_spec.reshape(1, -1)

        return self.PCA.transform(test_spec)


###############################################################################
def plt_spec_pca(flx, pca_flx, componets):
    """Comparative plot to see how efficient is the PCA compression"""
    plt.figure(figsize=(8, 4))

    # Original Image
    plt.subplot(1, 2, 1)
    plt.plot(flx)
    plt.xlabel(f"{flx.size} components", fontsize=14)
    plt.title("Original Spectra", fontsize=20)

    # principal components
    plt.subplot(1, 2, 2)
    plt.plot(pca_flx)
    plt.xlabel(f"{componets} componets", fontsize=14)
    plt.title("Reconstructed spectra", fontsize=20)
    plt.show()
    plt.close()


###############################################################################
def plot_2D(data, title):
    fig = plt.figure()
    plt.title(title)
    plt.plot(data[:, 0], data[:, 1], "b.")
    plt.xlabel("$z_1$", fontsize=18)
    plt.ylabel("$z_2$", fontsize=18, rotation=0)
    plt.savefig(f"{title}.png")
    plt.show()
    plt.close()


###############################################################################
