import os
import pickle
###############################################################################
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

###############################################################################
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Activation, BatchNormalization, LeakyReLU
from tensorflow.keras.layers import Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

###############################################################################
tf.compat.v1.disable_eager_execution()
# this is necessary because I use custom loss function
###############################################################################
class VAE:
    def __init__(
        self,
        input_dimensions: "int",
        encoder_units: "list",
        latent_dimensions: "int",
        decoder_units: "list",
        batch_size: "int",
        epochs: "int",
        learning_rate: "float",
        reconstruction_loss_weight: "float",
        output_activation: "str" = "linear",
    ) -> "tf.keras.model":

        """
        PARAMETERS

        input_dimensions:

        encoder_units: python list containing the number of units
            in each layer of the encoder

        latent_dimensions: number of  dimensions for the latent
            representation

        decoder_units: python list containing the number of units
            in each layer of the decoder

        batch_size: number of batches for the training set

        epochs: maximum number of epochs to train the algorithm

        learning_rate: value for the learning rate

        output_activation:

        reconstruction_loss_weight: weighting factor for the
            reconstruction loss
        """

        self.input_dimensions = input_dimensions

        self.encoder_units = encoder_units
        self.latent_dimensions = latent_dimensions
        self.decoder_units = decoder_units

        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.out_activation = output_activation

        self.reconstruction_weight = reconstruction_loss_weight

        self.encoder = None
        self.decoder = None
        self.model = None

        self._build()

    ############################################################################
    def reconstruct(self, spectra: "2D np.array") -> "2D np.array":

        if spectra.ndim == 1:
            spectra = spectra.reshape(1, -1)

        return self.model.predict(spectra)

    ############################################################################
    def encode(self, spectra: "2D np.array") -> "2D np.array":

        if spectra.ndim == 1:
            spectra = spectra.reshape(1, -1)

        z = self.encoder.predict(spectra)

        return z

    ############################################################################
    def decode(self, z: "2D np.array") -> "2D np.aray":

        if z.ndim == 1:
            coding = z.reshape(1, -1)

        spectra = self.decoder.predict(z)

        return spectra

    ############################################################################
    def train(self, spectra):
        print(spectra.shape)
        self.model.fit(
            x=spectra,
            y=spectra,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=2,
            shuffle=True
        )

    ############################################################################
    def _build(self) -> "":
        """
        Builds and returns a compiled variational auto encoder using
        keras API
        """

        self._build_encoder()
        self._build_decoder()
        self._build_vae()
        self._compile()

    ############################################################################
    def _compile(self):

        optimizer = Adam(learning_rate=self.learning_rate)

        self.model.compile(
            optimizer=optimizer,
            loss=self._loss,
            metrics=["mse"]
            # metrics=["mse" , self._kl_loss]
        )

    ############################################################################
    def _loss(self, y_target, y_predicted):
        """
        Standard loss function for the variational auto encoder
        , that is, for de reconstruction loss and the KL divergence.

        # y_target, y_predicted: keep consistency with keras API
        # a loss function expects this two parameters

        INPUTS
            y_true : input datapoint
            y_pred : predicted value by the network

        OUTPUT
            loss function with the keras format that is compatible
            with the .compile API
        """

        reconstruction_loss = self._reconstruction_loss(y_target, y_predicted)
        kl_loss = self._kl_loss(y_target, y_predicted)

        loss = self.reconstruction_weight * reconstruction_loss + kl_loss

        return loss

    ############################################################################
    def _reconstruction_loss(self, y_target, y_predicted):

        error = y_target - y_predicted
        reconstruction_loss = K.mean(K.square(error), axis=1)

        return reconstruction_loss

    ############################################################################
    def _kl_loss(self, y_target, y_predicted):

        kl_loss = -0.5 * K.sum(
            1
            + self.log_variance
            - K.square(self.mu)
            - K.exp(self.log_variance),
            axis=1,
        )

        return kl_loss

    ############################################################################
    def _build_vae(self):

        input = self._model_input
        output = self.decoder(self.encoder(input))
        self.model = Model(input, output, name="vae")

        # model = Model(input, output, name="vae")
        # # Retrieve the config
        # config = model.get_config()
        # # At loading time, register the custom objects with a `custom_object_scope`:
        # custom_objects = {
        #     # "latent_layer": self._latent_layer,
        #     # "kl_loss": self._kl_loss,
        #     # "reconstruction_loss": self._reconstruction_loss,
        #     "custom_loss": self._loss
        # }
        # with tf.keras.utils.custom_object_scope(custom_objects):
        #     self.model = Model.from_config(config)

    ############################################################################
    def _build_decoder(self):

        decoder_input = Input(
            shape=(self.latent_dimensions,), name="decoder_input"
        )

        decoder_block = self._add_block(input=decoder_input, block="decoder")

        decoder_output = self._output_layer(decoder_block)

        self.decoder = Model(decoder_input, decoder_output, name="decoder")
    ############################################################################
    def _output_layer(self, decoder_block: "keras.Dense"):

        units = self.encoder_units[-1]
        standard_deviation = np.sqrt(2.0 / units)

        initial_weights = tf.keras.initializers.RandomNormal(
            mean=0.0, stddev=standard_deviation
        )

        output_layer = Dense(
            self.input_dimensions,
            kernel_initializer=initial_weights,
            name="decoder_output_layer",
        )

        x = output_layer(decoder_block)
        x = Activation(self.out_activation, name="output_activation")(x)

        return x

    ############################################################################
    def _build_encoder(self):

        encoder_input = Input(
            shape=(self.input_dimensions,), name="encoder_input"
        )

        encoder_block = self._add_block(input=encoder_input, block="encoder")

        latent_layer = self._latent_layer(encoder_block)

        self._model_input = encoder_input
        self.encoder = Model(encoder_input, latent_layer, name="encoder")

    ############################################################################
    def _add_block(self, input: "keras.Input", block: "str"):

        x = input

        if block == "encoder":
            input_dimensions = self.input_dimensions
            block_units = self.encoder_units
        else:
            input_dimensions = self.latent_dimensions
            block_units = self.decoder_units

        standard_deviation = np.sqrt(2.0 / input_dimensions)

        for layer_index, number_units in enumerate(block_units):

            x, standard_deviation = self._add_layer(
                x, layer_index, number_units, standard_deviation, block
            )

        return x

    ############################################################################
    def _add_layer(
        self,
        x: "keras.Dense",
        layer_index: "int",
        number_units: "int",
        standard_deviation: "float",
        block: "str",
    ):

        initial_weights = tf.keras.initializers.RandomNormal(
            mean=0.0, stddev=standard_deviation
        )

        layer = Dense(
            number_units,
            kernel_initializer=initial_weights,
            name=f"{block}_layer_{layer_index + 1}",
        )

        x = layer(x)

        x = LeakyReLU(name=f"LeakyReLU_{block}_{layer_index + 1}")(x)

        x = BatchNormalization(
            name=f"batch_normaliztion_{block}_{layer_index + 1}"
        )(x)

        standard_deviation = np.sqrt(2.0 / number_units)

        return x, standard_deviation

    ############################################################################
    def _latent_layer(self, x: ""):

        self.mu = Dense(self.latent_dimensions, name="mu")(x)

        self.log_variance = Dense(self.latent_dimensions,
            name="log_variance")(x)
        ########################################################################
        def sample_normal_distribution(args):

            mu, log_variance = args
            epsilon = K.random_normal(
                shape=K.shape(mu), mean=0.0, stddev=1.0
            )

            point = mu + K.exp(log_variance / 2) * epsilon

            return point

        ########################################################################
        x = Lambda(sample_normal_distribution, name="encoder_outputs")(
            [self.mu, self.log_variance]
        )

        return x


    ############################################################################
    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    ############################################################################
    def save_model(self, directory: "str"):

        # tf.keras.models.save_model(self.model, f"{directory}/vae")
        # tf.keras.models.save_model(self.encoder, f"{directory}/encoder")
        # tf.keras.models.save_model(self.decoder, f"{directory}/decoder")
        self._save_parameters(directory)
        self._save_weights(directory)

    def _save_parameters(self, save_directory):

        parameters = [
            self.input_dimensions,
            self.encoder_units,
            self.latent_dimensions,
            self.decoder_units,
            self.batch_size,
            self.epochs,
            self.learning_rate,
            self.reconstruction_weight,
            self.out_activation,
        ]

        save_location = f"{save_directory}/parameters.pkl"
        with open(save_location, "wb") as file:
            pickle.dump(parameters, file)

    def _save_weights(self, save_directory):
        save_location = f"{save_directory}/weights.h5"
        self.model.save_weights(save_location)

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    @classmethod
    def load(cls, save_directory):
        parameters_location = f"{save_directory}/parameters.pkl"

        with open(parameters_location, "rb") as file:
            parameters = pickle.load(file)

        print(parameters)
        autoencoder = VAE(*parameters)
        weights_location = f"{save_directory}/weights.h5"
        autoencoder.load_weights(weights_location)
        return autoencoder
#     def plot_model(self):
#
#         plot_model(self.vae, to_file='DenseVAE.png', show_shapes='True')
#         plot_model(self.encoder, to_file='DenseEncoder.png', show_shapes='True')
#         plot_model(self.decoder, to_file='DenseDecoder.png', show_shapes='True')

class LoadAE:
    """Load AE model"""

    ############################################################################
    def __init__(self, model:'str', location:'str'):

        # self.model = load_model(f'{location}/{model}')
        # self.encoder = load_model(f'{location}/encoder')
        # self.decoder = load_model(f'{location}/decoder')
        self.model = load_model(f'{location}/{model}.h5')
        self.encoder = load_model(f'{location}/encoder.h5')
        self.decoder = load_model(f'{location}/decoder.h5')

    ###########################################################################
    def reconstruct(self, spectra: "2D np.array") -> "2D np.array":

        spectra = self._update_dimensions(spectra)

        return self.model.predict(spectra)

    ############################################################################
    def encode(self, spectra: "2D np.array") -> "2D np.array":

        spectra = self._update_dimensions(spectra)
        z = self.encoder.predict(spectra)
        return z

    ############################################################################
    def decode(self, z: "2D np.array") -> "2D np.aray":

        spectra = self._update_dimensions(spectra)
        spectra = self.decoder.predict(z)
        return spectra

    ############################################################################
    def _update_dimensions(self, x: "np.array"):

        if x.ndim == 1:
            x = x[np.newaxis, ...]

        return x
    ############################################################################
    # def plot_model(self):
    #
    #     plot_model(self.ae, to_file="DenseVAE.png", show_shapes="True")
    #     plot_model(
    #         self.encoder, to_file="DenseEncoder.png", show_shapes="True"
    #     )
    #     plot_model(
    #         self.decoder, to_file="DenseDecoder.png", show_shapes="True"
    #     )

    ############################################################################
    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()
