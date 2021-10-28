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
from tensorflow.keras import layers
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
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
###############################################################################
class Encoder(layers.Layer):
    """Maps spectra to a triplet (z_mean, z_log_var, z)."""

    def __init__(self,
        architecture: "dict",
        name: "str"="encoder",
        deterministic: "bool"=False,
        **kwargs
        ):

        super(Encoder, self).__init__(name=name, **kwargs)

        [
            self.input_dimensions,
            self.encoder_units,
            self.latent_dimensions,
        ] = self._get_architecture(architecture)

        self.deterministic = deterministic

        self.mu = None
        self.log_variance = None
        self.z = None

    def call(self, inputs):

        if not self.deterministic:

            x = self._encoder_block(inputs)

            mu, log_variance = self._get_mu_log_variance(x)

            self.sampling = Sampling()

            z = self.sampling((mu, log_variance))
            self.z = z

            return mu, log_variance, z

        z = self._encoder_block(inputs)
        self.z = z

        return z
    ############################################################################
    def _encoder_block(self, inputs):

        return self._add_block(input=inputs, block="encoder")

    ###########################################################################
    def _get_mu_log_variance(self, x: ""):

        self.mu = Dense(self.latent_dimensions, name="mu")(x)

        self.log_variance = Dense(self.latent_dimensions,
            name="log_variance")(x)

        return self.mu, self.log_variance
    ###########################################################################
    def _add_block(self, input: "keras.Input", block: "str"):

        x = input

        input_dimensions = self.input_dimensions
        block_units = self.encoder_units

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

    ###########################################################################
    def _get_architecture(self, architecture: "dict"):

        input_dimensions = int(architecture["input_dimensions"])

        encoder_units = architecture["encoder"]
        encoder_units = [int(units) for units in encoder_units.split("_")]

        latent_dimensions = int(architecture["latent_dimensions"])

        return [
            input_dimensions,
            encoder_units,
            latent_dimensions
        ]
    ###########################################################################
class Decoder(layers.Layer):
    """Converts z, the encoded digit vector, back into a readable digit."""

    def __init__(self,
        architecture: "dict",
        name: "str"="decoder",
        deterministic: "bool"=False,
        **kwargs
        ):

        super(Decoder, self).__init__(name=name, **kwargs)

        [
            self.input_dimensions,
            self.decoder_units
        ] = self._get_architecture(architecture)

        self.deterministic = deterministic
    ###########################################################################
    def call(self, inputs):
        x = self.dense_proj(inputs)
        return self.dense_output(x)

    ###########################################################################
    def _get_architecture(self, architecture: "dict"):

        input_dimensions = int(architecture["input_dimensions"])

        decoder_units = architecture["decoder"]
        decoder_units = [int(units) for units in decoder_units.split("_")]

        return [
            input_dimensions,
            decoder_units
        ]

    ###########################################################################
    ###########################################################################
    def _get_units(self, architecture):

        decoder_units = architecture["decoder"]
        decoder_units = [int(units) for units in decoder_units.split("_")]

        return decoder_units
    ###########################################################################

    ###########################################################################
    ###########################################################################
###############################################################################
class VAE:
    """Create a variational autoencoder"""
    def __init__(
        self,
        architecture: "dict"={None},
        hyperparameters: "dict"={None},
        reload: "bool"=False,
        parameters: "list"=[None],
    ) -> "VAE object":

        """
        PARAMETERS

            architecture: {
                "encoder" : 500_50, # encoder units per layer
                "latent_dimensions" : 10,
                "decoder" : 50_500, # decoder units per layer
                "input_dimensions" : number of spectra #(set in train)
            }

            hyperparameters: {
                learning_rate : 1e-4,
                batch_size : 1024,
                epochs : 5,
                out_activation : linear,
                reconstruction_weight : 1000
                klk_weight: 1
            }

            reload: if True, sets model to be loaded with class method
                load
                # not implemented yet

            parameters: parameters to get the model if load is True
                # using it now as a shorcut to class method
                # reload implementation
        """

        #######################################################################
        if reload:

            [
                self.input_dimensions,
                self.encoder_units,
                self.latent_dimensions,
                self.decoder_units,
                self.batch_size,
                self.epochs,
                self.learning_rate,
                self.reconstruction_weight,
                self.kl_weight,
                self.out_activation
            ] = parameters

            encoder = ' '.join(map(str, self.encoder_units)).replace(' ', '_')
            decoder = ' '.join(map(str, self.decoder_units)).replace(' ', '_')
            self.architecture_str = (
                f"{encoder}_{self.latent_dimensions}_{decoder}"
            )
        #######################################################################

        else:
            [
                self.input_dimensions,
                self.encoder_units,
                self.latent_dimensions,
                self.decoder_units,
                self.architecture_str
            ] = self._get_architecture(architecture)

            [
                self.batch_size,
                self.epochs,
                self.learning_rate,
                self.out_activation,
                self.reconstruction_weight,
                self.kl_weight,
            ] = self._get_hyperparameters(hyperparameters)

        self.encoder = None
        self.decoder = None
        self.model = None

        self._build()

    ###########################################################################
    def reconstruct(self, spectra: "2D np.array") -> "2D np.array":
        """
        Once the VAE is trained, this method is used to obtain
        the spectra learned by the model

        PARAMETERS
            spectra: contains fluxes of observed spectra

        OUTPUTS
            predicted_spectra: contains generated spectra by the model
                from observed spectra (input)
        """

        if spectra.ndim == 1:
            spectra = spectra.reshape(1, -1)

        predicted_spectra = self.model.predict(spectra)

        return predicted_spectra

    ###########################################################################
    def encode(self, spectra: "2D np.array") -> "2D np.array":
        """
        Given an array of observed fluxes, this method outputs the
        latent representation learned by the VAE onece it is trained

        PARAMETERS
            spectra: contains fluxes of observed spectra

        OUTPUTS
            z: contains latent representation of the observed fluxes

        """

        if spectra.ndim == 1:
            spectra = spectra.reshape(1, -1)

        z = self.encoder.predict(spectra)

        return z

    ###########################################################################
    def decode(self, z: "2D np.array") -> "2D np.aray":
        """

        Given a set of points in latent space, this method outputs
        spectra according to the representation learned by the VAE
        onece it is trained

        PARAMETERS
            z: contains a set of latent representation

        OUTPUTS
            spectra: contains fluxes of spectra built by the model

        """

        if z.ndim == 1:
            coding = z.reshape(1, -1)

        spectra = self.decoder.predict(z)

        return spectra

    ###########################################################################
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

    ###########################################################################
    def _get_hyperparameters(self, hyperparameters: "dict"):

        batch_size = int(hyperparameters["batch_size"])
        epochs = int(hyperparameters["epochs"])
        learning_rate = float(hyperparameters["learning_rate"])
        out_activation = hyperparameters["out_activation"]
        reconstruction_weight = float(hyperparameters["reconstruction_weight"])
        kl_weight = float(hyperparameters["kl_weight"])

        return [
            batch_size,
            epochs,
            learning_rate,
            out_activation,
            reconstruction_weight,
            kl_weight
        ]
    ###########################################################################
    def _get_architecture(self, architecture: "dict"):

        input_dimensions = int(architecture["input_dimensions"])

        encoder_units = architecture["encoder"]
        tail = encoder_units
        encoder_units = [int(units) for units in encoder_units.split("_")]

        latent_dimensions = int(architecture["latent_dimensions"])
        decoder_units = architecture["decoder"]

        tail = f"{tail}_{latent_dimensions}_{decoder_units}"

        decoder_units = [int(units) for units in decoder_units.split("_")]


        return [
            input_dimensions,
            encoder_units,
            latent_dimensions,
            decoder_units,
            tail
        ]
    ###########################################################################
    def _build(self) -> "":
        """
        Builds and returns a compiled variational auto encoder using
        keras API
        """

        self._build_encoder()
        self._build_decoder()
        self._build_vae()
        self._compile()

    ###########################################################################
    def _compile(self):

        optimizer = Adam(learning_rate=self.learning_rate)

        self.model.compile(
            optimizer=optimizer,
            loss=self._loss,
            metrics=["mse"]
            # metrics=["mse" , self._kl_loss]
        )

    ###########################################################################
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

        loss = self.reconstruction_weight * reconstruction_loss\
            + self.kl_weight * kl_loss

        return loss

    ###########################################################################
    def _reconstruction_loss(self, y_target, y_predicted):

        error = y_target - y_predicted
        reconstruction_loss = K.mean(K.square(error), axis=1)

        return reconstruction_loss

    ###########################################################################
    def _kl_loss(self, y_target, y_predicted):

        kl_loss = -0.5 * K.sum(
            1
            + self.log_variance
            - K.square(self.mu)
            - K.exp(self.log_variance),
            axis=1,
        )

        return kl_loss

    ###########################################################################
    def _build_vae(self):

        input = self._model_input
        output = self.decoder(self.encoder(input))
        self.model = Model(input, output, name="vae")

    ###########################################################################
    def _build_decoder(self):

        decoder_input = Input(
            shape=(self.latent_dimensions,), name="decoder_input"
        )

        decoder_block = self._add_block(input=decoder_input, block="decoder")

        decoder_output = self._output_layer(decoder_block)

        self.decoder = Model(decoder_input, decoder_output, name="decoder")
    ###########################################################################
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

    ###########################################################################
    def _build_encoder(self):

        encoder_input = Input(
            shape=(self.input_dimensions,), name="encoder_input"
        )

        encoder_block = self._add_block(input=encoder_input, block="encoder")

        latent_layer = self._latent_layer(encoder_block)

        self._model_input = encoder_input
        self.encoder = Model(encoder_input, latent_layer, name="encoder")

    ###########################################################################
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

    ###########################################################################
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

    ###########################################################################
    def _latent_layer(self, x: ""):

        self.mu = Dense(self.latent_dimensions, name="mu")(x)

        self.log_variance = Dense(self.latent_dimensions,
            name="log_variance")(x)
        #######################################################################
        def sample_normal_distribution(args):

            mu, log_variance = args
            epsilon = K.random_normal(
                shape=K.shape(mu), mean=0.0, stddev=1.0
            )

            point = mu + K.exp(log_variance / 2) * epsilon

            return point

        #######################################################################
        x = Lambda(sample_normal_distribution, name="encoder_outputs")(
            [self.mu, self.log_variance]
        )

        return x


    ###########################################################################
    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    ###########################################################################
    def save_model(self, directory: "str"):

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

    def _load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    ###########################################################################
    @classmethod
    def load(cls, save_directory):
        parameters_location = f"{save_directory}/parameters.pkl"

        with open(parameters_location, "rb") as file:
            parameters = pickle.load(file)

        print(parameters)
        autoencoder = VAE(reload=True, parameters = parameters)
        weights_location = f"{save_directory}/weights.h5"
        autoencoder._load_weights(weights_location)
        return autoencoder
###############################################################################
