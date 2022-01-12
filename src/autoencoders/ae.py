import os
import pickle
###############################################################################
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Layer

###############################################################################
from autoencoders.customObjects import MyCustomLoss

###############################################################################
# taken from tf tutorias website
class SamplingLayer(Layer):
    """
    Uses (z_mean, z_log_variance) to sample z, the latent vector.
    And compute kl_divergence
    """

    ###########################################################################
    def __init__(self, name: str = "sampling_layer"):

        super(SamplingLayer, self).__init__(name=name)

    ###########################################################################
    def call(self, inputs):

        z_mean, z_log_var = inputs

        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))

        z = keras.backend.exp(0.5 * z_log_var) * epsilon
        z += z_mean

        return z

    ###########################################################################
    def get_config(self):

        return {"name": self.name}

    ###########################################################################
    # necessary to serialize the custom loss
    @classmethod
    def from_config(cls, config):
        return cls(**config)


###############################################################################
class AutoEncoder:
    """
    Create an AE model using the keras functional API, where custom
    layers are created by subclassing keras.layers.Layer, the same
    applies for custom metrics and losses.

    For all custom objects, the .get_config method is implemented to
    be able to serialize and clone the model.
    """

    ###########################################################################
    def __init__(
        self,
        architecture: dict = {None},
        hyperparameters: dict = {None},
        reload: bool = False,
        reload_from: str = ".",
    ):
        """
        PARAMETERS
            architecture:
            hyperparameters:
            reload:
            reload_from:
        """

        if reload is True:

            self.model = keras.models.load_model(
                f"{reload_from}",
                custom_objects={
                    "MyCustomLoss": MyCustomLoss,
                    "SamplingLayer": SamplingLayer,
                },
            )

            self.KLD = None # KL Divergence
            self.MMD = None # Maximum Mean Discrepancy

            self._set_class_instances_from_saved_model(reload_from)

        else:

            self.architecture = architecture
            self.hyperparameters = hyperparameters

            self.encoder = None
            self.KLD = None # KL Divergence
            self.MMD = None # Maximum Mean Discrepancy
            self.decoder = None
            self.model = None
            # To link encoder with decoder. Define here for documentation
            self.original_input = None
            self.original_output = None

            self.train_history = None

            self._build_model()

    ###########################################################################
    def _set_class_instances_from_saved_model(self, reload_from: str) -> None:

        #######################################################################
        self.architecture["model_name"] = self.model.name
        # Get encoder and decoder
        for submodule in self.model.submodules:

            if submodule.name == "encoder":

                self.encoder = submodule

            elif submodule.name == "decoder":

                self.decoder = submodule
        #######################################################################
        file_location = f"{reload_from}/parameters_and_train_history.pkl"
        with open(file_location, "rb") as file:
            parameters = pickle.load(file)

        [
        self.architecture,
        self.hyperparameters,
        self.train_history
        ] = parameters
        # the rest of the parameters would be loaded from a pickle file :P
        # self.architecture["is_variational"] = is_variational
        #
        # self.architecture = architecture
        # self.hyperparameters = hyperparameters
        #
        # self.original_input = None
        # # Not sure how to load this :O
        # self.KLD = None # KL Divergence
        # self.MMD = None # Maximum Mean Discrepancy
        # self.original_output = None
        # self.model = None
        #
        # self.train_history = None

    ###########################################################################
    def train(self,
        spectra: np.array,
    ) -> keras.callbacks.History:

        stopping_criteria = keras.callbacks.EarlyStopping(
            monitor="mse",
            patience=self.hyperparameters["patience"],
            verbose=1,
            restore_best_weights = True,
        )

        print(type(self.hyperparameters["use_multiprocessing"]))
        print(type(self.hyperparameters["workers"]))
        history = self.model.fit(
            x=spectra,
            y=spectra,
            batch_size=self.hyperparameters["batch_size"],
            epochs=self.hyperparameters["epochs"],
            verbose=1,  # progress bar
            use_multiprocessing=self.hyperparameters["use_multiprocessing"],
            workers = self.hyperparameters["workers"],
            shuffle=True,
            callbacks = stopping_criteria,

        )

        self.train_history = history

        return history

    ###########################################################################
    def reconstruct(self, spectra: np.array) -> np.array:
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
    def encode(self, spectra: np.array) -> np.array:
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
    def decode(self, z: np.array) -> np.array:
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
    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    ###########################################################################
    def save_model(
        self,
        save_to: str,
    ) -> None:

        save_to = f"{save_to}/{self.architecture['model_name']}"

        # There is no need to save the encoder and or decoder
        # keras.models.Model.sumodules instance has them
        self.model.save(f"{save_to}")
        #######################################################################
        parameters = [
            self.architecture,
            self.hyperparameters,
            self.train_history
        ]

        with open(f"{save_to}/parameters_and_train_history.pkl", "wb") as file:
            pickle.dump(parameters, file)

    ###########################################################################
    def _build_model(self) -> None:
        """
        Builds the the auto encoder model
        """
        self._build_encoder()
        self._build_decoder()
        self._build_ae()
        self._compile()

    ###########################################################################
    def _compile(self):

        optimizer = keras.optimizers.Adam(
            learning_rate=self.hyperparameters["learning_rate"]
        )

        reconstruction_weight = self.hyperparameters["reconstruction_weight"]

        MSE = MyCustomLoss(
            name="weighted_MSE",
            keras_loss=keras.losses.MSE,
            weight_factor=reconstruction_weight,
        )

        self.model.compile(optimizer=optimizer, loss=MSE, metrics=["mse"])

    ###########################################################################
    def _build_ae(self):

        self.original_output = self.decoder(self.encoder(self.original_input))

        self.model = keras.Model(
            self.original_input, self.original_output, name=self.architecture["model_name"]
        )

        # Add KLD and MMD here to have a nice print of summary
        # of encoder and decoder submodules :)
        if self.architecture["is_variational"] is True:
            alpha = self.hyperparameters["alpha"]
            lambda_ = self.hyperparameters["lambda"]
            KLD = self.KLD * (1 - alpha)
            MMD = self.MMD * (alpha + lambda_ -1)
            self.model.add_loss(KLD)
            self.model.add_loss(MMD)

    ###########################################################################
    def _build_decoder(self):
        """Build decoder"""

        decoder_input = keras.Input(
            shape=(self.architecture["latent_dimensions"],),
            name="decoder_input",
        )

        block_output = self._add_block(decoder_input, block="decoder")

        decoder_output = self._output_layer(block_output)

        self.decoder = keras.Model(
            decoder_input, decoder_output, name="decoder"
        )

    ###########################################################################
    def _output_layer(self, input_tensor: tf.Tensor) -> tf.Tensor:

        output_layer = Dense(
            units=self.architecture["input_dimensions"],
            activation=self.hyperparameters["output_activation"],
            name="decoder_output",
        )

        output_tensor = output_layer(input_tensor)

        return output_tensor

    ###########################################################################
    def _build_encoder(self):
        """Build encoder"""

        encoder_input = keras.Input(
            shape=(self.architecture["input_dimensions"],),
            name="encoder_input",
        )

        self.original_input = encoder_input

        block_output = self._add_block(encoder_input, block="encoder")

        if self.architecture["is_variational"] is True:

            z, z_mean, z_log_var = self._sampling_layer(block_output)

            # To add later on to the model before compiling

            # Compute KLD
            self.KLD = -0.5 * tf.reduce_mean(
                z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
            )

            # Compute MMD
            batch = tf.shape(z_mean)[0]

            z_prime = keras.backend.random_normal(
                shape=(batch, self.architecture["latent_dimensions"])
            )

            def compute_kernel(x, y):

                batch = tf.shape(x)[0]

                tiled_x = tf.tile(
                    tf.reshape(
                        x,
                        tf.stack([batch, 1, self.architecture["latent_dimensions"]])
                    ),
                    tf.stack([1, batch, 1])
                )


                tiled_y = tf.tile(
                    tf.reshape(
                        y,
                        tf.stack([1, batch, self.architecture["latent_dimensions"]])
                    ),
                    tf.stack([batch, 1, 1])
                )

                kernel = tf.exp(
                    -tf.reduce_mean(
                        tf.square(tiled_x - tiled_y),
                        axis=2
                    ) / tf.cast(self.architecture["latent_dimensions"], tf.float32)
                )

                return kernel


            z_prime_kernel = compute_kernel(z_prime, z_prime)
            z_prime_z_kernel = compute_kernel(z_prime, z)
            z_kernel = compute_kernel(z, z)

            self.MMD = tf.reduce_mean(z_prime_kernel) + tf.reduce_mean(z_kernel) - 2 * tf.reduce_mean(z_prime_z_kernel)

        else:

            z_layer = Dense(
                units=self.architecture["latent_dimensions"],
                activation="relu",
                name=f"z_deterministic",
            )

            z = z_layer(block_output)

        self.encoder = keras.Model(encoder_input, z, name="encoder")

    ###########################################################################
    def _add_block(self, input_tensor: tf.Tensor, block: str) -> tf.Tensor:
        """
        Build an graph of dense layers

        PARAMETERS
            input_tensor:
            block:

        OUTPUT
            x:
        """
        x = input_tensor

        if block == "encoder":

            block_units = self.architecture["encoder"]

        else:

            block_units = self.architecture["decoder"]

        for layer_index, number_units in enumerate(block_units):

            # in the first iteration, x is the input tensor in the block
            x = self._get_next_dense_layer_output(
                x, layer_index, number_units, block
            )

        return x

    ###########################################################################
    def _get_next_dense_layer_output(
        self,
        input_tensor: tf.Tensor,  # the output of the previous layer
        layer_index: int,
        number_units: int,
        block: str,
    ) -> tf.Tensor:

        """
        Define and get output of next Dense layer

        PARAMETERS
            input_tensor:
            layer_index:
            number_units:
            block:

        OUTPUT
            output_tensor:
        """

        layer = Dense(
            units=number_units,
            activation="relu",
            name=f"{block}_{layer_index + 1:02d}",
        )

        output_tensor = layer(input_tensor)

        return output_tensor

    ###########################################################################
    def _sampling_layer(
        self, encoder_output: tf.Tensor
    ) -> [tf.Tensor, tf.Tensor, tf.Tensor]:

        """
        Sample output of the encoder and add the kl loss

        PARAMETERS
            encoder_output:

        OUTPUT
            z, z_mean, z_log_var
        """

        mu_layer = Dense(
            units=self.architecture["latent_dimensions"], name="z_mean"
        )

        z_mean = mu_layer(encoder_output)

        log_var_layer = Dense(
            units=self.architecture["latent_dimensions"], name="z_log_variance"
        )

        z_log_var = log_var_layer(encoder_output)

        sampling_inputs = (z_mean, z_log_var)
        sample_layer = SamplingLayer(name="z_variational")

        z = sample_layer(sampling_inputs)

        return z, z_mean, z_log_var

    ###########################################################################
