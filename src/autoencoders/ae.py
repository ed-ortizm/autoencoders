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
        model_name: str = "VAE",
        is_variational: bool = True,
        reload: bool = False,
        reload_from: str = ".",
    ):
        """
        PARAMETERS
            input_dimensions:
            architecture:
            latent_dimensions:
            decoder_architecture:
            hyperparameters:
            reload:
        """

        if reload is True:

            self.model = keras.models.load_model(
                f"{reload_from}",
                custom_objects={
                    "MyCustomLoss": MyCustomLoss,
                    "SamplingLayer": SamplingLayer,
                },
            )

            self._set_class_instances_from_saved_model()

        else:

            self.model_name = model_name
            self.is_variational = is_variational

            self.architecture = architecture
            self.hyperparameters = hyperparameters

            self.original_input = None
            self.encoder = None
            self.KLD = None # KL Divergence
            self.MMD = None # Maximum Mean Discrepancy
            self.decoder = None
            self.original_output = None
            self.model = None

            self.train_history = None

            self._build_model()

    ###########################################################################
    def _set_class_instances_from_saved_model(self):

        #######################################################################
        self.model_name = self.model.name
        # Get encoder and decoder
        for submodule in self.model.submodules:

            if submodule.name == "encoder":

                self.encoder = submodule

            elif submodule.name == "decoder":

                self.decoder = submodule
        #######################################################################
        # the rest of the parameters would be loaded from a pickle file :P

    ###########################################################################
    def train(self,
        spectra: np.array,
    ) -> keras.callbacks.History:

        stopping_criteria = keras.callbacks.EarlyStopping(
            patience=self.hyperparameters["patience"],
            verbose=1
        )
        history = self.model.fit(
            x=spectra,
            y=spectra,
            batch_size=self.hyperparameters["batch_size"],
            epochs=self.hyperparameters["epochs"],
            verbose=1,  # progress bar
            # use_multiprocessing=True,
            shuffle=True,

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
        # model_name: str,
        save_to: str,
        # save_encoder: bool = True,
        # save_decoder: bool = True
    ) -> None:

        self.model.save(f"{save_to}")

        # Looks like there is no need for this, since Model.sumodules
        # instance has all the info I need :P

        # if save_encoder is True:
        #     self.encoder.save(f"{save_to}/{model_name}/encoder")
        #
        # if save_decoder is True:
        #     self.decoder.save(f"{save_to}/{model_name}/decoder")

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
            self.original_input, self.original_output, name=self.model_name
        )

        if self.is_variational is True:
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

        if self.is_variational is True:

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


###############################################################################
###############################################################################
# tf.compat.v1.disable_eager_execution()
# this is necessary because I use custom loss function

###############################################################################
class VAE_old:
    """Create a variational autoencoder"""

    def __init__(
        self,
        architecture: "dict" = {None},
        hyperparameters: "dict" = {None},
        reload: "bool" = False,
        parameters: "list" = [None],
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
                self.out_activation,
                self.train_history,
            ] = parameters

            encoder = " ".join(map(str, self.encoder_units)).replace(" ", "_")
            decoder = " ".join(map(str, self.decoder_units)).replace(" ", "_")
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
                self.architecture_str,
            ] = self._get_architecture(architecture)

            [
                self.batch_size,
                self.epochs,
                self.learning_rate,
                self.out_activation,
                self.reconstruction_weight,
                self.kl_weight,
            ] = self._get_hyperparameters(hyperparameters)

            self.train_history = {}  # has parameters and history keys

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
    def decode(self, z: "2D np.array") -> "2D np.array":
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

        history = self.model.fit(
            x=spectra,
            y=spectra,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=1,  # progress bar
            use_multiprocessing=True,
            shuffle=True,
        )

        self.train_history["parameters"] = history.params
        self.train_history["history"] = history.history

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
            kl_weight,
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
            tail,
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

        loss = (
            self.reconstruction_weight * reconstruction_loss
            + self.kl_weight * kl_loss
        )

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

        output_layer = Dense(
            units=self.input_dimensions,
            activation=self.out_activation,
            name="decoder_output",
        )

        x = output_layer(decoder_block)

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
            units=number_units,
            activation="relu",
            kernel_initializer=initial_weights,
            name=f"{block}_{layer_index + 1}",
        )

        x = layer(x)

        x = BatchNormalization(name=f"BN_{block}_{layer_index + 1}")(x)

        standard_deviation = np.sqrt(2.0 / number_units)

        return x, standard_deviation

    ###########################################################################
    def _latent_layer(self, x: ""):

        self.mu = Dense(units=self.latent_dimensions, name="mu")(x)

        self.log_variance = Dense(
            units=self.latent_dimensions, name="log_variance"
        )(x)
        #######################################################################
        def sample_normal_distribution(args):

            mu, log_variance = args
            epsilon = K.random_normal(shape=K.shape(mu), mean=0.0, stddev=1.0)

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

    ###########################################################################

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
            self.kl_weight,
            self.out_activation,
            self.train_history,
        ]

        save_location = f"{save_directory}/parameters.pkl"
        with open(save_location, "wb") as file:
            pickle.dump(parameters, file)

    ###########################################################################

    def _save_weights(self, save_directory):
        save_location = f"{save_directory}/weights.h5"
        self.model.save_weights(save_location)

    ###########################################################################

    def _load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    ###########################################################################
    @classmethod
    def load(cls, save_directory):
        parameters_location = f"{save_directory}/parameters.pkl"

        with open(parameters_location, "rb") as file:
            parameters = pickle.load(file)

        autoencoder = VAE(reload=True, parameters=parameters)
        weights_location = f"{save_directory}/weights.h5"
        autoencoder._load_weights(weights_location)
        return autoencoder
