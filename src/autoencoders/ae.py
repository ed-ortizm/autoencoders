"""Define class to set and reload autoencoders and variational AEs"""

import pickle
from typing import Any, Tuple, Optional, Union

import numpy as np
import keras
from keras import layers
import tensorflow as tf

from sdss.utils.managefiles import FileDirectory

# pylint: disable=W0223
class SamplingLayer(keras.layers.Layer):
    """
    Sampling layer for variational autoencoders (VAEs).

    This custom Keras layer performs the reparameterization trick:
    given the mean and log-variance of a latent Gaussian distribution,
    it returns a sampled latent vector `z`.

    Parameters
    ----------
    name : str, optional
        Name of the layer. Default is "sampleLayer".
    """

    def __init__(self, name: str = "sampleLayer") -> None:
        """
        Initialize the SamplingLayer.

        Parameters
        ----------
        name : str
            The name of the layer.
        """
        super().__init__(name=name)

    # pylint: disable=W0221
    def call(
        self,
        inputs: Tuple[tf.Tensor, tf.Tensor],
        *args: Any,
        **kwargs: Any
    ) -> tf.Tensor:
        """
        Perform the reparameterization trick.

        Parameters
        ----------
        inputs : tuple of tf.Tensor
            A tuple (z_mean, z_log_var) where:
              - z_mean is the mean of the latent Gaussian distribution
              - z_log_var is the log-variance of the same distribution

        Returns
        -------
        tf.Tensor
            A tensor `z` sampled from the distribution
            N(z_mean, exp(z_log_var)).
        """
        z_mean, z_log_var = inputs

        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))

        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def get_config(self) -> dict[str, str]:
        """
        Return the config dictionary for serialization.

        Returns
        -------
        dict
            A dictionary containing the layer configuration.
        """
        return {"name": self.name}

    @classmethod
    def from_config(cls, config: dict[str, str]) -> "SamplingLayer":
        """
        Create a layer instance from a config dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary from `get_config`.

        Returns
        -------
        SamplingLayer
            A new instance of SamplingLayer.
        """
        return cls(**config)

class MyCustomLoss(keras.losses.Loss):
    """
    Custom loss wrapper for scaling a built-in Keras loss function.

    This class allows any standard Keras loss function (e.g., MeanSquaredError)
    to be scaled by a constant factor, making it useful when combining losses
    (e.g., reconstruction loss + KL divergence + MMD in VAEs).

    Parameters
    ----------
    name : str
        Name of the loss function (for tracking/logging).
    keras_loss : keras.losses.Loss
        A built-in or user-defined Keras loss instance to be scaled.
    weight_factor : float, optional
        Multiplier for scaling the loss (default is 1.0).
    """

    def __init__(
        self,
        name: str,
        keras_loss: keras.losses.Loss,
        weight_factor: float = 1.0,
    ) -> None:
        super().__init__(name=name)
        self.keras_loss = keras_loss
        self.weight_factor = weight_factor

    def call(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor
    ) -> tf.Tensor:
        """
        Apply the scaled loss function.

        Parameters
        ----------
        y_true : tf.Tensor
            Ground truth values.
        y_pred : tf.Tensor
            Predicted values.

        Returns
        -------
        tf.Tensor
            Scaled loss value.
        """
        return self.weight_factor * self.keras_loss(y_true, y_pred)

    def get_config(self) -> dict[str, Any]:
        """
        Return a dictionary for serializing this loss object.

        Returns
        -------
        dict
            Configuration dictionary containing:
            - name of the loss
            - string name of the keras_loss class
            - weight_factor used
        """
        return {
            "name": self.name,
            "keras_loss": self.keras_loss.__class__.__name__,
            "weight_factor": self.weight_factor,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MyCustomLoss":
        """
        Reconstruct the custom loss object from its config.

        Parameters
        ----------
        config : dict
            Configuration dictionary returned from `get_config`.

        Returns
        -------
        MyCustomLoss
            A new instance of the custom loss.
        """
        loss_name = config.pop("keras_loss")
        loss_class = getattr(keras.losses, loss_name)
        loss_instance = loss_class()
        return cls(keras_loss=loss_instance, **config)



# class AutoEncoder:
#     def __init__(): ...
#     def _build_model(): ...

#     def _build_encoder(): ...
#     def _sampling_layer(): ...
#     def _build_decoder(): ...
#     def _output_layer(): ...
#     def _add_block(): ...
#     def _get_next_dense_layer_output(): ...

#     def _build_ae(): ...
#     def _compile(): ...

#     def train(): ...
#     def reconstruct(): ...
#     def encode(): ...
#     def decode(): ...

#     def save_model(): ...
#     def _set_class_instances_from_saved_model(): ...

#     def summary(): ...
#     def get_architecture_and_model_str(): ...

#     @staticmethod
#     def compute_mmd(): ...

class AutoEncoder(FileDirectory):
    """
    AutoEncoder class supporting both standard and variational architectures
    using the Keras Functional API.

    This class builds encoder-decoder models with optional KL-divergence and
    Maximum Mean Discrepancy (MMD) regularizations for variational modeling.
    It supports serialization, training, and latent space manipulation.

    Attributes
    ----------
    architecture : dict
        Dictionary describing the model architecture.
    hyperparameters : dict
        Dictionary of hyperparameters used during training.
    model : keras.Model
        Compiled Keras model instance.
    encoder : keras.Model
        The encoder part of the autoencoder.
    decoder : keras.Model
        The decoder part of the autoencoder.
    original_input : tf.Tensor
        The original input placeholder for the Keras model.
    original_output : tf.Tensor
        The decoded output tensor of the model.
    KLD : Optional[tf.Tensor]
        KL divergence loss tensor (if variational).
    MMD : Optional[tf.Tensor]
        Maximum Mean Discrepancy loss tensor (if variational).
    history : Optional[dict]
        Dictionary storing training history.
    """

    def __init__(
        self,
        architecture: Optional[dict] = None,
        hyperparameters: Optional[dict] = None,
        reload: bool = False,
        reload_from: Optional[str] = None,
    ) -> None:
        """
        Initialize the AutoEncoder. Either builds a new model from scratch
        or loads it from saved files if `reload` is True.

        Parameters
        ----------
        architecture : dict, optional
            Dictionary describing encoder/decoder layout and VAE flag.
        hyperparameters : dict, optional
            Dictionary of hyperparameters for training.
        reload : bool, default=False
            If True, loads model and training metadata from disk.
        reload_from : str, optional
            Path to the directory containing saved model and metadata.
        """
        super().__init__()

        if reload:
            keras_model_path = f"{reload_from}/model.keras"
            metadata_path = (
                f"{reload_from}/architecture_hyperparms_train_history.pkl"
            )

            self.model = keras.models.load_model(
                keras_model_path,
                custom_objects={
                    "MyCustomLoss": MyCustomLoss,
                    "SamplingLayer": SamplingLayer,
                },
                compile=False,  # Recompilation will happen later
            )

            self.KLD = None
            self.MMD = None

            [
                self.encoder,
                self.decoder,
                self.architecture,
                self.hyperparameters,
                self.history,
            ] = self._set_class_instances_from_saved_model(metadata_path)

            self.architecture["model_name"] = self.model.name

        else:
            self.architecture = architecture
            self.hyperparameters = hyperparameters

            self.encoder = None
            self.decoder = None
            self.model = None
            self.original_input = None
            self.original_output = None
            self.KLD = None
            self.MMD = None
            self.history = None

            self._build_model()

    def _build_model(self) -> None:
        """
        Constructs the full autoencoder model.

        This method sequentially builds the encoder, decoder,
        connects them to form the full model, and compiles it.
        """
        self._build_encoder()
        self._build_decoder()
        self._build_ae()
        self._compile()

    def _build_encoder(self) -> None:
        """
        Construct the encoder submodel of the AutoEncoder.

        If the model is variational, computes additional symbolic layers
        for KL divergence (KLD) and Maximum Mean Discrepancy (MMD),
        which are later integrated into the full model as losses.

        Sets
        ----
        self.encoder : keras.Model
            Compiled encoder model.
        self.original_input : tf.Tensor
            Input placeholder used to build the full autoencoder.
        self.KLD : tf.Tensor
            Symbolic tensor for KL divergence loss (if variational).
        self.MMD : tf.Tensor
            Symbolic tensor for MMD loss (if variational).
        """
        encoder_input = keras.Input(
            shape=(self.architecture["input_dimensions"],),
            name="encoder_input",
        )
        self.original_input = encoder_input

        block_output = self._add_block(encoder_input, block="encoder")

        if self.architecture["is_variational"]:
            z, z_mean, z_log_var = self._sampling_layer(block_output)

            # Compute KL divergence
            def compute_kld(inputs: list[tf.Tensor]) -> tf.Tensor:
                z_mean, z_log_var = inputs
                return -0.5 * tf.reduce_mean(
                    z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1,
                    axis=1,  # Reduce over feature dimensions only
                )

            raw_kld = keras.layers.Lambda(
                compute_kld, name="kld_loss"
            )([z_mean, z_log_var])

            latent_dim = self.architecture["latent_dimensions"]

            # Sample from the prior for MMD
            def create_true_samples(z: tf.Tensor) -> tf.Tensor:
                batch_size = tf.shape(z)[0]
                return tf.random.normal([batch_size, latent_dim])

            true_samples_layer = keras.layers.Lambda(
                create_true_samples, name="true_samples"
            )(z)

            raw_mmd = keras.layers.Lambda(
                AutoEncoder.compute_mmd, name="mmd_loss"
            )([true_samples_layer, z])

            alpha = self.hyperparameters["alpha"]
            lambda_ = self.hyperparameters["lambda"]

            self.KLD = keras.layers.Lambda(
                lambda x: x * (1 - alpha), name="kld"
            )(raw_kld)

            self.MMD = keras.layers.Lambda(
                lambda x: x * (alpha + lambda_ - 1), name="mmd"
            )(raw_mmd)

        else:
            z_layer = layers.Dense(
                units=self.architecture["latent_dimensions"],
                activation="relu",
                name="z_deterministic",
            )
            z = z_layer(block_output)

        self.encoder = keras.Model(encoder_input, z, name="encoder")

    def _sampling_layer(
        self, encoder_output: tf.Tensor
    ) -> list[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Generate latent space samples for a variational autoencoder.

        Applies the reparameterization trick using the encoder's output
        to produce stochastic latent vectors `z`.

        Parameters
        ----------
        encoder_output : tf.Tensor
            Output of the dense layers in the encoder.

        Returns
        -------
        list of tf.Tensor
            A list containing:
                - z : Sampled latent vector
                - z_mean : Mean of the latent distribution
                - z_log_var : Log variance of the latent distribution
        """
        mu_layer = layers.Dense(
            units=self.architecture["latent_dimensions"], name="z_mean"
        )
        z_mean = mu_layer(encoder_output)

        log_var_layer = layers.Dense(
            units=self.architecture["latent_dimensions"], name="z_log_variance"
        )
        z_log_var = log_var_layer(encoder_output)

        sampling_inputs = (z_mean, z_log_var)
        sample_layer = SamplingLayer(name="z_variational")

        z = sample_layer(sampling_inputs)

        return z, z_mean, z_log_var

    @staticmethod
    def compute_mmd(inputs: list[tf.Tensor]) -> tf.Tensor:
        """
        Compute the symbolic Maximum Mean Discrepancy (MMD) loss between the
        approximate posterior q(z) and a prior p(z) using the kernel method.

        This method is intended to be used inside a Lambda layer, allowing it
        to participate in the computational graph as a symbolic loss term.

        Parameters
        ----------
        inputs : list of tf.Tensor
            A list containing two tensors:
            - true_samples: Samples drawn from the prior distribution p(z)
            - z: Latent vectors sampled from the approximate posteriorq(z|x)

        Returns
        -------
        tf.Tensor
            A symbolic tensor of shape (batch_size, 1) with a constant MMD
            value repeated across the batch, so it integrates seamlessly
            into Keras loss API.

        Notes
        -----
        This implementation uses a Gaussian kernel to compare distributions.
        The MMD value is broadcasted per sample so that Keras can average it
        correctly when used with custom loss functions.
        """
        true_samples, z = inputs

        def compute_kernel(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
            """
            Compute pairwise Gaussian kernel between two batches of vectors.

            Parameters
            ----------
            x, y : tf.Tensor
                Tensors of shape (batch_size, latent_dim)

            Returns
            -------
            tf.Tensor
                Kernel matrix of shape (batch_size, batch_size)
            """
            x_size = tf.shape(x)[0]
            y_size = tf.shape(y)[0]
            dim = tf.shape(x)[1]

            tiled_x = tf.tile(tf.reshape(x, [x_size, 1, dim]), [1, y_size, 1])
            tiled_y = tf.tile(tf.reshape(y, [1, y_size, dim]), [x_size, 1, 1])

            return tf.exp(
                -tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2)
                / tf.cast(dim, tf.float32)
            )

        x_kernel = compute_kernel(true_samples, true_samples)
        y_kernel = compute_kernel(z, z)
        xy_kernel = compute_kernel(true_samples, z)

        mmd = (
            tf.reduce_mean(x_kernel)
            + tf.reduce_mean(y_kernel)
            - 2 * tf.reduce_mean(xy_kernel)
        )

        # Return broadcasted value to match expected shape for loss
        return tf.ones_like(z[:, :1]) * mmd

    def _build_decoder(self):
        """Build decoder"""

        decoder_input = keras.Input(
            shape=(self.architecture["latent_dimensions"],),
            name="decoder_input",
        )

        block_output = self._add_block(decoder_input, block="decoder")

        decoder_output = self._output_layer(block_output)

        self.decoder = keras.Model(
            decoder_input, decoder_output,
            name="reconstruction"
            # name="decoder"
            )

    def _output_layer(self, input_tensor: tf.Tensor) -> tf.Tensor:

        output_layer = layers.Dense(
            units=self.architecture["input_dimensions"],
            activation=self.hyperparameters["output_activation"],
            name="decoder_output",
        )

        output_tensor = output_layer(input_tensor)

        return output_tensor

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
            x = AutoEncoder._get_next_dense_layer_output(
                x, layer_index, number_units, block
            )

        return x

    @staticmethod
    def _get_next_dense_layer_output(
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

        layer = layers.Dense(
            units=number_units,
            activation="relu",
            name=f"{block}_{layer_index + 1:02d}",
        )

        output_tensor = layer(input_tensor)

        return output_tensor


    def _build_ae(self):

        self.original_output = self.decoder(self.encoder(self.original_input))

        if self.architecture["is_variational"] is True:
            self.model = keras.Model(
                inputs=self.original_input,
                outputs={
                    "reconstruction": self.original_output,
                    "kld": self.KLD,
                    "mmd": self.MMD,
                },
                name=self.architecture["model_name"],
            )
        else:
            self.model = keras.Model(
                inputs=self.original_input,
                outputs=self.original_output,
                name=self.architecture["model_name"],
            )

    def _compile(self):

        optimizer = keras.optimizers.Adam(
            learning_rate=self.hyperparameters["learning_rate"]
        )

        reconstruction_weight = self.hyperparameters["reconstruction_weight"]

        MSE = MyCustomLoss(
            name="weighted_MSE",
            keras_loss=keras.losses.MeanSquaredError(),
            weight_factor=reconstruction_weight,
        )
        if self.architecture["is_variational"] is True:

            # pylint: disable=W0613
            def passthrough_loss(y_true, y_pred):
                return tf.reduce_mean(y_pred)

            self.model.compile(
                optimizer=optimizer,
                loss={
                    "reconstruction": MSE,
                    "kld": passthrough_loss,
                    "mmd": passthrough_loss,
                },
                metrics={
                    "reconstruction": ["mse"]
                    # "kld": ["mean"],
                    # "mmd": ["mean"],
                    },
                )
        else:
            self.model.compile(
                optimizer=optimizer, loss=MSE, metrics=["mse"]
                )

    def train(self, spectra: np.array) -> keras.callbacks.History:
        """Train model with spectra array"""

        stopping_criteria = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.hyperparameters["early_stop_patience"],
            verbose=self.hyperparameters["verbose_early_stop"],
            mode="min",
            restore_best_weights=True,
        )

        learning_rate_schedule = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.1,
            patience=self.hyperparameters["learning_rate_patience"],
            verbose=self.hyperparameters["verbose_learning_rate"],
            min_lr=0,
            mode="min",
        )

        callbacks = [stopping_criteria, learning_rate_schedule]

        print("Model output names:", self.model.output_names)
        print("Shapes:")
        for name, out in zip(self.model.output_names, self.model.outputs):
            print(f"{name}: shape={out.shape}")

        history = self.model.fit(
            x=spectra,
            y={
                "reconstruction": spectra,
                "kld": np.zeros((len(spectra), 1), dtype=np.float32),
                "mmd": np.zeros((len(spectra), 1), dtype=np.float32)
            },
            batch_size=self.hyperparameters["batch_size"],
            epochs=self.hyperparameters["epochs"],
            verbose=self.architecture["verbose"],  # 1 for progress bar
            shuffle=True,
            callbacks=callbacks,
            validation_split=self.hyperparameters["validation_split"],
        )

        self.history = history.history

        return history

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

        predicted_spectra = self.model.predict(spectra, verbose=0)

        return predicted_spectra

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

        z = self.encoder.predict(spectra, verbose=0)

        return z

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
            z = z.reshape(1, -1)

        spectra = self.decoder.predict(z)

        return spectra

    def save_model(self, save_to: str) -> None:
        """
        Save the model and training metadata to the specified directory.

        This method saves the full Keras model to a `.keras` file and stores
        training-related information such as architecture, hyperparameters, and
        training history in a separate pickle file.

        Files saved:
            - model.keras : serialized Keras model
            - architecture_hyperparms_train_history.pkl : training metadata

        Parameters
        ----------
        save_to : str
            Path to the directory where the model and metadata should be saved.

        Notes
        -----
        Encoder and decoder models are not saved separately as they are
            included in the full model structure and can be accessed as
            submodules.
        """

        super().check_directory(save_to, exit_program=False)

        keras_model_path = f"{save_to}/model.keras"
        self.model.save(keras_model_path)

        parameters = [self.architecture, self.hyperparameters, self.history]

        with open(
            f"{save_to}/architecture_hyperparms_train_history.pkl", "wb"
            ) as file:
            pickle.dump(parameters, file)

    def _set_class_instances_from_saved_model(
        self, metadata_path: str
    ) -> list:
        """
        Load encoder, decoder, and training metadata from saved model.

        Parameters
        ----------
        metadata_path : str
            Full path to the .pkl file containing training metadata.

        Returns
        -------
        list
            [encoder, decoder, architecture, hyperparameters, train_history]
        """

        encoder = None
        decoder = None

        for submodule in self.model.submodules:
            if isinstance(submodule, keras.Model):
                if submodule.name == "encoder":
                    encoder = submodule
                elif submodule.name == "decoder":
                    decoder = submodule

        with open(metadata_path, "rb") as file:
            architecture, hyperparameters, train_history = pickle.load(file)

        return [encoder, decoder, architecture, hyperparameters, train_history]

    def summary(self):
        """Return Keras buitl int summary of Model class"""
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def get_architecture_and_model_str(self) -> list:
        """
        Retrieve model architecture and name, e.g:
        [512_256_10_256_512, infoVae_rec_3458_alpha_1_lambda_10
        """

        architecture_str = self.architecture["encoder"]
        architecture_str += [self.architecture["latent_dimensions"]]
        architecture_str += self.architecture["decoder"]
        architecture_str = "_".join(str(unit) for unit in architecture_str)

        model_name = (
            f"{self.architecture['model_name']}"
            f"_rec_{self.hyperparameters['reconstruction_weight']:1.0f}"
            f"_alpha_{self.hyperparameters['alpha']:1.0f}"
            f"_lambda_{self.hyperparameters['lambda']:1.0f}"
        )

        return [architecture_str, model_name]
