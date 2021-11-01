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
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

###############################################################################
class AE:
    """Create a variational autoencoder"""
    def __init__(
        self,
        architecture: "dict"={None},
        hyperparameters: "dict"={None},
        reload=False
    ) -> "AE object":

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
            }

        """

        self.encoder = None
        self.decoder = None
        self.model = None


        if not reload:

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
            ] = self._get_hyperparameters(hyperparameters)


            self._build()

    ###########################################################################
    @classmethod
    def load(cls, save_directory):

        ae = AE(reload=True)
        encoder_directory = f"{save_directory}/encoder"
        ae.encoder = load_model(encoder_directory)

        decoder_directory = f"{save_directory}/decoder"
        ae.decoder = load_model(decoder_directory)

        model_directory = f"{save_directory}/ae"
        ae.model = load_model(model_directory)

        return ae
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
        self.model.fit(
            x=spectra,
            y=spectra,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=1, # progress bar
            shuffle=True
        )

    ###########################################################################
    def _get_hyperparameters(self, hyperparameters: "dict"):

        batch_size = int(hyperparameters["batch_size"])
        epochs = int(hyperparameters["epochs"])
        learning_rate = float(hyperparameters["learning_rate"])
        out_activation = hyperparameters["out_activation"]

        return [
            batch_size,
            epochs,
            learning_rate,
            out_activation,
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
        self._build_ae()
        self._compile()

    ###########################################################################
    def _compile(self):

        optimizer = Adam(learning_rate=self.learning_rate)

        self.model.compile(
            optimizer=optimizer,
            loss="mse",
            metrics=["mse"]
        )

    ###########################################################################
    def _build_ae(self):

        input = self._model_input
        output = self.decoder(self.encoder(input))
        self.model = Model(input, output, name="ae")

    ###########################################################################
    def _build_decoder(self):

        decoder_input = Input(
            shape=(self.latent_dimensions,),
            name="decoder_input"
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

            block_units = self.encoder_units

        else:

            block_units = self.decoder_units

        for layer_index, number_units in enumerate(block_units):

            x = self._add_layer(
                x,
                layer_index,
                number_units,
                block
            )

        return x

    ###########################################################################
    def _add_layer(
        self,
        x: "keras.Dense",
        layer_index: "int",
        number_units: "int",
        block: "str",
    ):

        layer = Dense(
            units=number_units,
            activation="relu",
            name=f"{block}_{layer_index + 1}",
        )

        x = layer(x)

        return x

    ###########################################################################
    def _latent_layer(self, encoder_block: "keras.Dense"):
        """
        PARAMETERS
            encoder_block

        """

        latent_layer = Dense(
            units=self.latent_dimensions,
            activation="linear",
            name="latent_layer"
        )

        z = latent_layer(encoder_block)

        return z

    ###########################################################################
    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    ###########################################################################
    def save_model(self, directory: "str"):

        directory = f"{directory}/{self.architecture_str}"

        self.model.save(f"{directory}/ae")
        self.encoder.save(f"{directory}/encoder")
        self.decoder.save(f"{directory}/decoder")

###############################################################################
