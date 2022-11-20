"""Custom objects for infoVAE"""
import tensorflow as tf
from tensorflow import keras


# taken from tf tutorias website
class SamplingLayer(keras.layers.Layer):
    """
    Uses (z_mean, z_log_variance) to sample z, the latent vector.
    And compute kl_divergence
    """

    def __init__(self, name: str = "sampleLayer"):

        super(SamplingLayer, self).__init__(name=name)

    @staticmethod
    def call(inputs):
        """Call method"""

        z_mean, z_log_var = inputs

        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))

        z = keras.backend.exp(0.5 * z_log_var) * epsilon
        z += z_mean

        return z

    def get_config(self):
        """get_config method"""

        return {"name": self.name}

    # necessary to serialize the custom loss
    @classmethod
    def from_config(cls, config):
        """from_config method"""
        return cls(**config)


class MyCustomLoss(keras.losses.Loss):
    """
    Create custom loss function for autoencoders using prebuilt losses
    in keras.losse, for instance, keras.losses.MSE
    """

    ###########################################################################
    def __init__(
        self,
        name: str,
        keras_loss: keras.losses.Loss,
        weight_factor: float = 1.0,
    ):
        """
        PARAMETERS
            keras_loss: a builtin keras loss function, for instance,
                tf.losses.MSE
            name: name of the custom function
            regularization_factor: weight factor for keras_loss function
        """
        super().__init__(name=name)
        self.keras_loss = keras_loss
        self.weight_factor = weight_factor

    def call(self, y_true, y_pred):
        """Call method"""

        loss = self.weight_factor * self.keras_loss(y_true, y_pred)

        return loss

    ###########################################################################
    # necessary to serialize the custom loss
    def get_config(self):
        """get_config method"""
        return {
            "name": self.name,
            "keras_loss": self.keras_loss,
            "weight_factor": self.weight_factor,
        }

    ###########################################################################
    # necessary to serialize the custom loss
    @classmethod
    def from_config(cls, config):
        """from_config method"""
        return cls(**config)
