"""Custom objects for infoVAE"""
import keras
import tensorflow as tf

# pylint: disable=W0223
class SamplingLayer(keras.layers.Layer):
    """
    Sampling layer for variational autoencoders.
    Uses (z_mean, z_log_variance) to sample z from the latent distribution.
    """

    def __init__(self, name: str = "sampleLayer"):
        super().__init__(name=name)

    def call(self, inputs, *args, **kwargs):
        z_mean, z_log_var = inputs

        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))

        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def get_config(self):
        return {"name": self.name}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class MyCustomLoss(keras.losses.Loss):
    """
    Create custom loss function for autoencoders using a built-in
    Keras loss function (e.g., MeanSquaredError), with a scaling factor.
    """

    def __init__(
        self,
        name: str,
        keras_loss: keras.losses.Loss,
        weight_factor: float = 1.0,
    ):
        super().__init__(name=name)
        self.keras_loss = keras_loss
        self.weight_factor = weight_factor

    def call(self, y_true, y_pred):
        return self.weight_factor * self.keras_loss(y_true, y_pred)

    def get_config(self):
        return {
            "name": self.name,
            # save class name
            "keras_loss": self.keras_loss.__class__.__name__,
            "weight_factor": self.weight_factor,
        }

    @classmethod
    def from_config(cls, config):
        loss_name = config.pop("keras_loss")
        loss_class = getattr(keras.losses, loss_name)
        loss_instance = loss_class()
        return cls(keras_loss=loss_instance, **config)
