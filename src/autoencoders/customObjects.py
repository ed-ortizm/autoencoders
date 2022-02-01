import tensorflow as tf
from tensorflow import keras

###############################################################################
# taken from tf tutorias website
class SamplingLayer(keras.layers.Layer):
    """
    Uses (z_mean, z_log_variance) to sample z, the latent vector.
    And compute kl_divergence
    """

    ###########################################################################
    def __init__(self, name: str = "sampleLayer"):

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

    ###########################################################################
    def call(self, y_true, y_pred):

        loss = self.weight_factor * self.keras_loss(y_true, y_pred)

        return loss

    ###########################################################################
    # necessary to serialize the custom loss
    def get_config(self):
        return {
            "name": self.name,
            "keras_loss": self.keras_loss,
            "weight_factor": self.weight_factor,
        }

    ###########################################################################
    # necessary to serialize the custom loss
    @classmethod
    def from_config(cls, config):
        return cls(**config)

###############################################################################
# class MyCustomMetric(keras.metrics.Metric):
#     def __init__(self, name: str="customMetric", **kwargs):
#         super(MyCustomMetric, self).__init__(name=name, **kwargs)
#         pass
#
#     def update_state(self, y_true: np.array, y_pred:np.array, sample_weight: float=None):
#         # return
#         pass
#
#     def result(self):
#         pass
#
#     def reset_state(self):
#         # The state of the metric will be reset at the start of each epoch.
#         pass
###############################################################################
