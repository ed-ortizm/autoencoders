import tensorflow as tf
from tensorflow import keras
###############################################################################
class MyCustomMetric(keras.metrics.Metric):
    def __init__(self, name="my_custom_metric", **kwargs):
        super(MyCustomMetric, self).__init__(name=name, **kwargs)
        pass

    def update_state(self, y_true, y_pred, sample_weight=None):
        # return
        pass

    def result(self):
        pass

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        pass

###############################################################################
class MyCustomLoss(keras.losses.Loss):
    """
    Create custom loss function for autoencoders using prebuilt losses
    in keras.losse, for instance, keras.losses.MSE
    """
    ###########################################################################
    def __init__(self,
        name: str,
        keras_loss: function,
        regularization_factor: float=1.,
    ):
        """
        PARAMETERS
            keras_loss: a builtin keras loss function, for instance,
                tf.losses.MSE
            name: name of the custom function
            regularization_factor: weigth factor for keras_loss function
        """
        super().__init__(name=name)
        self.keras_loss = keras_loss
        self.regularization_factor = regularization_factor

    ###########################################################################
    def call(self, y_true, y_pred):
        return self.regularization_factor * self.keras_loss(y_true, y_pred)
    ###########################################################################
    # necessary to serialize the custom loss
    def get_config(self):
        return {
            "name" : self.name,
            "keras_loss": self.keras_loss,
            "regularization_factor" : self.regularization_factor
        }
    ###########################################################################
    # necessary to serialize the custom loss
    @classmethod
    def from_config(cls, config):
        return cls(**config)
