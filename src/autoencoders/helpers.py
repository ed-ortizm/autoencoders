import json


def get_list(str: "get.parser.parameter"):
    return json.loads(str)


def get_parameters_for_autoencoder(
    hyperparameters: "dict",
    architecture: "dict",
    )->"list":
    """
    Get hyperparameters of the autoencoder

    PARAMETERS
        hyperparameters: dictionary containing hyperparameters of the network
        architecture: dictionary containing architecture of the network
    """

    pass
    #
    # input_shape = get_list(parser.get("architecture", "input"))
    #
    # encoder_filters = get_list(parser.get("encoder", "filters"))
    # encoder_kernels = get_list(parser.get("encoder", "kernels"))
    # encoder_strides = get_list(parser.get("encoder", "strides"))
    #
    # latent = parser.getint("architecture", "latent")
    #
    # decoder_filters = get_list(parser.get("decoder", "filters"))
    # decoder_kernels = get_list(parser.get("decoder", "kernels"))
    # decoder_strides = get_list(parser.get("decoder", "strides"))
    # return [
    #     input_shape,
    #     encoder_filters,
    #     encoder_kernels,
    #     encoder_strides,
    #     latent,
    #     decoder_filters,
    #     decoder_kernels,
    #     decoder_strides
    # ]
