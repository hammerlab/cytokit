import keras
import numpy as np

DEFAULT_CONV_ARGS = {"padding": "same", 'kernel_initializer': 'he_normal'}
DEFAULT_BN_ARGS = {'momentum': 0.9}
DEFAULT_POWERS = [4, 5, 6, 7, 8]  # Feature map depths 16 - 256


def get_model_core(n_class, input_shape, p=DEFAULT_POWERS, ks=(3, 3), batch_norm=True, dropout=None,
                   conv_activation=keras.layers.activations.relu,
                   conv_args=DEFAULT_CONV_ARGS,
                   bn_args=DEFAULT_BN_ARGS):
    n = len(p)
    if dropout is not None and np.isscalar(dropout):
        dropout = [dropout] * n

    if dropout is not None and len(dropout) != n:
        raise ValueError('Must provide {} dropout probabilities (one for each layer) not {}'.format(n, len(dropout)))

    x = keras.layers.Input(shape=input_shape)

    blocks = []
    for i in range(n):
        # Initialize to either input or max pool of last layer (note that this
        # means the final layer added has no pooling)
        l = x if len(blocks) == 0 else keras.layers.MaxPooling2D()(blocks[-1])

        # First conv layer
        l = conv_activation(keras.layers.Conv2D(2**p[i], ks, **conv_args)(l))
        if batch_norm:
            l = keras.layers.BatchNormalization(**bn_args)(l)
        if dropout is not None:
            l = keras.layers.Dropout(dropout[i])(l)

        # Second conv layer
        l = conv_activation(keras.layers.Conv2D(2**p[i], ks, **conv_args)(l))
        if batch_norm:
            l = keras.layers.BatchNormalization(**bn_args)(l)

        blocks.append(l)

    # Loop from second to last power back to first
    for i in range(n-1)[::-1]:

        # Prefer Conv2DTranspose instead of upsampling (e.g. l = keras.layers.UpSampling2D()(blocks[-1]))
        l = keras.layers.Conv2DTranspose(2**p[i], (2, 2), strides=(2, 2), padding='same') (blocks[-1])
        l = keras.layers.merge.concatenate([l, blocks[i]])

        # First conv layer
        l = conv_activation(keras.layers.Conv2D(2**p[i], ks, **conv_args)(l))
        if batch_norm:
            l = keras.layers.BatchNormalization(**bn_args)(l)
        if dropout is not None:
            l = keras.layers.Dropout(dropout[i])(l)

        # Second conv layer
        l = conv_activation(keras.layers.Conv2D(2**p[i], ks, **conv_args)(l))
        if batch_norm:
            l = keras.layers.BatchNormalization(**bn_args)(l)

        blocks.append(l)

    y = conv_activation(keras.layers.Conv2D(n_class, (1, 1), **conv_args)(blocks[-1]))

    return [x, y]


def get_model(n_class, input_shape, activation, **kwargs):

    [x, y] = get_model_core(n_class, input_shape, **kwargs)

    if activation is not None:
        y = keras.layers.Activation(activation)(y)

    model = keras.models.Model(x, y)

    return model
