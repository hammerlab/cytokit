import keras
import numpy as np

DEFAULT_CONV_ARGS = {'activation': 'relu', 'padding': 'same'}
DEFAULT_BN_ARGS = {'momentum': 0.9}


def get_model_core(n_class, input_shape, ks=(3, 3),
                   conv_args=DEFAULT_CONV_ARGS,
                   bn_args=DEFAULT_BN_ARGS):
    x = keras.layers.Input(shape=input_shape)

    a = keras.layers.Conv2D(64, ks, **conv_args)(x)
    a = keras.layers.BatchNormalization(**bn_args)(a)

    a = keras.layers.Conv2D(64, ks, **conv_args)(a)
    a = keras.layers.BatchNormalization(**bn_args)(a)

    y = keras.layers.MaxPooling2D()(a)

    b = keras.layers.Conv2D(128, ks, **conv_args)(y)
    b = keras.layers.BatchNormalization(**bn_args)(b)

    b = keras.layers.Conv2D(128, ks, **conv_args)(b)
    b = keras.layers.BatchNormalization(**bn_args)(b)

    y = keras.layers.MaxPooling2D()(b)

    c = keras.layers.Conv2D(256, ks, **conv_args)(y)
    c = keras.layers.BatchNormalization(**bn_args)(c)

    c = keras.layers.Conv2D(256, ks, **conv_args)(c)
    c = keras.layers.BatchNormalization(**bn_args)(c)

    y = keras.layers.MaxPooling2D()(c)

    d = keras.layers.Conv2D(512, ks, **conv_args)(y)
    d = keras.layers.BatchNormalization(**bn_args)(d)

    d = keras.layers.Conv2D(512, ks, **conv_args)(d)
    d = keras.layers.BatchNormalization(**bn_args)(d)

    # UP

    d = keras.layers.UpSampling2D()(d)

    y = keras.layers.merge.concatenate([d, c])

    e = keras.layers.Conv2D(256, ks, **conv_args)(y)
    e = keras.layers.BatchNormalization(**bn_args)(e)

    e = keras.layers.Conv2D(256, ks, **conv_args)(e)
    e = keras.layers.BatchNormalization(**bn_args)(e)

    e = keras.layers.UpSampling2D()(e)

    y = keras.layers.merge.concatenate([e, b])

    f = keras.layers.Conv2D(128, ks, **conv_args)(y)
    f = keras.layers.BatchNormalization(**bn_args)(f)

    f = keras.layers.Conv2D(128, ks, **conv_args)(f)
    f = keras.layers.BatchNormalization(**bn_args)(f)

    f = keras.layers.UpSampling2D()(f)

    y = keras.layers.merge.concatenate([f, a])

    y = keras.layers.Conv2D(64, ks, **conv_args)(y)
    y = keras.layers.BatchNormalization(**bn_args)(y)

    y = keras.layers.Conv2D(64, ks, **conv_args)(y)
    y = keras.layers.BatchNormalization(**bn_args)(y)

    y = keras.layers.Conv2D(n_class, (1, 1), **conv_args)(y)

    return [x, y]


def get_model(n_class, input_shape, activation='softmax', **kwargs):

    [x, y] = get_model_core(n_class, input_shape, **kwargs)

    if activation is not None:
        y = keras.layers.Activation(activation)(y)

    model = keras.models.Model(x, y)

    return model
