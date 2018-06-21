import numpy as np
import tensorflow as tf
import keras

DEFAULT_OPT_CONV = {"activation": "relu", "padding": "same", 'kernel_initializer': 'he_normal'}
DEFAULT_OPT_BN = {'momentum': 0.9}

def get_model_core(input_shape, p=[6,7,8,9], ks=(3, 3), batch_norm=True, dropout=None, 
                   opt_conv=DEFAULT_OPT_CONV, opt_bn=DEFAULT_OPT_BN):
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
        l = keras.layers.Conv2D(2**p[i], ks, **opt_conv)(l)
        if batch_norm:
            l = keras.layers.BatchNormalization(**opt_bn)(l)
        if dropout is not None:
            l = keras.layers.Dropout(dropout[i])(l)
            
        # Second conv layer
        l = keras.layers.Conv2D(2**p[i], ks, **opt_conv)(l)
        if batch_norm:
            l = keras.layers.BatchNormalization(**opt_bn)(l)

        blocks.append(l)
        
    # Loop from second to last power back to first
    for i in range(n-1)[::-1]:
        
        # Prefer Conv2DTranspose instead of upsampling (e.g. l = keras.layers.UpSampling2D()(blocks[-1]))
        l = keras.layers.Conv2DTranspose(2**p[i], (2, 2), strides=(2, 2), padding='same') (blocks[-1])
        l = keras.layers.merge.concatenate([l, blocks[i]])
        
        # First conv layer
        l = keras.layers.Conv2D(2**p[i], ks, **opt_conv)(l)
        if batch_norm:
            l = keras.layers.BatchNormalization(**opt_bn)(l)
        if dropout is not None:
            l = keras.layers.Dropout(dropout[i])(l)
            
        # Second conv layer
        l = keras.layers.Conv2D(2**p[i], ks, **opt_conv)(l)
        if batch_norm:
            l = keras.layers.BatchNormalization(**opt_bn)(l)
            
        blocks.append(l)
        
    return [x, blocks[-1]]


def get_model(nclass, input_shape, activation='softmax', opt_conv=DEFAULT_OPT_CONV, **kwargs):

    [x, y] = get_model_core(input_shape, opt_conv=opt_conv, **kwargs)

    y = keras.layers.Conv2D(nclass, (1, 1), **opt_conv)(y)

    if activation is not None:
        y = keras.layers.Activation(activation)(y)

    model = keras.models.Model(x, y)

    return model

def get_nuclei_model_v1(input_shape):
    return get_model(3, input_shape, activation='sigmoid', p=[4,5,6,7,8], batch_norm=True, dropout=None)
    

def class_weighted_softmax_cross_entropy_loss(weights):
    w = weights
    # Class values are: bg, nucleus interior, nucleus border
    def loss(y_true, y_pred):
        class_weights = tf.constant([[[w]]])
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred)
        weights = tf.reduce_sum(class_weights * y_true, axis=-1)
        weighted_losses = weights * unweighted_losses
        return tf.reduce_mean(weighted_losses)
    return loss


def weighted_pixelwise_crossentropy(weights):
    """Get pixel-weighted sigmoid cross entropy loss"""
    
    # Assume that weights could cover any dimension after the 
    # batch dimension (i.e. the first one) -- from there, it
    # should simply be broadcastable to the dimensions of a single
    # prediction/label (which for images may be 3D)
    w = [weights]
    
    def loss(y_true, y_pred):
        _epsilon = 10e-8
        y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)
        return - tf.reduce_mean(y_true * w *  tf.log(y_pred) + (1 - y_true)  *  tf.log(1 - y_pred))
    return loss


def plot_model_history(history, keys=None, ncol=2, width=8, height=2, window=None):
    import plotnine as pn
    import pandas as pd
    import numpy as np
    
    if keys is None:
        keys = sorted(list(history.keys()))
    n = len(keys)
    nrow = int(np.ceil(n / ncol))
    figsize = (ncol * width, nrow * height)
        
    if window is None:
        window = slice(None)
    df = pd.concat([
        pd.DataFrame(dict(epoch=np.array(range(len(h)))[window], value=h[window])).assign(key=k)
        for k, h in history.items()
    ])
    return (
        pn.ggplot(df, pn.aes(x='epoch', y='value')) +
        pn.geom_line() + 
        pn.facet_wrap('~key', ncol=ncol, scales='free') +
        pn.theme(figure_size=figsize)
    )


def binary_channel_precision(channel, name, min_p=.5):
    from keras import backend as K
    t_channel = channel
    t_min_p = min_p
    
    def metric_fn(y_true, y_pred):
        y_pred_class = K.cast(y_pred[..., t_channel] > t_min_p, "float32")
        y_true_class = K.cast(y_true[..., t_channel], "float32")
        true_positives = K.sum(K.round(K.clip(y_pred_class * y_true_class, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred_class, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    metric_fn.__name__ = name
    return metric_fn



# def get_model_core(input_shape, p, ks=(3, 3)):
#     if len(p) != 4:
#         raise ValueError('Expecting 4 power of 2 layer depths, given power list = {}'.format(p))
#     
#     x = keras.layers.Input(shape=input_shape)
#         
#     a = keras.layers.Conv2D(2**p[0], ks, **opt_conv)(x)
#     a = keras.layers.BatchNormalization(**opt_bn)(a)
# 
#     a = keras.layers.Conv2D(2**p[0], ks, **opt_conv)(a)
#     a = keras.layers.BatchNormalization(**opt_bn)(a)
# 
#     y = keras.layers.MaxPooling2D()(a)
# 
#     b = keras.layers.Conv2D(2**p[1], ks, **opt_conv)(y)
#     b = keras.layers.BatchNormalization(**opt_bn)(b)
# 
#     b = keras.layers.Conv2D(2**p[1], ks, **opt_conv)(b)
#     b = keras.layers.BatchNormalization(**opt_bn)(b)
# 
# 
#     y = keras.layers.MaxPooling2D()(b)
# 
#     c = keras.layers.Conv2D(2**p[2], ks, **opt_conv)(y)
#     c = keras.layers.BatchNormalization(**opt_bn)(c)
# 
#     c = keras.layers.Conv2D(2**p[2], ks, **opt_conv)(c)
#     c = keras.layers.BatchNormalization(**opt_bn)(c)
# 
# 
#     y = keras.layers.MaxPooling2D()(c)
# 
#     d = keras.layers.Conv2D(2**p[3], ks, **opt_conv)(y)
#     d = keras.layers.BatchNormalization(**opt_bn)(d)
# 
#     d = keras.layers.Conv2D(2**p[3], ks, **opt_conv)(d)
#     d = keras.layers.BatchNormalization(**opt_bn)(d)
# 
# 
#     # UP
# 
#     d = keras.layers.UpSampling2D()(d)
# 
#     y = keras.layers.merge.concatenate([d, c], axis=3)
#     #y = keras.layers.merge([d, c], concat_axis=3, mode="concat")
# 
#     e = keras.layers.Conv2D(2**p[2], ks, **opt_conv)(y)
#     e = keras.layers.BatchNormalization(**opt_bn)(e)
# 
#     e = keras.layers.Conv2D(2**p[2], ks, **opt_conv)(e)
#     e = keras.layers.BatchNormalization(**opt_bn)(e)
# 
#     e = keras.layers.UpSampling2D()(e)
# 
# 
#     y = keras.layers.merge.concatenate([e, b], axis=3)
#     #y = keras.layers.merge([e, b], concat_axis=3, mode="concat")
# 
#     f = keras.layers.Conv2D(2**p[1], ks, **opt_conv)(y)
#     f = keras.layers.BatchNormalization(**opt_bn)(f)
# 
#     f = keras.layers.Conv2D(2**p[1], ks, **opt_conv)(f)
#     f = keras.layers.BatchNormalization(**opt_bn)(f)
# 
#     f = keras.layers.UpSampling2D()(f)
# 
#     y = keras.layers.merge.concatenate([f, a], axis=3)
#     #y = keras.layers.merge([f, a], concat_axis=3, mode="concat")
# 
#     y = keras.layers.Conv2D(2**p[0], ks, **opt_conv)(y)
#     y = keras.layers.BatchNormalization(**opt_bn)(y)
# 
#     y = keras.layers.Conv2D(2**p[0], ks, **opt_conv)(y)
#     y = keras.layers.BatchNormalization(**opt_bn)(y)
# 
#     return [x, y]


# def get_pretrained_model(nclass, input_shape, activation='softmax'):
#     from keras.applications.vgg16 import VGG16
#     vgg_model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
# 
#     # Creating dictionary that maps layer names to the layers
#     layers = dict([(layer.name, layer) for layer in vgg_model.layers])
# 
#     # (None, 256, 256, 64)
#     block1_conv2 = layers['block1_conv2'].output
# 
#     # (None, 128, 128, 128)
#     block2_conv2 = layers['block2_conv2'].output
# 
#     # (None, 64, 64, 256) 
#     block3_conv3 = layers['block3_conv3'].output
# 
#     # (None, 32, 32, 512) 
#     block4_conv3 = layers['block4_conv3'].output
# 
#     # (None, 16, 16, 512)
#     block5_conv3 = layers['block5_conv3'].output
# 
#     l = keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(block5_conv3)
#     l = keras.layers.merge.concatenate([l, block4_conv3])
# 
#     l = keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(block5_conv3)
#     l = keras.layers.merge.concatenate([l, block4_conv3])
# 
#     # Stacking the remaining layers of U-Net on top of it (modified from
#     # the U-Net code you provided)
#     up6 = Concatenate()([UpSampling2D(size=(2, 2))(vgg_top), block4_conv3])
#     conv6 = make_conv_block(256, up6, 6)
#     up7 = Concatenate()([UpSampling2D(size=(2, 2))(conv6), block3_conv3])
#     conv7 = make_conv_block(128, up7, 7)
#     up8 = Concatenate()([UpSampling2D(size=(2, 2))(conv7), block2_conv2])
#     conv8 = make_conv_block(64, up8, 8)
#     up9 = Concatenate()([UpSampling2D(size=(2, 2))(conv8), block1_conv2])
#     conv9 = make_conv_block(32, up9, 9)
#     conv10 = Conv2D(nb_labels, (1, 1), name='conv_10_1')(conv9)
#     x = Reshape((nb_rows * nb_cols, nb_labels))(conv10)
#     x = Activation('softmax')(x)
#     outputs = Reshape((nb_rows, nb_cols, nb_labels))(x)