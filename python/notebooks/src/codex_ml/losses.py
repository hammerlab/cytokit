

def objective_weighted_binary_crossentropy(weights):
    """Get pixel-weighted sigmoid cross entropy loss"""
    import numpy as np
    import tensorflow as tf

    # Assume that weights could cover any dimension after the
    # batch dimension (i.e. the first one) -- from there, it
    # should simply be broadcastable to the dimensions of a single
    # prediction/label (which for images may be 3D)
    w = np.array([weights])

    # Make sure weights are now 4D
    assert w.ndim == 4

    def loss(y_true, y_pred):
        _epsilon = 10e-8
        y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)
        return - tf.reduce_mean(y_true * w * tf.log(y_pred) + (1 - y_true) * tf.log(1 - y_pred))

    return loss

