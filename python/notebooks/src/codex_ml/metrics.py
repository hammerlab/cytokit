
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