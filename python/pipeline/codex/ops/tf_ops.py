
import tensorflow as tf


def coord_max(t):
    idx = tf.argmax(tf.reshape(t, [-1]), output_type=tf.int32)
    return tf.squeeze(tf.unravel_index(idx, tf.shape(t)))
