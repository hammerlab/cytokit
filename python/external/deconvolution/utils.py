import os
import re
import tensorflow as tf
from os import path as osp
import numpy as np


def get_files(directory, pattern):
    return [
        osp.join(directory, f) for f in os.listdir(directory)
        if re.match(pattern, f)
    ]


def validate_stack_shape(img, config):
    if img.ndim != 5:
        raise ValueError(
            'Expecting 5 dimensions in image stack, '
            'given shape = {}'.format(img.shape)
        )
    ncyc, nz, nch, nh, nw = img.shape

    if ncyc != config.n_cycles():
        raise ValueError(
            'Expecting {} cycles but found {} in image stack'
            .format(config.n_cycles(), ncyc)
        )

    if nz != config.n_z_planes():
       raise ValueError(
            'Expecting {} z planes but found {} in image stack'
            .format(config.n_z_planes(), nz)
        )

    if nch != config.n_channels_per_cycle():
       raise ValueError(
            'Expecting {} channels but found {} in image stack'
            .format(config.n_channels_per_cycle(), nch)
        )


def arr_to_uint(img, dtype):
    """ Convert image array to data type with no scaling """
    if img.min() < 0:
        raise ValueError('Expecting only positive values in array')
    minv, maxv = np.iinfo(dtype).min, np.iinfo(dtype).max
    return np.clip(img, minv, maxv).astype(dtype)


def disable_tf_logging():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    tf.logging.set_verbosity(tf.logging.WARN)


def get_immersion_ri(immersion):
    """Get refractive index for an immersion type"""
    if immersion == 'air':
        return 1.0
    elif immersion == 'water':
        return 1.33
    elif immersion == 'oil':
        return 1.5115
    else:
        raise ValueError('Immersion "{}" is not valid (must be air, water, or oil)'.format(immersion))