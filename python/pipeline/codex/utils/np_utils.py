"""Numpy array utilities"""
import numpy as np


def arr_to_uint(img, dtype):
    """ Convert image array to data type with no scaling """
    if img.min() < 0:
        raise ValueError('Expecting only positive values in array')
    minv, maxv = np.iinfo(dtype).min, np.iinfo(dtype).max
    return np.clip(img, minv, maxv).astype(dtype)
