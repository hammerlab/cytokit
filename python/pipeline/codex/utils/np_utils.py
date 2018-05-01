"""Numpy array utilities"""
import numpy as np


def arr_to_uint(img, dtype):
    """ Convert image array to unsigned integer data type with no scaling """
    if not np.issubdtype(dtype, np.unsignedinteger):
        raise ValueError(
            'Only unsigned integer arrays are valid for conversion; '
            'type given = {}'.format(dtype)
        )
    minv, maxv = np.iinfo(dtype).min, np.iinfo(dtype).max
    return np.clip(img, minv, maxv).astype(dtype)
