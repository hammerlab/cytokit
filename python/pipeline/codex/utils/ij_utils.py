import struct
import sys
import numpy as np


def get_config_resolution_args(config):
    """Get tifffile resolution arguments for an experiment configuration"""
    ms_params = config.microscope_params
    # ImageJ resolution is in pixels per unit instead of units per pixel
    xy, z = 1. / (ms_params.res_lateral_nm / 1000.), 1. / (ms_params.res_axial_nm / 1000.)
    resolution = (xy, xy)
    metadata = {'spacing': z, 'unit': 'um'}
    return resolution, metadata


def get_config_slice_label_args(config, shape):
    """Get tifffile tag arguments for slice labels"""
    if len(shape) != 5:
        raise ValueError('Slice label inference only possible for 5D array shapes (shape given = {})'.format(shape))

    # If tile shape does not match config in cycles and channels dimensions, return nothing
    ncyc, nz, nch, h, w = shape
    if (config.n_cycles, config.n_channels_per_cycle) != (ncyc, nch):
        return None

    # Pull channel names into array and reshape to ncycles x nchannels
    chs = np.array(config.channel_names)
    chs = chs.reshape((ncyc, nch))

    # Loop through block of channels for each cycle/frame and repeat z times
    labels = []
    for cyc_chs in chs:
        # cyc_chs is an array of channel names for a single cycle (i.e. row in chs)
        labels += list(cyc_chs) * nz

    # Return "extratags" argument
    return get_slice_label_tags(labels)


def get_channel_label_tags(labels, z=1, t=1):
    """Get slice labels for channel names to be repeated across Z and T axes of image

    Note: When constructing labels for hyperstacks, the order of the flat label list is applied
    to channels, then Z slices, then frames (i.e. T slices)
    """
    return get_slice_label_tags((labels * z) * t)


def get_slice_label_tags(labels):
    return get_imagej_tags({'Labels': labels})


def get_imagej_tags(metadata, byteorder=None):
    """Return IJMetadata and IJMetadataByteCounts tags from metadata dict.

    The tags can be passed to the TiffWriter.save function as extratags.

    * Source: https://stackoverflow.com/questions/50258287/how-to-specify-colormap-when-saving-tiff-stack
    """
    if byteorder in (None, '=', '|'):
        byteorder = '<' if sys.byteorder == 'little' else '>'
    header = [{'>': b'IJIJ', '<': b'JIJI'}[byteorder]]
    bytecounts = [0]
    body = []

    def writestring(data, byteorder):
        return data.encode('utf-16' + {'>': 'be', '<': 'le'}[byteorder])

    def writedoubles(data, byteorder):
        return struct.pack(byteorder+('d' * len(data)), *data)

    def writebytes(data, byteorder):
        return data.tobytes()

    metadata_types = (
        ('Info', b'info', 1, writestring),
        ('Labels', b'labl', None, writestring),
        ('Ranges', b'rang', 1, writedoubles),
        ('LUTs', b'luts', None, writebytes),
        ('Plot', b'plot', 1, writebytes),
        ('ROI', b'roi ', 1, writebytes),
        ('Overlays', b'over', None, writebytes))

    for key, mtype, count, func in metadata_types:
        if key not in metadata:
            continue
        if byteorder == '<':
            mtype = mtype[::-1]
        values = metadata[key]
        if count is None:
            count = len(values)
        else:
            values = [values]
        header.append(mtype + struct.pack(byteorder+'I', count))
        for value in values:
            data = func(value, byteorder)
            body.append(data)
            bytecounts.append(len(data))

    body = b''.join(body)
    header = b''.join(header)
    data = header + body
    bytecounts[0] = len(header)
    bytecounts = struct.pack(byteorder+('I' * len(bytecounts)), *bytecounts)
    return [
        (50839, 'B', len(data), data, True),
        (50838, 'I', len(bytecounts)//4, bytecounts, True)
    ]
