import fire
import os
import os.path as osp
import numpy as np
import pandas as pd
from codex import config as codex_config
from codex.ops import cytometry
from codex.ops import analysis
from codex.ops import tile_generator
from codex.ops import tile_crop
from codex import io as codex_io
from codex import cli
import logging
logging.basicConfig(level=logging.INFO, format=cli.LOG_FORMAT)

CH_SRC_RAW = 'raw'
CH_SRC_PROC = 'proc'
CH_SRC_SEGM = 'segm'
CH_SOURCES = [CH_SRC_RAW, CH_SRC_PROC, CH_SRC_SEGM]


def _get_channel_source(channel):
    res = None
    for src in CH_SOURCES:
        if channel.startswith(src + '_'):
            res = src
    return res


def _map_channels(config, channels):
    res = []
    for channel in channels:
        src = _get_channel_source(channel)
        if src is None:
            raise ValueError(
                'Channel with name "{}" is not valid.  Must start with one of the following: {}'
                .format(channel, [c + '_' for c in CH_SOURCES])
            )
        channel = '_'.join(channel.split('_')[1:])
        if src == CH_SRC_RAW or src == CH_SRC_PROC:
            coords = config.get_channel_coordinates(channel)
            res.append([channel, src, coords[0], coords[1]])
        elif src == CH_SRC_SEGM:
            coords = cytometry.get_channel_coordinates(channel)
            res.append([channel, src, coords[0], coords[1]])
        else:
            raise AssertionError('Source "{}" is invalid'.format(src))
    return pd.DataFrame(res, columns=['channel_name', 'source', 'cycle_index', 'channel_index'])


def _get_z_slice_fn(z):
    """Get array slice map to be applied to z dimension

    Args:
        z: One of the following:
            - "best": indicates that z slices should be inferred based on focal quality
            - "all": indicates that a slice for all z-planes should be used
            - A single integer
            - An integer list of indexes
            - A 2-item or 3-item tuple forming the slice (start, stop[, step])
    Returns:
        A function with signature (region_index, tile_x, tile_y) -> slice_for_array where slice_for_array
        will either be a true slice or a list of z-indexes (Note: all indexes are 0-based)
    """
    # Look for keyword strings
    if isinstance(z, str) and z == 'best':
        map = analysis.get_best_focus_coord_map(self.data_dir)
        return lambda ri, tx, ty: [map[(ri, tx, ty)]]
    if isinstance(z, str) and z == 'all':
        return lambda ri, tx, ty: slice(None)

    # Look for tuple based slices
    if isinstance(z, tuple):
        if 2 <= len(z) <= 3:
            raise ValueError('When specifying z-slice as a tuple, it must contain 2 or 3 items (not {})'.format(z))
        slicer = slice(*[int(v) for v in z])
        return lambda ri, tx, ty: slicer

    # Look for direct list assignments
    if isinstance(z, list):
        return lambda ri, tx, ty: z

    # Otherwise, attempt to convert argument to integer and return as single-item list
    return lambda ri, tx, ty: [int(z)]


def _get_tile_locations(config, region_indexes, tile_indexes):
    res = []
    for tile_location in config.get_tile_indices:
        if region_indexes is not None and tile_location.region_index not in region_indexes:
            continue
        if tile_indexes is not None and tile_location.tile_index not in tile_indexes:
            continue
        res.append(tile_location)
    return res


class Operator(object):

    def __init__(self, config_path, data_dir):
        self.config = codex_config.load(config_path)
        self.data_dir = data_dir

    def extract(self, name, channels, z='best', region_indexes=None, tile_indexes=None):
        channel_map = _map_channels(self.config, channels).groupby('source')
        channel_sources = sorted(list(channel_map.groups.keys()))

        z_slice_fn = _get_z_slice_fn(z)
        region_indexes = cli.resolve_int_list_arg(region_indexes)
        tile_indexes = cli.resolve_int_list_arg(tile_indexes)

        logging.info('Creating extraction "{}" ...'.format(name))

        tile_locations = _get_tile_locations(self.config, region_indexes, tile_indexes)

        for i, loc in tile_locations:
            logging.info('Extracting tile {} of {}'.format(i+1, len(tile_locations)))
            extract_tile = []
            for src in channel_sources:
                generator = tile_generator.CodexTileGenerator(
                    self.config, self.data_dir, loc.region_index, loc.tile_index,
                    mode='raw' if src == CH_SRC_RAW else 'stack'
                )
                tile = generator.run(None)

                # Crop raw images if necessary
                if src == CH_SRC_RAW:
                    tile = tile_crop.CodexTileCrop(self.config).run(tile)

                for _, r in channel_map.get_group(src).iterrows():
                    z_slice = z_slice_fn(loc.region_index, loc.tile_x, loc.tile_y)
                    # Extract (z, h, w) subtile
                    subtile = tile[r['cycle_index'], z_slice, r['channel_index']]
                    assert subtile.ndims == 3, \
                        'Expecting subtile have 3 dimensions but got shape {}'.format(subtile.shape)
                    extract_tile.append(subtile)

            # Stack the subtiles to give array with shape (z, channels, h, w) and then reshape to 5D
            # format like (cycles, z, channels, h, w)
            extract_tile = np.stack(extract_tile, axis=1)[np.newaxis]

            path = codex_io.get_extract_image_path(log.region_index, log.tile_x, loc.tile_y, name)
            codex_io.save_tile(osp.join(self.data_dir, path), extract_tile)

        logging.info('Extraction complete')