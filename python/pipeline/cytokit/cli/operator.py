import fire
import os
import os.path as osp
import numpy as np
import pandas as pd
from cytokit import config as cytokit_config
from cytokit.ops import cytometry
from cytokit.ops import tile_generator
from cytokit.ops import tile_crop
from cytokit.utils import ij_utils
from cytokit import io as cytokit_io
from cytokit import cli
from cytokit.function import core
from cytokit.function import data as function_data
from cytokit.cli import CH_SRC_RAW, CH_SRC_PROC, CH_SRC_CYTO, CH_SOURCES
import logging

PATH_FMT_MAP = {
    CH_SRC_RAW: None,
    CH_SRC_PROC: cytokit_io.FMT_PROC_IMAGE,
    CH_SRC_CYTO: cytokit_io.FMT_CYTO_IMAGE
}


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
        elif src == CH_SRC_CYTO:
            coords = cytometry.get_channel_coordinates(channel)
            res.append([channel, src, coords[0], coords[1]])
        else:
            raise AssertionError('Source "{}" is invalid'.format(src))
    return pd.DataFrame(res, columns=['channel_name', 'source', 'cycle_index', 'channel_index'])


def _get_z_slice_fn(z, data_dir):
    """Get array slice map to be applied to z dimension

    Args:
        z: String or 1-based index selector for z indexes constructed as any of the following:
            - "best": Indicates that z slices should be inferred based on focal quality
            - "all": Indicates that a slice for all z-planes should be used
            - str or int: A single value will be interpreted as a single index
            - tuple: A 2-item or 3-item tuple forming the slice (start, stop[, step]); stop is inclusive
            - list: A list of integers will be used as is
        data_dir: Data directory necessary to infer 'best' z planes
    Returns:
        A function with signature (region_index, tile_x, tile_y) -> slice_for_array where slice_for_array
        will either be a slice instance or a list of z-indexes (Note: all indexes are 0-based)
    """
    if not z:
        raise ValueError('Z slice cannot be defined as empty value (given = {})'.format(z))

    # Look for keyword strings
    if isinstance(z, str) and z == 'best':
        map = function_data.get_best_focus_coord_map(data_dir)
        return lambda ri, tx, ty: [map[(ri, tx, ty)]]
    if isinstance(z, str) and z == 'all':
        return lambda ri, tx, ty: slice(None)

    # Parse argument as 1-based index list and then convert to 0-based
    zi = cli.resolve_index_list_arg(z, zero_based=True)
    return lambda ri, tx, ty: zi


def _get_tile_locations(config, region_indexes, tile_indexes):
    res = []
    for tile_location in config.get_tile_indices():
        if region_indexes is not None and tile_location.region_index not in region_indexes:
            continue
        if tile_indexes is not None and tile_location.tile_index not in tile_indexes:
            continue
        res.append(tile_location)
    return res


class Operator(cli.DataCLI):

    def _get_function_configs(self):
        return self.config.operator_params

    def extract(self, name, channels, z='best', region_indexes=None, tile_indexes=None, raw_dir=None):
        """Create a new data extraction include either raw, processed, or cytometric imaging data

        Args:
            name: Name of extraction to be created; This will be used to construct result path like
                EXP_DIR/output/extract/`name`
            channels: List of strings indicating channel names (case-insensitive) prefixed by source for that
                channel (e.g. proc_DAPI, raw_CD4, cyto_nucleus_boundary); Available sources are:
                - "raw": Raw data images
                - "proc": Data generated as a results of preprocessing
                - "cyto": Cytometric object data (nuclei and cell boundaries)
            z: String or 1-based index selector for z indexes constructed as any of the following:
                - "best": Indicates that z slices should be inferred based on focal quality (default option)
                - "all": Indicates that a slice for all z-planes should be used
                - str or int: A single value will be interpreted as a single index
                - tuple: A 2-item or 3-item tuple forming the slice (start, stop[, step]); stop is inclusive
                - list: A list of integers will be used as is
            region_indexes: 1-based sequence of region indexes to process; can be specified as:
                - None: Region indexes will be inferred from experiment configuration
                - str or int: A single value will be interpreted as a single index
                - tuple: A 2-item or 3-item tuple forming the slice (start, stop[, step]); stop is inclusive
                - list: A list of integers will be used as is
            tile_indexes: 1-based sequence of tile indexes to process; has same semantics as `region_indexes`
            raw_dir: If using any channels sourced from raw data, this directory must be specified and should
                be equivalent to the same raw directory used during processing (i.e. nearly all operations like
                this are run relative to an `output_dir` -- the result of processing -- but in this case
                the original raw data path is needed as well)
        """
        channel_map = _map_channels(self.config, channels).groupby('source')
        channel_sources = sorted(list(channel_map.groups.keys()))

        z_slice_fn = _get_z_slice_fn(z, self.data_dir)
        region_indexes = cli.resolve_index_list_arg(region_indexes, zero_based=True)
        tile_indexes = cli.resolve_index_list_arg(tile_indexes, zero_based=True)

        logging.info('Creating extraction "%s"', name)

        tile_locations = _get_tile_locations(self.config, region_indexes, tile_indexes)

        extract_path = None
        for i, loc in enumerate(tile_locations):
            logging.info('Extracting tile {} of {}'.format(i+1, len(tile_locations)))
            extract_tile = []

            # Create function used to crop out z-slices from extracted volumes
            z_slice = z_slice_fn(loc.region_index, loc.tile_x, loc.tile_y)

            slice_labels = []
            for src in channel_sources:

                # Initialize tile generator for this data source (which are all the same except
                # for when using raw data, which does not have pre-assembled tiles available)
                tile_gen_dir = self.data_dir
                tile_gen_mode = 'stack'
                if src == CH_SRC_RAW:
                    if not raw_dir:
                        raise ValueError('When extracting raw data channels, the `raw_dir` argument must be provided')
                    tile_gen_dir = raw_dir
                    tile_gen_mode = 'raw'
                generator = tile_generator.CytokitTileGenerator(
                    self.config, tile_gen_dir, loc.region_index, loc.tile_index,
                    mode=tile_gen_mode, path_fmt_name=PATH_FMT_MAP[src]
                )
                tile = generator.run(None)

                # Crop raw images if necessary
                if src == CH_SRC_RAW:
                    tile = tile_crop.CytokitTileCrop(self.config).run(tile)

                # Sort channels by name to make extract channel order deterministic
                for _, r in channel_map.get_group(src).sort_values('channel_name').iterrows():

                    # Extract (z, h, w) subtile
                    sub_tile = tile[r['cycle_index'], z_slice, r['channel_index']]
                    logging.debug(
                        'Extraction for cycle %s, channel %s (%s), z slice %s, source "%s" complete (tile shape = %s)',
                        r['cycle_index'], r['channel_index'], r['channel_name'], z_slice, src, sub_tile.shape
                    )
                    assert sub_tile.ndim == 3, \
                        'Expecting sub_tile to have 3 dimensions but got shape {}'.format(sub_tile.shape)
                    slice_labels.append('{}_{}'.format(src, r['channel_name']))
                    extract_tile.append(sub_tile)

            # Stack the subtiles to give array with shape (z, channels, h, w) and then reshape to 5D
            # format like (cycles, z, channels, h, w)
            extract_tile = np.stack(extract_tile, axis=1)[np.newaxis]
            assert extract_tile.ndim == 5, \
                'Expecting extract tile to have 5 dimensions but got shape {}'.format(extract_tile.shape)

            extract_path = cytokit_io.get_extract_image_path(loc.region_index, loc.tile_x, loc.tile_y, name)
            extract_path = osp.join(self.data_dir, extract_path)
            logging.debug(
                'Saving tile with shape %s (dtype = %s) to "%s"',
                extract_tile.shape, extract_tile.dtype, extract_path
            )

            # Construct slice labels as repeats across z-dimension (there is only one time/cycle dimension)
            slice_label_tags = ij_utils.get_channel_label_tags(slice_labels, z=extract_tile.shape[1], t=1)
            cytokit_io.save_tile(
                extract_path, extract_tile, config=self.config,
                infer_labels=False, extratags=slice_label_tags
            )

        logging.info('Extraction complete (results saved to %s)', osp.dirname(extract_path) if extract_path else None)

    def montage(self, name, extract_name, region_indexes=None, crop=None):
        """Create a montage of extracted tiles

        Args:
            name: Name of montage to be created; This will be used to construct result path like
                EXP_DIR/output/montage/`name`
            extract_name: Name of extract to use to generate montage
            region_indexes: 1-based sequence of region indexes to process; can be specified as:
                - None: Region indexes will be inferred from experiment configuration
                - str or int: A single value will be interpreted as a single index
                - tuple: A 2-item or 3-item tuple forming the slice (start, stop[, step]); stop is inclusive
                - list: A list of integers will be used as is
            tile_indexes: 1-based sequence of tile indexes to process; has same semantics as `region_indexes`
            crop: Either none (default) or a 4-item list in the format (y_start, y_end, x_start, x_end) as
                bounding indices (0-based) that will be applied as a slice on the final montage (this is helpful
                for generating more reasonably sized montage subsets over large datasets)
        """
        logging.info('Creating montage "%s" from extraction "%s"', name, extract_name)
        region_indexes = cli.resolve_index_list_arg(region_indexes, zero_based=True)
        prep_fn = None
        if crop is not None:
            prep_fn = lambda tile: tile[:, :, :, crop[0]:crop[1], crop[2]:crop[3]]
        core.create_montage(self.data_dir, self.config, extract_name, name, region_indexes, prep_fn=prep_fn)


if __name__ == '__main__':
    fire.Fire(Operator)
