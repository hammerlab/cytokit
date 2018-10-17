import os
import cytokit
import os.path as osp
import numpy as np
from cytokit import io as cytokit_io
import logging
logger = logging.getLogger(__name__)


#########################
# Aggregation Functions #
#########################


def aggregate_cytometry_statistics(output_dir, config, mode='all', export_csv=True, export_fcs=True, variant=None):
    from cytokit.function import data as function_data

    # Aggregate all cytometry csv data (across tiles)
    res = function_data.get_cytometry_data(output_dir, config, mode=mode)

    # Get file extension, possibly with user-defined "variant" name to be included in all
    # resulting file names
    def ext(file_ext):
        return file_ext if variant is None else '{}.{}'.format(variant, file_ext)

    # Export result as csv
    csv_path, fcs_path = None, None
    if export_csv:
        csv_path = osp.join(output_dir, cytokit_io.get_cytometry_agg_path(ext('csv')))
        cytokit_io.save_csv(csv_path, res, index=False)
        logger.info('Saved cytometry aggregation results to csv at "{}"'.format(csv_path))
    if export_fcs:
        import re
        import fcswrite
        nonalnum = '[^0-9a-zA-Z]+'

        # For FCS exports, save only integer and floating point values and replace any non-alphanumeric
        # column name characters with underscores
        res_fcs = res.select_dtypes(['int', 'float']).rename(columns=lambda c: re.sub(nonalnum, '_', c))
        fcs_path = osp.join(output_dir, cytokit_io.get_cytometry_agg_path(ext('fcs')))
        if not osp.exists(osp.dirname(fcs_path)):
            os.makedirs(osp.dirname(fcs_path), exist_ok=True)
        fcswrite.write_fcs(filename=fcs_path, chn_names=res_fcs.columns.tolist(), data=res_fcs.values)
        logger.info('Saved cytometry aggregation results to fcs at "{}"'.format(fcs_path))
    return csv_path, fcs_path


######################
# Notebook Functions #
######################


def _get_nb_path(nb_name):
    return osp.join(cytokit.nb_dir, 'analysis', nb_name)


def run_nb(nb_name, nb_output_path, nb_params):
    import papermill as pm
    nb_input_path = _get_nb_path(nb_name)
    pm.execute_notebook(nb_input_path, nb_output_path, parameters=nb_params)


#####################
# Montage Functions #
#####################

def create_montage(output_dir, config, extract, name, region_indexes, prep_fn=None):
    from cytokit.utils import ij_utils

    # Loop through regions and generate a montage for each, skipping any (with a warning) that
    # do not have focal plane selection information
    if region_indexes is None:
        region_indexes = config.region_indexes

    path = None
    for ireg in region_indexes:
        logger.info('Generating montage for region %d of %d', ireg + 1, len(region_indexes))
        tiles = []
        labels = None
        for itile in range(config.n_tiles_per_region):
            tx, ty = config.get_tile_coordinates(itile)
            path = cytokit_io.get_extract_image_path(ireg, tx, ty, extract)
            tile, meta = cytokit_io.read_tile(osp.join(output_dir, path), return_metadata=True)
            if labels is None:
                labels = meta['labels']
            tiles.append(tile)
        reg_img_montage = montage(tiles, config)
        if prep_fn is not None:
            reg_img_montage = prep_fn(reg_img_montage)
        path = osp.join(output_dir, cytokit_io.get_montage_image_path(ireg, name))
        logger.info('Saving montage to file "%s"', path)
        tags = [] if labels is None else ij_utils.get_slice_label_tags(labels)
        cytokit_io.save_tile(path, reg_img_montage, config=config, infer_labels=False, extratags=tags)
    logger.info('Montage generation complete; results saved to "%s"', None if path is None else osp.dirname(path))


def montage(tiles, config):
    """Montage a list of tile images

    Args:
        tiles: A list of images having at least 2 dimensions (rows, cols) though any number of leading dimensions
            is also supported (for cycles, channels, z-planes, etc); must have length equal to region width * region height
        config: Experiment configuration
    Returns:
        An array of same data type as tiles with all n-2 dimensions the same as individual tiles, but with the last
        two dimensions expanded as a "montage" of all 2D images contained within the tiles
    """
    rw, rh = config.region_width, config.region_height

    # Determine shape/type of individual tiles by checking the first one
    # (and assume all others will be equal)
    dtype_proto, shape_proto = tiles[0].dtype, tiles[0].shape
    if len(shape_proto) < 2:
        raise ValueError('Tiles must all be at least 2D (shape given = {})'.format(shape_proto))
    shape_rc, shape_ex = shape_proto[-2:], shape_proto[:-2]
    th, tw = shape_rc

    # Preserve leading dimensions and assume 2D image for each will be the
    # same size multiplied by the number of tiles in row or column axis directions
    img_montage = np.zeros(np.concatenate((shape_ex, shape_rc * np.array([rh, rw]))).astype(np.int), dtype=dtype_proto)

    for itile, tile in enumerate(tiles):
        if tile.shape != shape_proto:
            raise ValueError(
                'All tiles must have the same shape (found {}, expected {})'.format(tile.shape, shape_proto))
        tx, ty = config.get_tile_coordinates(itile)
        idx = [slice(None) for _ in shape_ex] + [slice(ty * th, (ty + 1) * th), slice(tx * tw, (tx + 1) * tw)]
        img_montage[tuple(idx)] = tile
    return img_montage
