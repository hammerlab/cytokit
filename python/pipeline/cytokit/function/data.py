import os
import os.path as osp
import numpy as np
import pandas as pd
from skimage import measure
from cytokit import exec
from cytokit import io as cytokit_io
from cytokit.ops import best_focus
from cytokit.ops.op import CytokitOp
from cytokit.image import proc as cvproc
from cytokit.image import ops as cvops
from cytokit.cli import CH_SRC_CYTO

CYTOMETRY_STATS_AGG_MODES = ['best_z_plane', 'all']


def get_processor_data(output_dir, return_path=False):
    path = osp.join(output_dir, cytokit_io.get_processor_data_path())
    proc_data = exec.read_processor_data(path)
    if return_path:
        return proc_data, path
    else:
        return proc_data


def get_best_focus_data(output_dir):
    """Get precomputed best focus plane information

    Note that this will return a data frame with references to 0-based region/tile indexes
    """
    processor_data, path = get_processor_data(output_dir, return_path=True)
    best_focus_op = CytokitOp.get_op_for_class(best_focus.CytokitFocalPlaneSelector)
    if best_focus_op not in processor_data:
        raise ValueError(
            'No focal plane statistics found in statistics file "{}".  '
            'Are you sure the processor.py app was run with `run_best_focus`=True?'
            .format(path)
        )
    return processor_data[best_focus_op][['region_index', 'tile_index', 'tile_x', 'tile_y', 'best_z']] \
        .dropna().drop_duplicates().astype(int)


def get_best_focus_coord_map(output_dir):
    """Get map of best z planes as (region_index, tile_x, tile_y) -> best_z (all zero-based)"""
    return get_best_focus_data(output_dir).set_index(['region_index', 'tile_x', 'tile_y'])['best_z'].to_dict()


def get_cytometry_data(output_dir, config, mode='all'):
    import pandas as pd
    from cytokit.cytometry import data as cytometry_data

    if mode not in CYTOMETRY_STATS_AGG_MODES:
        raise ValueError(
            'Cytometry stats aggregation mode must be one of {} not "{}"'
            .format(CYTOMETRY_STATS_AGG_MODES, mode)
        )

    # Aggregate all cytometry csv data (across tiles)
    cyto_data = cytometry_data.aggregate(config, output_dir)

    # If configured, select only data associated with "best" z planes
    if mode == 'best_z_plane':
        # Extract best focal plane selections from precomputed processor data
        focus_data = get_best_focus_data(output_dir)

        # Merge to cytometry data on region / tile index (this will add a single column, "best_z")
        merge_data = pd.merge(
            cyto_data, focus_data[['region_index', 'tile_index', 'best_z']],
            on=['region_index', 'tile_index'],
            how='left'
        )
        if merge_data['best_z'].isnull().any():
            # Create list of regions / tiles with null z planes
            ex = merge_data[merge_data['best_z'].isnull()][['region_index', 'tile_x', 'tile_y']]
            raise ValueError(
                'Failed to find best z plane settings for at least one tile;\n'
                'The following (region, tile_x, tile_y) combinations have no known best z-planes: {}'
                .format(ex.values)
            )
        # Filter result to where z plane equals best z and drop best_z field
        cyto_data = merge_data[merge_data['best_z'] == merge_data['z']].drop('best_z', axis=1)

    return cyto_data


###############################
# Single Cell Image Functions #
###############################

def get_extract_image_meta(output_dir, extract):
    path = osp.join(output_dir, cytokit_io.get_extract_image_path(0, 0, 0, extract))
    _, meta = cytokit_io.read_tile(path, return_metadata=True)
    return meta


def get_single_cell_image_data(output_dir, df, extract, ranges=None, colors=None, image_size=None, **kwargs):
    """Add single cell images and properties (based on an extract) to a cytometry data frame

    Args:
        df: Cytometry data frame
        extract: Name of extract from which to extract single cell data; must contain the channel `cyto_cell_boundary`
        ranges: Dictionary mapping extract channel names to min/max value ranges (to control contrast); Example:
            ```
            ranges = {
                'proc_DAPI': [0, 255],
                'proc_CD3': [0, 150],
                'cyto_nucleus_boundary': [0, 1],
                'cyto_cell_boundary': [0, 1]
            }
            ```
        colors: Dictionary mapping extract channel names to color names or RGB colors as floats (
            see cytokit.image.color for list of all named colors); Example:
            ```
            colors = {
                'proc_DAPI': 'blue',
                'proc_CD3': 'green',
                'proc_DAPI2': 'none', # channel is hidden ([0, 0, 0] also works for this)
                'cyto_nucleus_boundary': [1., 0., 0.], # red
                'cyto_cell_boundary': [1., 1., 0.] # orange
            }
            ```
        image_size: Size of individual cell images; default is None which returns images of varying sizes necessary
            to fit bounding box of each cell object.  If given (e.g. image_size=(64,64)), images are resized
            to the target 2D shape WITHOUT resampling -- instead the image is cropped around the center or padded
            with zeros to ensure that the resulting images have the same scale (this ensures that cell images
            are comparable on the same image scale).
    """

    def add_cell_images(g):
        reg, tx, ty = g.iloc[0][['region_index', 'tile_x', 'tile_y']]

        # Extract the relevant 2D image to be used for both cell object isolation and cell image display
        path = osp.join(output_dir, cytokit_io.get_extract_image_path(reg, tx, ty, extract))
        img, meta = cytokit_io.read_tile(path, return_metadata=True)
        icyc, iz = kwargs.get('cycle', 0), kwargs.get('z', 0)
        img = img[icyc, iz]
        channels = list(meta['structured_labels'][icyc, iz])
        processor = cvproc.get_image_processor(channels, ranges=ranges, colors=colors)

        # Get the cell image data frame containing the original cell id, cell image based on processed
        # raw image, and associated cell image properties
        cell_data = pd.DataFrame(extract_single_cell_image_data(
            g, img, processor.run(img), channels, image_size=image_size
        ))

        # Verify that the only shared field between the two datasets is 'id'
        assert g.columns.isin(cell_data.columns).sum() == 1, \
            'Cell data frame should only have one overlapping field with cytometry data frame;' \
            '\nCell fields = {}\nCytometry fields = {}'.format(cell_data.columns, g.columns)

        # Left join cytometry data on single cell data
        return pd.merge(g, cell_data, how='left', on='id')

    return df.groupby(['region_index', 'tile_x', 'tile_y'], group_keys=False)\
        .apply(add_cell_images).reset_index(drop=True)


def extract_single_cell_image_data(df, raw_tile, display_tile, channels, image_size=None,
                         object_type='cell_boundary', apply_mask=True):
    if df is None:
        return None

    cell_boundary_channel = CH_SRC_CYTO + '_' + object_type
    if cell_boundary_channel not in channels:
        raise ValueError(
            'Cannot generate single cell images because extract does not contain cell boundary channel;'
            'channels present = {}, channel expected = {}'.format(channels, cell_boundary_channel)
        )

    # Fetch raw tile image with original channels, and extract cell boundaries
    cell_tile = raw_tile[channels.index(cell_boundary_channel)].copy()

    # Make sure that data frame with cell data does not contain ids that don't exist
    # in the label image, noting that the reverse is not necessarily true since the
    # data frame may have been filtered
    id_diff = np.setdiff1d(df['id'].unique(), np.unique(cell_tile))
    if len(id_diff) > 0:
        raise AssertionError(
            'Cytometry data contains cell ids not found in tile; '
            'Ids in data frame: {}\nIds in label image: {}\nMissing: {}'
            .format(df['id'].unique(), np.unique(cell_tile), id_diff)
        )

    # Eliminate cell objects not in sample
    cell_tile[~np.isin(cell_tile, df['id'].values)] = 0

    # Extract regions in RGB image (display tile) corresponding to cell labelings
    cell_data = extract_single_cell_images(
        cell_tile, display_tile,
        is_boundary=True,
        patch_shape=image_size,
        apply_mask=apply_mask,
        fill_value=0
    )

    # Return list of dictionaries where each represents one cell (with at least an id and image)
    return cell_data


def extract_single_cell_images(cell_image, target_image, patch_shape=None, is_boundary=True,
                             apply_mask=True, fill_value=0):
    """Extract single cell images from a target image

    Args:
        cell_image: 2D label image containing cell objects (each with different id)
        target_image: Image from which to extract patches around cells; must be at least 2D
            in format HW[D1, D2, ...]
        patch_shape: Target shape of individual cell images; If None (default) no cropping/padding
            will occur but if set, this value should be a 2 item sequence [rows, cols] and cell image patches
            will be conformed to this shape by either cropping or padding out from the center
        is_boundary: Whether or not cell image is of boundary or masks (default True)
        apply_mask: Whether or not to set pixels outside of cell binary image to `fill_value` (default True)
        fill_value: Pixel values for parts of cell image outside cell object (default 0)
    """
    if target_image.shape[:2] != cell_image.shape[:2]:
        raise ValueError(
            'Cell label image (shape = {}) must have same HW dimensions as target image (shape = {})'
            .format(cell_image.shape, target_image.shape)
        )

    if patch_shape is not None and len(patch_shape) != 2:
        raise ValueError('Target patch shape should be a 2 item sequence (given = {})'.format(patch_shape))

    cells = []
    props = measure.regionprops(cell_image)
    for p in props:

        # Extract bounding box offsets for extraction
        min_row, min_col, max_row, max_col = p.bbox

        # Extract patch from target image (make sure to copy for subsequent mutations)
        patch = target_image[min_row:max_row, min_col:max_col].copy()

        # Remove off-target pixels, if necessary
        if apply_mask:
            # Set mask containing which pixels in patch to keep
            if is_boundary:
                mask = p.convex_image
            else:
                mask = p.filled_image

            # Set value outside of mask to provided fill value
            patch[~mask] = fill_value

        # Resize if necessary (without transforming original image content)
        if patch_shape is not None:
            patch = cvops.resize_image_with_crop_or_pad(
                patch, tuple(patch_shape) + patch.shape[2:],
                constant_values=fill_value)

        cells.append(dict(id=p.label, properties=p, image=patch))
    return cells