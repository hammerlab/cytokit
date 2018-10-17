import os
import os.path as osp
from cytokit import exec
from cytokit import io as cytokit_io
from cytokit.ops import best_focus
from cytokit.ops.op import CytokitOp

CYTOMETRY_STATS_AGG_MODES = ['best_z_plane', 'all']


def get_best_focus_data(output_dir):
    """Get precomputed best focus plane information

    Note that this will return a data frame with references to 0-based region/tile indexes
    """
    processor_data_filepath = osp.join(output_dir, cytokit_io.get_processor_data_path())
    processor_data = exec.read_processor_data(processor_data_filepath)
    best_focus_op = CytokitOp.get_op_for_class(best_focus.CytokitFocalPlaneSelector)
    if best_focus_op not in processor_data:
        raise ValueError(
            'No focal plane statistics found in statistics file "{}".  '
            'Are you sure the processor.py app was run with `run_best_focus`=True?'
            .format(processor_data_filepath)
        )
    return processor_data[best_focus_op][['region_index', 'tile_index', 'tile_x', 'tile_y', 'best_z']] \
        .dropna().drop_duplicates().astype(int)


def get_best_focus_coord_map(output_dir):
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
