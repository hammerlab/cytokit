import os
import os.path as osp
from codex import cli
from codex import io as codex_io
from codex.ops import best_focus
from codex.ops.op import CodexOp


def get_best_focus_data(output_dir):
    """Get precomputed best focus plane information

    Note that this will return a data frame with references to 0-based region/tile indexes
    """

    # Extract best focal plane selections from precomputed processor data
    best_focus_op = CodexOp.get_op_for_class(best_focus.CodexFocalPlaneSelector)
    processor_data_filepath = osp.join(output_dir, codex_io.get_processor_data_path())
    focus_data = cli.read_processor_data(processor_data_filepath)
    if best_focus_op not in focus_data:
        raise ValueError(
            'No focal plane statistics found in statistics file "{}".  '
            'Are you sure the processor.py app was run with `run_best_focus`=True?'
            .format(processor_data_filepath)
        )
    return focus_data[best_focus_op][['region_index', 'tile_index', 'tile_x', 'tile_y', 'best_z']]\
        .dropna().drop_duplicates().astype(int)


def get_best_focus_coord_map(output_dir):
    return get_best_focus_data(output_dir).set_index(['region_index', 'tile_x', 'tile_y'])['best_z'].to_dict()
