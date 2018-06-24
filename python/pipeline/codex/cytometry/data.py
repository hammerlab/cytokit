from codex import io as codex_io
import warnings
import os.path as osp
import pandas as pd


def aggregate(config, output_dir):
    """Aggregate cytometry data associated with an experiment into a single dataframe

    Args:
        config: Experiment configuration
        output_dir: Output directory for experiment
    """
    df = []
    for idx in config.get_tile_indices():
        path = codex_io.get_cytometry_file_path('.csv', idx.region_index, idx.tile_x, idx.tile_y)
        path = osp.join(output_dir, path)
        if not osp.exists(path):
            warnings.warn(
                'Expected cytometry data file at "{}" does not exist.  '
                'It will be ignored but this is worth investigating'
                .format(path)
            )
            continue
        df.append(pd.read_csv(path))
    return pd.concat(df)
