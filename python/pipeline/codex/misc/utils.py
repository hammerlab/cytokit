import pandas as pd
import re
import os


def read_expression_file(config, path, channel_prefix=None):
    """Read expression file containing marker expression levels as well as cell size and location

    These files are typically the result of the CODEXSegm process and are often named like
    `reg001_Expression_Compensated.txt` or `reg001_Expression_Uncompensated.txt`

    Args:
        config: Experiment configuration
        path: Path to expression file
        channel_prefix: String to prepend to resolved channel names (default is None)
    :return: DataFrame containing expression data
    """
    import pandas as pd

    # Load configuration and expression data resulting from CODEX segmentation
    df = pd.read_csv(path, sep='\t')
    df = df.rename(columns={'Filename:Filename': 'tile'})

    # Rename "Cyc" fields with corresponding channel names
    cyc_names = list(df.filter(regex='Cyc_').columns.values)
    assert len(cyc_names) == len(config.channel_names)
    ch_names = [channel_prefix + cn if channel_prefix else cn for cn in config.channel_names]
    df = df.rename(columns=dict(zip(cyc_names, ch_names)))

    # Add tile index as "Xt" and "Yt"
    def add_tile_index(df, c, i):
        idx = df.columns.tolist().index(c)
        df.insert(idx, c + 't', df['tile']\
            .apply(lambda v: int(v.split('_')[i].replace(c, ''))))
    add_tile_index(df, 'X', 1)
    add_tile_index(df, 'Y', 2)
    return df


def get_best_focal_planes(config, data_dir, ireg=0):
    """Get data frame containing information about best focal planes for each tile

    Note: This is a temporary implementation that should ultimately extract this
        information from dedicated files rather than inferring it from file names
    """
    best_focus_dir = os.path.join(data_dir, 'bestFocus')
    prefix = 'reg{:03d}'.format(ireg + 1)
    m = re.compile(prefix + '_X(\d+)_Y(\d+)_Z(\d+).tif')
    res = []
    for bf in os.listdir(best_focus_dir):
        if not bf.startswith(prefix + '_X'):
            continue
        groups = m.match(bf).groups()
        if len(groups) != 3:
            raise ValueError('X/Y/Z coords could not be extracted from file "{}"'.format(bf))
        tx, ty, tz = groups
        res.append((int(tx), int(ty), int(tz)))

    if len(res) != config.region_width * config.region_height:
        raise ValueError('Expecting best z planes for {} tiles but only found {}' \
                         .format(config.region_width * config.region_height, len(res)))
    return pd.DataFrame(res, columns=['x', 'y', 'z'])