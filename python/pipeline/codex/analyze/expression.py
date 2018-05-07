"""Library functions for interacting with CODEX expression files resulting from segmentation"""


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
