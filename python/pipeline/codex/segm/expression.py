"""Library functions for interacting with CODEX expression files resulting from segmentation"""


def read_expression_file(config, path):
    """Read expression file containing marker expression levels as well as cell size and location

    These files are typically the result of the CODEXSegm process and are often named like
    `reg001_Expression_Compensated.txt` or `reg001_Expression_Uncompensated.txt`

    :param config: Experiment configuration
    :param path: Path to expression file
    :return: DataFrame containing expression data
    """
    import pandas as pd

    # Load configuration and expression data resulting from CODEX segmentation
    df = pd.read_csv(path, sep='\t')
    df = df.rename(columns={'Filename:Filename': 'tile'})
    cyc_names = list(df.filter(regex='Cyc_').columns.values)
    assert len(cyc_names) == len(config.channel_names)
    df = df.rename(columns=dict(zip(cyc_names, config.channel_names)))
    return df
