import json
import os
import os.path as osp
from cytokit import io as cytokit_io
import pandas as pd


def record_processor_data(data, output_dir):
    """Save processor data as json file"""
    path = osp.join(output_dir, cytokit_io.get_processor_data_path())
    if not osp.exists(osp.dirname(path)):
        os.makedirs(osp.dirname(path), exist_ok=True)

    # If a data file already exists, merge the two preferring newer entries
    if osp.exists(path):
        past_data = read_processor_data(path)
        past_data.update(data)
        data = past_data

    # Ensure that values in data dict are lists of dicts (not DataFrames already
    # since they will not serialize correctly)
    data = {
        k: v.to_dict(orient='records') if isinstance(v, pd.DataFrame) else v
        for k, v in data.items()
    }

    # Use pandas for serialization as it has built-in numpy type converters
    pd.Series(data).to_json(path, orient='index')
    return path


def read_processor_data(path):
    """Load processor data as dict of data frames"""
    with open(path, 'r') as fd:
        return {k: pd.DataFrame(v) for k, v in json.load(fd).items()}
