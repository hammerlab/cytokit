import json
import sys
import os
import os.path as osp
import codex
from codex import io as codex_io
import pandas as pd

LOG_FORMAT = '%(asctime)s:%(levelname)s:%(process)d:%(name)s: %(message)s'


def record_execution(output_dir):
    """Record execution arguments and environment as json file"""

    path = osp.join(output_dir, codex_io.get_processor_exec_path(date=pd.to_datetime('now').strftime('%Y%m%d%H%M')))
    if not osp.exists(osp.dirname(path)):
        os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, 'w') as fd:
        json.dump({'args': sys.argv, 'env': codex.get_env_vars()}, fd)
    return path


def record_processor_data(data, output_dir):
    """Save processor data as json file"""
    path = osp.join(output_dir, codex_io.get_processor_data_path())
    if not osp.exists(osp.dirname(path)):
        os.makedirs(osp.dirname(path), exist_ok=True)

    # If a data file already exists, merge the two preferring newer entries
    if osp.exists(path):
        past_data = read_processor_data(path)
        past_data.update(data)
        data = past_data

    # Use pandas for serialization as it has built-in numpy type converters
    pd.Series(data).to_json(path, orient='index')
    return path


def read_processor_data(path):
    """Load processor data as dict of data frames"""
    with open(path, 'r') as fd:
        return {k: pd.DataFrame(v) for k, v in json.load(fd).items()}
