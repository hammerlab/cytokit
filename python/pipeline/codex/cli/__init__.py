import json
import sys
import os
import os.path as osp
import codex
from codex import io as codex_io
from codex import config as codex_config
import pandas as pd

LOG_FORMAT = '%(asctime)s:%(levelname)s:%(process)d:%(name)s: %(message)s'


def get_config(config_path):
    """Load experiment configuration

    Args:
        config_path: Either a path to a configuration file or a directory containing a
            configuration file with the default name (controlled by CODEX_CONFIG_DEFAULT_FILENAME)
    Returns:
        Configuration object
    """
    # Load experiment configuration and "register" the environment meaning that any variables not
    # explicitly defined by env variables should set based on what is present in the configuration
    # (it is crucial that this happen first)
    config = codex_config.load(config_path)
    config.register_environment()
    return config


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


def resolve_int_list_arg(arg):
    """Resolve a CLI argument as a list of integers"""
    if arg is None:
        return None
    if isinstance(arg, int):
        return [arg]
    if isinstance(arg, str):
        return [int(arg)]
    if isinstance(arg, tuple):
        # Interpret as range (ignore any other items in tuple beyond second)
        return list(range(arg[0], arg[1]))
    return arg
