import json
import sys
import os
import os.path as osp
import cytokit
import logging
from cytokit import io as cytokit_io
from cytokit import config as cytokit_config
import pandas as pd

LOG_FORMAT = '%(asctime)s:%(levelname)s:%(process)d:%(name)s: %(message)s'

CH_SRC_RAW = 'raw'
CH_SRC_PROC = 'proc'
CH_SRC_CYTO = 'cyto'
CH_SOURCES = [CH_SRC_RAW, CH_SRC_PROC, CH_SRC_CYTO]


def get_config(config_path):
    """Load experiment configuration

    Args:
        config_path: Either a path to a configuration file or a directory containing a
            configuration file with the default name (controlled by CYTOKIT_CONFIG_DEFAULT_FILENAME)
    Returns:
        Configuration object
    """
    # Load experiment configuration and "register" the environment meaning that any variables not
    # explicitly defined by env variables should set based on what is present in the configuration
    # (it is crucial that this happen first)
    config = cytokit_config.load(config_path)
    config.register_environment()
    return config


def record_execution(output_dir):
    """Record execution arguments and environment as json file"""
    path = osp.join(output_dir, cytokit_io.get_processor_exec_path(date=pd.to_datetime('now').strftime('%Y%m%d%H%M')))
    if not osp.exists(osp.dirname(path)):
        os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, 'w') as fd:
        json.dump({'args': sys.argv, 'env': cytokit.get_env_vars()}, fd)
    return path


def resolve_int_list_arg(arg):
    """Resolve a CLI argument as a list of integers"""
    if arg is None:
        return None
    if isinstance(arg, int):
        return [arg]
    if isinstance(arg, str):
        return [int(arg)]
    if isinstance(arg, tuple):
        if len(arg) not in [2, 3]:
            raise ValueError(
                'When specifying argument as a tuple it must contain 2 or 3 items indicating '
                'a range as start, stop[, step] w/ inclusive stop (given = {})'.format(arg))
        # Interpret as range inclusive of end point
        vals = [int(v) for v in arg]
        return list(range(vals[0], vals[1] + 1, 1 if len(vals) < 3 else vals[2]))
    if isinstance(arg, list):
        return [int(v) for v in arg]
    raise ValueError('Argument of type {} could not be interpreted as a list (given = {})'.format(type(arg), arg))


def resolve_index_list_arg(arg, zero_based=False):
    indexes = resolve_int_list_arg(arg)
    if indexes is None:
        return None
    if any([i < 1 for i in indexes]):
        raise ValueError(
            'Index argument is supposed to be 1-based but resolved to list with '
            'index < 1 (arg = {}, resolved indexes = {})'.format(arg, indexes)
        )
    return [i - 1 for i in indexes] if zero_based else indexes


def get_logging_init_fn(py_log_level, tf_py_log_level, tf_cpp_log_level):
    from cytokit.utils import tf_utils

    def init():
        logging.basicConfig(level=tf_utils.log_level_code(py_log_level), format=LOG_FORMAT)
        tf_utils.init_tf_logging(tf_cpp_log_level, tf_py_log_level)
    return init


class CLI(object):

    def __init__(self, py_log_level=logging.INFO,
                 tf_py_log_level=logging.ERROR,
                 tf_cpp_log_level=logging.ERROR):
        """CLI Initialization

        Args:
            py_log_level: Logging level for Cytokit and dependent modules (except TensorFlow); can be
                specified as string or integer compatible with python logging levels (e.g. 'info', 'debug',
                'warn', 'error', 'fatal' or corresponding integers); default is 'info'
            tf_py_log_level: TensorFlow python logging level; same semantics as `py_log_level`; default is 'error'
            tf_cpp_log_level: TensorFlow C++ logging level; same semantics as `py_log_level`; default is 'error'
        """
        # Get and run logging initializer
        self._logging_init_fn = get_logging_init_fn(py_log_level, tf_py_log_level, tf_cpp_log_level)
        self._logging_init_fn()


class DataCLI(CLI):

    def __init__(self,
                 data_dir, config_path=None,
                 py_log_level=logging.INFO,
                 tf_py_log_level=logging.ERROR,
                 tf_cpp_log_level=logging.ERROR):
        """CLI Initialization

        Args:
            data_dir: Path to experiment output root directory
            config_path: Either a directory containing a configuration by the default name (controlled via
             env variable "CYTOKIT_CONFIG_DEFAULT_FILENAME"), e.g. "experiment.yaml", or a path
                to a single file; If not provided this will default to `data_dir`
            py_log_level: Logging level for Cytokit and dependent modules (except TensorFlow); can be
                specified as string or integer compatible with python logging levels (e.g. 'info', 'debug',
                'warn', 'error', 'fatal' or corresponding integers); default is 'info'
            tf_py_log_level: TensorFlow python logging level; same semantics as `py_log_level`; default is 'error'
            tf_cpp_log_level: TensorFlow C++ logging level; same semantics as `py_log_level`; default is 'error'
        """
        super(DataCLI, self).__init__(py_log_level, tf_py_log_level, tf_cpp_log_level)
        self.config = get_config(config_path or data_dir)
        self.data_dir = data_dir

    def run_all(self, **kwargs):
        for config in self._get_function_configs():
            if len(config) != 1:
                raise ValueError('Processing function configuration "{}" is not valid (should only have 1 key)'.format(config))
            op = list(config.keys())[0]
            if not hasattr(self, op):
                raise ValueError('CLI function name "{}" is invalid'.format(op))
            logging.debug('Running operation "%s" with arguments "%s"', op, config[op])
            fn = getattr(self, op)

            # Extract kwargs relevant for this operation
            params = {k: v for k, v in kwargs.items() if k in fn.__code__.co_varnames}

            # Merge kwargs with configured parameters and pass all to function call
            fn(**{**config[op], **params})
