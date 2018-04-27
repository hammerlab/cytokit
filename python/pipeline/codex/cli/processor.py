#!/usr/bin/python
"""CODEX preprocessing pipeline CLI application"""
import fire
from codex.exec import pipeline
from codex.utils import tf_utils
import logging

LOG_FORMAT = '%(asctime)s:%(levelname)s:%(process)d:%(name)s: %(message)s'


class CodexProcessor(object):

    def localhost(
            self, data_dir, output_dir, 
            region_indexes=None, tile_indexes=None, config_dir=None,
            n_workers=None, gpus=None, memory_limit=32e9,
            tile_prefetch_capacity=2, run_best_focus=True, n_iter_decon=25,
            codex_py_log_level=logging.INFO, 
            tf_py_log_level=logging.ERROR,
            tf_cpp_log_level=logging.ERROR):
        """Run CODEX pre-processing pipeline

        This application will conduct the following operations on raw tif stacks:
            - Drift compensation
            - Deconvolution
            - Selection of best focal planes within z-stacks
            - Cropping of tile overlap

        Nothing beyond an input data directory and an output directory are required (see arguments
        below), but GPU information should be provided via the `gpus` argument to ensure that
        all present devices are utilized.  Otherwise, all arguments have logical defaults that
        should only need to be changed in special scenarios.

        Args:
            data_dir: Path to directoring containing raw acquisition data files
            output_dir: Directory to save results in; will be created if it does not exist
            region_indexes: 1-based sequence of region indexes to process; can be specified as:
                - None: Region indexes will be inferred from experiment configuration
                - str or int: A single value will be interpreted as a single index 
                - tuple: A two-item tuple will be interpreted as a right-open range (e.g. '(1,4)' --> [1, 2, 3]) 
                - list: A list of integers will be used as is
            tile_indexes: 1-based sequence of tile indexes to process; has same semantics as `region_indexes`
            config_dir: Directory containing experiment configuration files; defaults to `data_dir` if not given
            n_workers: Number of tiles to process in parallel; should generally match number of gpus and if
                the `gpus` argument is given, then the length of that list will be used as a default (otherwise
                default is 1)
            gpus: 0-based list of gpu indexes to use for processing; has same semantics as other integer
                list arguments like `region_indexes` and `tile_indexes` (i.e. can be a scalar, list, or 2-tuple)
            memory_limit: Maximum amount of memory to allow per-worker; defaults to 32G
            tile_prefetch_capacity: Number of input tiles to buffer into memory for processing; default is 2
                which is nearly always good as this means one tile will undergo processing while a second
                is buffered into memory asynchronously
            run_best_focus: Flag indicating that best focal plan selection operations should be executed
            n_iter_decon: Number of deconvolution iterations
            codex_py_log_level: Logging level for CODEX and dependent modules (except TensorFlow); can be
                specified as string or integer compatible with python logging levels (e.g. 'info', 'debug',
                'warn', 'error', 'fatal' or corresponding integers)
            tf_py_log_level: TensorFlow python logging level; same semantics as `codex_py_log_level`
            tf_cpp_log_level: TensorFlow C++ logging level; same semantics as `codex_py_log_level`
        """
        # Resolve arguments with multiple supported forms
        region_indexes = resolve_int_list_arg(region_indexes)
        tile_indexes = resolve_int_list_arg(tile_indexes)
        gpus = resolve_int_list_arg(gpus)

        # Initialize logging (use a callable function for passing to spawned processes in pipeline)
        def logging_init_fn():
            logging.basicConfig(level=tf_utils.log_level_code(codex_py_log_level), format=LOG_FORMAT)
            tf_utils.init_tf_logging(tf_cpp_log_level, tf_py_log_level)
        logging_init_fn()

        # Set dynamic defaults
        if config_dir is None:
            config_dir = data_dir 
        if n_workers is None:
            # Default to 1 worker given no knowledge of available gpus 
            n_workers = len(gpus) if gpus is not None else 1

        # Execute pipeline on localhost
        conf = pipeline.PipelineConfig(
            region_indexes, tile_indexes, config_dir, data_dir, output_dir,
            n_workers, gpus, memory_limit,
            tile_prefetch_capacity=tile_prefetch_capacity,
            run_best_focus=run_best_focus,
            n_iter_decon=n_iter_decon
        )
        pipeline.run(conf, logging_init_fn=logging_init_fn)

    def gke(self):
        # Ultimately, a GKE impementation should use the same "localhost" code above on cluster containers
        pass


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


if __name__ == '__main__':
    fire.Fire(CodexProcessor)
