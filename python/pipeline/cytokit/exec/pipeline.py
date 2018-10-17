"""Cytokit preprocessing pipeline core logic

This is not intended to be run directly but rather used by mutliple external
interfaces to implement the core process that comprises Cytokit processing.
"""
import os, logging, itertools, queue, sys
import numpy as np
import pandas as pd
from os import path as osp
from threading import Thread
from timeit import default_timer as timer
from cytokit import io as cytokit_io
from cytokit import config as cytokit_config
from cytokit import exec
from cytokit.function import data as function_data
from cytokit.ops import op
from cytokit.ops import tile_generator
from cytokit.ops import cytometry
from cytokit.ops import tile_crop
from cytokit.ops import drift_compensation
from cytokit.ops import best_focus
from cytokit.ops import deconvolution
from cytokit.ops import tile_summary
from cytokit.ops import illumination_correction
from cytokit.ops import spectral_unmixing
from dask.distributed import Client, LocalCluster
logger = logging.getLogger(__name__)

# Set 1 hour time limit on tile loading/reading operations
TIMEOUT = 1 * 60 * 60


class OpFlags(object):

    def __init__(self,
            run_best_focus=True, run_drift_comp=True, run_summary=True,
            run_tile_generator=True, run_crop=True, run_deconvolution=True,
            run_cytometry=True, run_illumination_correction=True, run_spectral_unmixing=True):
        self.run_tile_generator = run_tile_generator
        self.run_crop = run_crop
        self.run_deconvolution = run_deconvolution
        self.run_drift_comp = run_drift_comp
        self.run_best_focus = run_best_focus
        self.run_summary = run_summary
        self.run_cytometry = run_cytometry
        self.run_illumination_correction = run_illumination_correction
        self.run_spectral_unmixing = run_spectral_unmixing

    def postprocessing_enabled(self):
        return self.run_illumination_correction or \
            self.run_spectral_unmixing

    def preprocessing_enabled(self):
        return self.run_tile_generator or \
            self.run_crop or \
            self.run_deconvolution or \
            self.run_drift_comp or \
            self.run_best_focus or \
            self.run_summary or \
            self.run_cytometry


class TaskConfig(object):

    def __init__(self, pipeline_config, region_indexes, tile_indexes, gpu, tile_prefetch_capacity=2):
        self.region_indexes = region_indexes
        self.tile_indexes = tile_indexes
        self.data_dir = pipeline_config.data_dir
        self.output_dir = pipeline_config.output_dir
        self.gpu = gpu
        self.op_flags = pipeline_config.op_flags
        self.tile_prefetch_capacity = tile_prefetch_capacity
        self.exp_config = pipeline_config.exp_config

        if len(self.region_indexes) != len(self.tile_indexes):
            raise ValueError(
                'Region and tile index lists must have same length (region indexes = {}, tile indexes = {})'
                .format(self.region_indexes, self.tile_indexes)
            )

    @property
    def n_tiles(self):
        return len(self.tile_indexes)

    def __str__(self):
        return str({k: v for k, v in self.__dict__.items() if k != 'exp_config'})

    __repr__ = __str__


class PipelineConfig(object):

    def __init__(self, exp_config, region_indexes, tile_indexes, data_dir, output_dir, n_workers,
                 gpus, memory_limit, op_flags, **task_kwargs):
        self.exp_config = exp_config
        self.region_idx = region_indexes
        self.tile_idx = tile_indexes
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.n_workers = n_workers
        self.gpus = gpus
        self.memory_limit = memory_limit
        self.op_flags = op_flags
        self.task_kwargs = task_kwargs

        # Default region and tile index list to that in experiment configuration if not provided explicitly
        if self.region_idx is None:
            # Convert back to 1-based index to conform to 1-based-into-configs convention
            self.region_idx = [i + 1 for i in self.exp_config.region_indexes]
        if self.tile_idx is None:
            self.tile_idx = list(range(1, self.exp_config.n_tiles_per_region + 1))

        # Validate that only 1-based indexes are provided
        if any([i <= 0 for i in self.region_idx]):
            raise ValueError('Region indexes must be specified as 1-based index (indexes given = {})'.format(self.region_idx))
        if any([i <= 0 for i in self.tile_idx]):
            raise ValueError('Tile indexes must be specified as 1-based index (indexes given = {})'.format(self.tile_idx))

    def __str__(self):
        return str({
            k: v for k, v in self.__dict__.items()
            if k not in ['exp_config', 'tile_idx', 'region_idx']
        })

    __repr__ = __str__

    def get_task_config(self, region_indexes, tile_indexes, gpu):
        return TaskConfig(
            pipeline_config=self,
            region_indexes=region_indexes,
            tile_indexes=tile_indexes,
            gpu=gpu,
            **self.task_kwargs
        )

    @property
    def region_indexes(self):
        """Get 0-based region index array"""
        return np.array(self.region_idx) - 1

    @property
    def tile_indexes(self):
        """Get 0-based tile index array"""
        return np.array(self.tile_idx) - 1

    @property
    def region_tiles(self):
        """Get 0-based pairs of region and tile indexes to process"""
        # Compute cartesian product of region and tile (0-based) index list 
        return np.array(list(itertools.product(*(self.region_indexes, self.tile_indexes))))


def load_tiles(q, task_config):
    if task_config.op_flags.run_tile_generator:
        tile_gen_mode = tile_generator.TILE_GEN_MODE_RAW
    else:
        tile_gen_mode = tile_generator.TILE_GEN_MODE_STACK

    for region_index, tile_index in zip(task_config.region_indexes, task_config.tile_indexes):
        with tile_generator.CytokitTileGenerator(
                task_config.exp_config, task_config.data_dir,
                region_index, tile_index,
                mode=tile_gen_mode) as op:
            tile = op.run(None)
            logger.info('Loaded tile %s for region %s [shape = %s]', tile_index + 1, region_index + 1, tile.shape)
            q.put((tile, region_index, tile_index), block=True, timeout=TIMEOUT)


def init_dirs(output_dir):
    for path in [output_dir]:
        if not osp.exists(path):
            os.makedirs(path, exist_ok=True)


def initialize_task(task_config):
    # Initialize global GPU settings
    if task_config.gpu is not None:
        if op.get_gpu_device() is None:
            logger.debug('Setting gpu device {}'.format(task_config.gpu))
            op.set_gpu_device(task_config.gpu)
        else:
            logger.debug('GPU device already set to {}'.format(op.get_gpu_device()))

    # Initialize output directory
    init_dirs(task_config.output_dir)


def preprocess_tile(tile, tile_indices, ops, log_fn, task_config):
    output_dir = task_config.output_dir

    # Drift Compensation
    if ops.align_op:
        tile = ops.align_op.run(tile)
        log_fn('Drift compensation complete', tile)
    else:
        log_fn('Skipping drift compensation', debug=True)

    # Crop off overlap in imaging process
    if ops.crop_op:
        tile = ops.crop_op.run(tile)
        log_fn('Tile overlap crop complete', tile)
    else:
        log_fn('Skipping tile crop', debug=True)

    # Best Focal Plane Selection
    best_focus_data = None
    if ops.focus_op:
        # Used the cropped, but un-deconvolved tile for focal plane selection
        best_focus_data = ops.focus_op.run(tile)
        ops.focus_op.save(tile_indices, output_dir, best_focus_data)
        log_fn('Focal plane selection complete', best_focus_data[0])
    else:
        log_fn('Skipping focal plane selection', debug=True)

    # Deconvolution
    if ops.decon_op:
        tile = ops.decon_op.run(tile)
        log_fn('Deconvolution complete', tile)
    else:
        log_fn('Skipping deconvolution', debug=True)

    # Cytometry (segmentation + quantification)
    if ops.cytometry_op:
        best_focus_z_plane = best_focus_data[1] if best_focus_data else None
        cyto_data = ops.cytometry_op.run(tile, best_focus_z_plane=best_focus_z_plane)
        paths = ops.cytometry_op.save(tile_indices, output_dir, cyto_data)
        log_fn('Tile cytometry complete; Statistics saved to "{}"'.format(paths[-1]), cyto_data[0])
    else:
        log_fn('Skipping tile cytometry', debug=True)

    # Tile summary statistic operations
    if ops.summary_op:
        ops.summary_op.run(tile)
        log_fn('Tile statistic summary complete')
    else:
        log_fn('Skipping tile statistic summary', debug=True)

    # Save the output tile if tile generation/assembly was enabled
    if task_config.op_flags.run_tile_generator:
        path = cytokit_io.get_processor_img_path(tile_indices.region_index, tile_indices.tile_x, tile_indices.tile_y)
        cytokit_io.save_tile(osp.join(output_dir, path), tile, config=task_config.exp_config)
        log_fn('Saved preprocessed tile to path "{}"'.format(path), tile)


def postprocess_tile(tile, tile_indices, ops, log_fn, task_config):
    output_dir = task_config.output_dir

    # Illumination Correction
    if ops.illumination_op:
        # Prepare and save illumination images, if not already done
        ops.illumination_op.prepare_region_data(output_dir)
        path = ops.illumination_op.save_region_data(output_dir)
        if path is not None:
            log_fn('Illumination data saved to "{}"'.format(path))

        # Run correction for tile
        tile = ops.illumination_op.run(tile, tile_indices)
        log_fn('Illumination correction complete', tile)
    else:
        log_fn('Skipping illumination correction', debug=True)

    # Spectral Unmixing
    if ops.unmixing_op:
        # Prepare unmixing models for each region
        ops.unmixing_op.prepare_region_data(output_dir)

        # Run correction for tile
        tile = ops.unmixing_op.run(tile, tile_indices)
        log_fn('Spectral unmixing complete', tile)
    else:
        log_fn('Skipping spectral unmixing', debug=True)

    # Get best focus data
    # TODO Prevent needing to re-read the processor data file each time
    best_focus_data = function_data.get_best_focus_coord_map(output_dir)
    best_focus_z_plane = best_focus_data[(tile_indices.region_index, tile_indices.tile_x, tile_indices.tile_y)]

    # Rerun cytometry based on corrected tile
    cyto_data = ops.cytometry_op.run(tile, best_focus_z_plane=best_focus_z_plane)
    paths = ops.cytometry_op.save(tile_indices, output_dir, cyto_data)
    log_fn('Postprocessing cytometry complete; Statistics saved to "{}"'.format(paths[-1]), cyto_data[0])

    # Save resulting tile
    path = cytokit_io.get_processor_img_path(
        tile_indices.region_index, tile_indices.tile_x, tile_indices.tile_y)
    cytokit_io.save_tile(osp.join(output_dir, path), tile, config=task_config.exp_config)
    log_fn('Saved postprocessed tile to "{}"'.format(path), tile)


def get_log_fn(i, n_tiles, region_index, tx, ty):
    def log_fn(msg, res=None, debug=False):
        details = [
            'tile {} of {} ({:.2f}%)'.format(i + 1, n_tiles, 100*(i+1)/n_tiles),
            'reg/x/y = {}/{}/{}'.format(region_index + 1, tx + 1, ty + 1)
        ]
        if res is not None:
            details.append('shape {} / dtype {}'.format(res.shape, res.dtype))
        lf = logger.debug if debug else logger.info
        lf(msg + ' [' + ' | '.join(details) + ']')
    return log_fn


def get_preprocess_op_set(task_config):
    exp_config = task_config.exp_config
    return op.CytokitOpSet(
        align_op=drift_compensation.CytokitDriftCompensator(exp_config) if task_config.op_flags.run_drift_comp else None,
        focus_op=best_focus.CytokitFocalPlaneSelector(exp_config) if task_config.op_flags.run_best_focus else None,
        decon_op=deconvolution.CytokitDeconvolution(exp_config) if task_config.op_flags.run_deconvolution else None,
        summary_op=tile_summary.CytokitTileSummary(exp_config) if task_config.op_flags.run_summary else None,
        crop_op=tile_crop.CytokitTileCrop(exp_config) if task_config.op_flags.run_crop else None,
        cytometry_op=cytometry.get_op(exp_config) if task_config.op_flags.run_cytometry else None
    )


def get_postprocess_op_set(task_config):
    exp_config = task_config.exp_config
    return op.CytokitOpSet(
        illumination_op=illumination_correction.IlluminationCorrection(exp_config)
        if task_config.op_flags.run_illumination_correction else None,
        unmixing_op=spectral_unmixing.SpectralUnmixing(exp_config)
        if task_config.op_flags.run_spectral_unmixing else None,
        cytometry_op=cytometry.get_op(exp_config)
        if task_config.op_flags.postprocessing_enabled() else None
    )


def concat(datasets):
    """Merge dictionaries containing lists for each key"""
    res = {}
    for dataset in datasets:
        for k, v in dataset.items():
            res[k] = res.get(k, []) + v
    return res


def run_task(task_config, ops, process_fn):
    initialize_task(task_config)

    tile_queue = queue.Queue(maxsize=task_config.tile_prefetch_capacity)
    load_thread = Thread(target=load_tiles, args=(tile_queue, task_config))
    load_thread.start()

    measure_data = {}
    with ops:
        n_tiles = task_config.n_tiles
        for i in range(n_tiles):
            tile, region_index, tile_index = tile_queue.get(block=True, timeout=TIMEOUT)
            tx, ty = task_config.exp_config.get_tile_coordinates(tile_index)
            tile_indices = cytokit_config.TileIndices(
                region_index=region_index, tile_index=tile_index, tile_x=tx, tile_y=ty)

            # Set the "context" to store with monitor data meaning that the current tile location
            # will become global information associated automatically with all other statistics
            context = tile_indices._asdict()

            log_fn = get_log_fn(i, n_tiles, region_index, tx, ty)

            with op.new_monitor(context) as monitor:
                process_fn(tile, tile_indices, ops, log_fn, task_config)

                # Accumulate monitor data across tiles
                measure_data = concat([measure_data, monitor.data])

                log_fn('Processing complete')
                
    return measure_data


def run_tasks(pl_conf, task_type, task_fn, logging_init_fn):
    # Initialize local dask cluster
    logger.debug('Pipeline configuration: %s', pl_conf)
    cluster = LocalCluster(
        n_workers=pl_conf.n_workers, threads_per_worker=1,
        processes=True, memory_limit=pl_conf.memory_limit,
        ip='0.0.0.0'
    )
    client = Client(cluster)

    # Split total region + tile indexes to process into separate lists for each worker
    # (by indexes of those combinations)
    tiles = pl_conf.region_tiles
    idx_batches = np.array_split(np.arange(len(tiles)), pl_conf.n_workers)

    # Assign gpus to tasks in round-robin fashion
    def get_gpu(i):
        if pl_conf.gpus is None:
            return None
        return pl_conf.gpus[i % len(pl_conf.gpus)]

    # Generate a single task configuration for each worker
    tasks = [
        pl_conf.get_task_config(region_indexes=tiles[idx_batch, 0], tile_indexes=tiles[idx_batch, 1], gpu=get_gpu(i))
        for i, idx_batch in enumerate(idx_batches)
    ]

    logger.info('Starting %s pipeline for %s tasks (%s workers)', task_type, len(tasks), pl_conf.n_workers)
    logger.debug('Task definitions:\n\t%s', '\n\t'.join([str(t) for t in tasks]))
    try:
        # Passing logging initialization operation, if given, to workers now
        # running in separate processes
        if logging_init_fn:
            client.run(logging_init_fn)

        # Disable the "auto_restart" feature of dask workers which is of no use in this context
        for worker in cluster.workers:
            worker.auto_restart = False

        # Pass tasks to each worker to execute in parallel
        res = client.map(task_fn, tasks)
        res = [r.result() for r in res]
        if len(res) != len(tasks):
            raise ValueError('Parallel execution returned {} results but {} were expected'.format(len(res), len(tasks)))
    finally:
        # Note that this often produces a non-critical error due to: https://github.com/dask/distributed/issues/1969
        # but that closing these resources is necessary to avoid GPU oom in post-processing
        client.close()
        cluster.close()

    # Save measurement data to disk
    measure_data = concat(res)
    if measure_data:
        path = exec.record_processor_data(measure_data, pl_conf.output_dir)
        logging.info('%s complete; Measurement data saved to "%s"', task_type, path)
    else:
        logging.info('%s complete', task_type)


def run_preprocess_task(task):
    ops = get_preprocess_op_set(task)
    return run_task(task, ops, preprocess_tile)


def run_postprocess_task(task):
    # Configure the task to use an input directory equal to pre-processing output and to source
    # preprocessed tiles instead of raw files
    task.data_dir = task.output_dir
    task.op_flags.run_tile_generator = False

    ops = get_postprocess_op_set(task)
    return run_task(task, ops, postprocess_tile)


def run(pl_conf, logging_init_fn=None):
    start = timer()

    if pl_conf.op_flags.preprocessing_enabled():
        run_tasks(pl_conf, 'Pre-processing', run_preprocess_task, logging_init_fn)

    if pl_conf.op_flags.postprocessing_enabled():
        run_tasks(pl_conf, 'Post-processing', run_postprocess_task, logging_init_fn)

    stop = timer()
    logger.info('Pipeline execution completed in %.0f seconds', stop - start)




