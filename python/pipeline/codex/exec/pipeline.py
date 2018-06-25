"""CODEX preprocessing pipeline core logic

This is not intended to be run directly but rather used by mutliple external
interfaces to implement the core process that comprises CODEX processing.
"""
import os, logging, itertools, queue
import numpy as np
from os import path as osp
from threading import Thread
from timeit import default_timer as timer
from codex import io as codex_io
from codex import config as codex_config
from codex.ops import op
from codex.ops import tile_generator
from codex.ops import cytometry
from codex.ops import tile_crop
from codex.ops import drift_compensation
from codex.ops import best_focus
from codex.ops import deconvolution
from codex.ops import tile_summary
from dask.distributed import Client, LocalCluster
logger = logging.getLogger(__name__)

# Set 1 hour time limit on tile loading/reading operations
TIMEOUT = 1 * 60 * 60


class TaskConfig(object):

    def __init__(self, pipeline_config, region_indexes, tile_indexes, gpu, tile_prefetch_capacity=2, 
            run_best_focus=True, run_drift_comp=True, run_summary=True,
            run_tile_generator=True, run_crop=True, run_deconvolution=True, run_cytometry=True):
        self.region_indexes = region_indexes
        self.tile_indexes = tile_indexes
        self.config_path = pipeline_config.config_path
        self.data_dir = pipeline_config.data_dir
        self.output_dir = pipeline_config.output_dir
        self.gpu = gpu
        self.tile_prefetch_capacity = tile_prefetch_capacity
        self.run_tile_generator = run_tile_generator
        self.run_crop = run_crop
        self.run_deconvolution = run_deconvolution
        self.run_drift_comp = run_drift_comp
        self.run_best_focus = run_best_focus
        self.run_summary = run_summary
        self.run_cytometry = run_cytometry
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

    def __init__(self, region_indexes, tile_indexes, config_path, data_dir, output_dir, n_workers,
                 gpus, memory_limit, **task_kwargs):
        self.region_idx = region_indexes
        self.tile_idx = tile_indexes
        self.config_path = config_path
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.n_workers = n_workers
        self.gpus = gpus
        self.memory_limit = memory_limit
        self.task_kwargs = task_kwargs

        # Load experiment configuration in order to determine defaults
        self.exp_config = codex_config.load(config_path)

        # "Register the enviornment" meaning that any variables not explicitly defined by env variables
        # should set based on what is present in the configuration
        self.exp_config.register_environment()

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
            k:v for k, v in self.__dict__.items() 
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
    if task_config.run_tile_generator:
        tile_gen_mode = tile_generator.TILE_GEN_MODE_RAW
    else:
        tile_gen_mode = tile_generator.TILE_GEN_MODE_STACK

    for region_index, tile_index in zip(task_config.region_indexes, task_config.tile_indexes):
        with tile_generator.CodexTileGenerator(
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


def process_tile(tile, ops, log_fn):

    # Drift Compensation
    if ops.align_op:
        align_tile = ops.align_op.run(tile)
        log_fn('Drift compensation complete', align_tile)
    else:
        align_tile = tile
        log_fn('Skipping drift compensation')

    if ops.crop_op:
        crop_tile = ops.crop_op.run(align_tile)
        log_fn('Tile overlap crop complete', crop_tile)
    else:
        crop_tile = tile
        log_fn('Skipping tile crop')

    # Deconvolution
    if ops.decon_op:
        decon_tile = ops.decon_op.run(crop_tile)
        log_fn('Deconvolution complete', decon_tile)
    else:
        decon_tile = crop_tile
        log_fn('Skipping deconvolution')

    # Best Focal Plane Selection
    focus_tile, focus_z_plane = None, None
    if ops.focus_op:
        focus_z_plane, _, _ = ops.focus_op.run(tile)
        focus_tile = decon_tile[:, [focus_z_plane], :, :, :]
        log_fn('Focal plane selection complete', focus_tile)
    else:
        log_fn('Skipping focal plane selection')

    # Tile summary statistic operations
    if ops.summary_op:
        ops.summary_op.run(decon_tile)
        log_fn('Tile statistic summary complete')
    else:
        log_fn('Skipping tile statistic summary')

    res_tile = decon_tile

    cyto_seg_tile, cyto_viz_tile, cyto_stats = None, None, None
    if ops.cytometry_op:
        cyto_seg_tile, cyto_viz_tile, cyto_stats = ops.cytometry_op.run(decon_tile)
        log_fn('Tile cytometry complete')
    else:
        log_fn('Skipping tile cytometry')

    return res_tile, (focus_tile, focus_z_plane), (cyto_seg_tile, cyto_viz_tile, cyto_stats)


def get_log_fn(i, n_tiles, region_index, tx, ty):
    def log_fn(msg, res=None):
        details = [
            'tile {} of {} ({:.2f}%)'.format(i + 1, n_tiles, 100*(i+1)/n_tiles),
            'reg/x/y = {}/{}/{}'.format(region_index + 1, tx + 1, ty + 1)
        ]
        if res is not None:
            details.append('shape {} / dtype {}'.format(res.shape, res.dtype))
        logger.info(msg + ' [' + ' | '.join(details) + ']')
    return log_fn


def get_op_set(task_config):
    exp_config = task_config.exp_config
    return op.CodexOpSet(
        align_op=drift_compensation.CodexDriftCompensator(exp_config) if task_config.run_drift_comp else None,
        focus_op=best_focus.CodexFocalPlaneSelector(exp_config) if task_config.run_best_focus else None,
        decon_op=deconvolution.CodexDeconvolution(exp_config) if task_config.run_deconvolution else None,
        summary_op=tile_summary.CodexTileSummary(exp_config) if task_config.run_summary else None,
        crop_op=tile_crop.CodexTileCrop(exp_config) if task_config.run_crop else None,
        cytometry_op=cytometry.Cytometry(exp_config) if task_config.run_cytometry else None
    )


def concat(datasets):
    """Merge dictionaries containing lists for each key"""
    res = {}
    for dataset in datasets:
        for k, v in dataset.items():
            res[k] = res.get(k, []) + v
    return res


def run_pipeline_task(task_config):
    initialize_task(task_config)

    tile_queue = queue.Queue(maxsize=task_config.tile_prefetch_capacity)
    load_thread = Thread(target=load_tiles, args=(tile_queue, task_config))
    load_thread.start()

    ops = get_op_set(task_config)

    monitor_data = {}
    with ops:
        n_tiles = task_config.n_tiles
        for i in range(n_tiles):
            tile, region_index, tile_index = tile_queue.get(block=True, timeout=TIMEOUT)
            tx, ty = task_config.exp_config.get_tile_coordinates(tile_index)

            context = dict(tile=tile_index, region=region_index, tile_x=tx, tile_y=ty)
            log_fn = get_log_fn(i, n_tiles, region_index, tx, ty)

            with op.new_monitor(context) as monitor:
                res_tile, focus_data, cyto_data = process_tile(tile, ops, log_fn)

                # Save z-plane for stack if best focal plane selection is enabled
                if ops.focus_op:
                    focus_tile, focus_z_plane = focus_data
                    img_path = codex_io.get_best_focus_img_path(region_index, tx, ty, focus_z_plane)
                    codex_io.save_tile(osp.join(task_config.output_dir, img_path), focus_tile)
                    log_fn('Saved best focus tile to path "{}"'.format(img_path), focus_tile)

                # Save cytometry data if enabled
                if ops.cytometry_op:
                    cyto_seg_tile, cyto_viz_tile, cyto_stats = cyto_data

                    cyto_tile_path = codex_io.get_cytometry_file_path(region_index, tx, ty, 'seg.tif')
                    codex_io.save_tile(osp.join(task_config.output_dir, cyto_tile_path), cyto_seg_tile)

                    cyto_tile_path = codex_io.get_cytometry_file_path(region_index, tx, ty, 'viz.tif')
                    codex_io.save_tile(osp.join(task_config.output_dir, cyto_tile_path), cyto_viz_tile)

                    # Append useful metadata to cytometry stats
                    cyto_stats.insert(0, 'tile_y', ty + 1)
                    cyto_stats.insert(0, 'tile_x', tx + 1)
                    cyto_stats.insert(0, 'tile_i', tile_index + 1)
                    cyto_stats.insert(0, 'region', region_index + 1)
                    cyto_stats_path = codex_io.get_cytometry_file_path(region_index, tx, ty, 'csv')
                    cyto_stats.to_csv(osp.join(task_config.output_dir, cyto_stats_path), index=False)

                    log_fn('Saved cytometry data to path "{}"'.format(cyto_stats_path), cyto_seg_tile)

                # Save the output tile if tile generation/assembly was enabled
                if task_config.run_tile_generator:
                    # Save the tile resulting from pipeline execution
                    res_path = codex_io.get_processor_img_path(region_index, tx, ty)
                    codex_io.save_tile(osp.join(task_config.output_dir, res_path), res_tile)
                    log_fn('Saved processed tile to path "{}"'.format(res_path), res_tile)

                # Accumulate monitor data across tiles
                monitor_data = concat([monitor_data, monitor.data])

                log_fn('Processing complete')
                
    return monitor_data


def run(pl_conf, logging_init_fn=None):
    start = timer()

    # Initialize local dask cluster
    logger.info('Initializing pipeline tasks for %s workers', pl_conf.n_workers)
    logger.debug('Pipeline configuration: %s', pl_conf)
    cluster = LocalCluster(
        n_workers=pl_conf.n_workers, threads_per_worker=1,
        processes=True, memory_limit=pl_conf.memory_limit
    )
    client = Client(cluster)

    # Split total region + tile indexes to process into separate lists for each worker 
    # (by indexes of those index combinations)
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

    logger.info('Starting pipeline for %s tasks', len(tasks))
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
        res = client.map(run_pipeline_task, tasks)
        res = [r.result() for r in res]
        if len(res) != len(tasks):
            raise ValueError('Parallel execution returned {} results but {} were expected'.format(len(res), len(tasks)))
        stop = timer()
        logger.info('Pipeline execution completed in %.0f seconds', stop - start)
    finally:
        client.close()
        cluster.close()

    # Merge monitoring data across pipeline tasks and return result
    return concat(res)




