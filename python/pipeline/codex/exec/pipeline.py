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
from codex.ops import tile_crop
from codex.ops import drift_compensation
from codex.ops import best_focus
from codex.ops import deconvolution
from dask.distributed import Client, LocalCluster
logger = logging.getLogger(__name__)

# Set 1 hour time limit on tile loading/reading operations
TIMEOUT = 1 * 60 * 60


class TaskConfig(object):

    def __init__(self, pipeline_config, region_indexes, tile_indexes, gpu, tile_prefetch_capacity=2, run_best_focus=True, n_iter_decon=25):
        self.region_indexes = region_indexes
        self.tile_indexes = tile_indexes
        self.config_dir = pipeline_config.config_dir
        self.data_dir = pipeline_config.data_dir
        self.output_dir = pipeline_config.output_dir
        self.gpu = gpu
        self.tile_prefetch_capacity = tile_prefetch_capacity
        self.run_best_focus = run_best_focus
        self.n_iter_decon = n_iter_decon
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

    def __init__(self, region_indexes, tile_indexes, config_dir, data_dir, output_dir, n_workers,
                 gpus, memory_limit, **task_kwargs):
        self.region_idx = region_indexes
        self.tile_idx = tile_indexes
        self.config_dir = config_dir
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.n_workers = n_workers
        self.gpus = gpus
        self.memory_limit = memory_limit
        self.task_kwargs = task_kwargs

        # Load experiment configuration in order to determine defaults
        self.exp_config = codex_config.load(config_dir)

        # Default region and tile index list to that in experiment configuration if not provided explicitly
        if self.region_idx is None:
            self.region_idx = self.exp_config.region_indexes
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
        return np.array(self.region_idx) - 1

    @property
    def tile_indexes(self):
        return np.array(self.tile_idx) - 1

    @property
    def region_tiles(self):
        """Get 0-based pairs of region and tile indexes to process"""
        # Compute cartesian product of region and tile (0-based) index list 
        return np.array(list(itertools.product(*(self.region_indexes, self.tile_indexes))))


def load_tiles(q, task_config):
    for region_index, tile_index in zip(task_config.region_indexes, task_config.tile_indexes):
        with tile_generator.CodexTileGenerator(task_config.exp_config, task_config.data_dir, region_index, tile_index) as op:
            tile = op.run()
            logger.info('Loaded tile %s for region %s [shape = %s]', tile_index + 1, region_index + 1, tile.shape)
            q.put((tile, region_index, tile_index), block=True, timeout=TIMEOUT)


def init_dirs(output_dir):
    for path in [output_dir]:
        if not osp.exists(path):
            os.makedirs(path, exist_ok=True)


def _initialize(task_config):
    if task_config.gpu is not None:
        if op.get_gpu_device() is None:
            logger.debug('Setting gpu device {}'.format(task_config.gpu))
            op.set_gpu_device(task_config.gpu)
        else:
            logger.debug('GPU device already set to {}'.format(op.get_gpu_device()))


def run_pipeline_task(task_config):
    _initialize(task_config)

    exp_config = task_config.exp_config
    init_dirs(task_config.output_dir)

    tile_queue = queue.Queue(maxsize=task_config.tile_prefetch_capacity)
    load_thread = Thread(target=load_tiles, args=(tile_queue, task_config))
    load_thread.start()

    if task_config.run_best_focus:
        focus_op = best_focus.CodexFocalPlaneSelector(exp_config).initialize()

    times = []
    with drift_compensation.CodexDriftCompensator(exp_config) as align_op, \
            tile_crop.CodexTileCrop(exp_config) as crop_op, \
            deconvolution.CodexDeconvolution(exp_config, n_iter=task_config.n_iter_decon) as decon_op:
        n_tiles = task_config.n_tiles
        for i in range(n_tiles):
            start = timer()
            tile, region_index, tile_index = tile_queue.get(block=True, timeout=TIMEOUT)
            tx, ty = exp_config.get_tile_coordinates(tile_index)

            def log(msg, res=None):
                details = [
                    'tile {} of {} ({:.2f}%)'.format(i + 1, n_tiles, 100*(i+1)/n_tiles),
                    'reg/x/y = {}/{}/{}'.format(region_index + 1, tx + 1, ty + 1)
                ]
                if res is not None:
                    details.append('shape {}'.format(res.shape))
                logger.info(msg + ' [' + ' | '.join(details) + ']')

            align_tile = align_op.run(tile)
            log('Drift compensation complete', align_tile)

            crop_tile = crop_op.run(align_tile)
            log('Tile overlap crop complete', crop_tile)

            decon_tile = decon_op.run(crop_tile)
            log('Deconvolution complete', decon_tile)

            if task_config.run_best_focus:
                best_z, classifications, probabilities = focus_op.run(tile)
                focus_tile = decon_tile[:, best_z, :, :, :]

                log('Best focus classifications: {}'.format(classifications), focus_tile)
                img_path = codex_io.get_best_focus_img_path(region_index, tx, ty, best_z)

                log('Saving best focus tile to path "{}"'.format(img_path), focus_tile)
                codex_io.save_tile(osp.join(task_config.output_dir, img_path), focus_tile)

            res_tile = decon_tile
            res_path = codex_io.get_processor_img_path(region_index, tx, ty)
            log('Saving result to path "{}"'.format(res_path), res_tile)
            codex_io.save_tile(osp.join(task_config.output_dir, res_path), res_tile)

            log('Processing complete')
            stop = timer()
            times.append((region_index, tile_index, stop - start))

    if task_config.run_best_focus:
        focus_op.shutdown()

    return np.array(times)


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
        logger.debug('Per-tile execution times:\n%s', '\n'.join([str(v) for v in res]))
        logger.info('Pipeline execution completed in %s seconds', stop - start)
    finally:
        client.close()
        cluster.close()




