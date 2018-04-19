
from codex.ops import tile_generator
from codex.ops import tile_crop
from codex.ops import drift_compensation
from codex.ops import best_focus
from codex.ops import deconvolution
from codex import config as codex_config
from timeit import default_timer as timer
from skimage.external.tifffile import imread, imsave
import queue
from dask.distributed import Client, LocalCluster
import numpy as np
import os
from os import path as osp 
from threading import Thread
import logging

TIMEOUT = 4*60*60

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(name)s:%(process)d/%(threadName)s: %(message)s"
)

logger = logging.getLogger('Pipeline')

def load_tiles(q, data_dir, config, region_index, tile_indexes):
    i = 0
    for tile_index in tile_indexes:
        with tile_generator.CodexTileGenerator(config, data_dir, region_index, tile_index) as op:
            tile = op.run()
        logger.info('Loaded tile {} [shape = {}]'.format(tile_index, tile.shape))
        q.put((tile, tile_index), block=True, timeout=TIMEOUT)
        
def init_dirs(output_dir):
    for path in [osp.join(output_dir, 'bestFocus')]:
        if not osp.exists(path):
            os.makedirs(path, exist_ok=True)

def save_tile(file, tile):
    if not osp.exists(osp.dirname(file)):
        os.makedirs(osp.dirname(file), exist_ok=True)
    imsave(file, tile, imagej=True)
    
    
def run_pipeline(args):
    from codex.ops import op
    tile_indexes, gpu = args
    
    if op.get_gpu_device() is None:
        logger.info('Setting gpu device {}'.format(gpu))
        op.set_gpu_device(gpu)
    else:
        logger.info('GPU device already set to {}'.format(op.get_gpu_device()))
    
    data_dir = 'F:\\7-7-17-multicycle'
    output_dir = 'F:\\7-7-17-multicycle-out-pipeline\\1-Processor'
    region_index = 0
    tile_prefetch_capacity = 2
    config = codex_config.load(data_dir)
    init_dirs(output_dir)
    
    tile_queue = queue.Queue(maxsize=tile_prefetch_capacity)
    load_thread = Thread(target=load_tiles, args=(tile_queue, data_dir, config, region_index, tile_indexes))
    load_thread.start()

    for _ in range(len(tile_indexes)):
        tile, tile_index = tile_queue.get(block=True, timeout=TIMEOUT)
        tx, ty = tile_index // 5, tile_index % 5
        logger.info('Beginning processing for tile {} (X={}, Y={})'.format(tile_index, tx, ty))
        
        with drift_compensation.CodexDriftCompensator(config) as op:
            tile_aligned = op.run(tile)
            logger.info('Finished drift compensation for tile {} [shape = {}]'.format(tile_index, tile_aligned.shape))
            
        with tile_crop.CodexTileCrop(config) as op:
            crop_tile = op.run(tile_aligned)
            logger.info('Finished crop for tile {} [shape = {}]'.format(tile_index, crop_tile.shape))
            
        # with best_focus.CodexFocalPlaneSelector(config) as op:
        #     focus_tile, best_z, classifications, probabilities = op.run(tile)
            
        #     logger.info('Best focus classifications for tile {}: {}'.format(tile_index, classifications))
        #     focus_file = os.path.join(
        #         output_dir, 'bestFocus',
        #         'reg{:03d}_X{:02d}_Y{:02d}_Z{:02d}.tif'.format(region_index + 1, tx + 1, ty + 1, best_z + 1)
        #     )
        #     logger.info('Saving best focus tile to path "{}" [shape = {}]'.format(focus_file, focus_tile.shape))
        #     save_tile(focus_file, focus_tile)
        
        with deconvolution.CodexDeconvolution(config, n_iter=25) as op:
            decon_tile = op.run(crop_tile)
            logger.info('Finished deconvolution for tile {} [shape = {}]'.format(tile_index, decon_tile.shape))
            
        res_tile = decon_tile
        res_file = os.path.join(output_dir, 'reg{:03d}_X{:02d}_Y{:02d}.tif'.format(region_index + 1, tx + 1, ty + 1))
        logger.info('Saving result to path "{}" [shape = {}]'.format(res_file, res_tile.shape))
        save_tile(res_file, res_tile)
        
        logger.info('Processing for tile {} complete'.format(tile_index))
    
    return 0

def main():
    import tensorflow as tf
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    tf.logging.set_verbosity(tf.logging.WARN)
    
    #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    start = timer()
    
    n_workers = 3
    n_gpus = 2
    
    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1, processes=True, memory_limit=32e9)
    client = Client(cluster)
    logger.info('Created cluster')

    try:
        workers = cluster.workers
        #client.run(init_logging)
        
        tiles = np.array_split(np.arange(25), n_workers)
        print('Tile batches:', tiles)
        args = [(tiles[i], i % n_gpus) for i in range(n_workers)]
        res = client.map(run_pipeline, args)
        res = [r.result() for r in res]
    finally:
        client.close()
        cluster.close()

    stop = timer()
    
    print('Execution time:', stop - start)
    
if __name__ == '__main__':
    main()
 

