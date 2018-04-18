
from codex.ops import tile_generator
from codex.ops import drift_compensation
from codex import config as codex_config
from timeit import default_timer as timer
from dask.distributed import Client, LocalCluster
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(name)s:%(process)d/%(threadName)s: %(message)s"
)

logger = logging.getLogger('Pipeline')


def run_pipeline(args):
    from codex.ops import op
    from skimage.external.tifffile import imsave
    tile_index, gpu = args
    
    if op.get_gpu_device() is None:
        logger.info('Setting gpu device {}'.format(gpu))
        op.set_gpu_device(gpu)
    else:
        logger.info('GPU device already set to {}'.format(op.get_gpu_device()))
    
    data_dir = 'F:\\7-7-17-multicycle'
    output_dir = 'F:\\7-7-17-multicycle-out-pipeline'
    region_index = 0
    conf = codex_config.load(data_dir)
    
    with tile_generator.CodexTileGenerator(conf, data_dir, region_index, tile_index) as op:
        tile = op.run()
        logger.info('Loaded tile {} [shape = {}]'.format(tile_index, tile.shape))
    
    with drift_compensation.CodexDriftCompensator(config) as op:
        tile_aligned = op.run(tile)
        logger.info('Finished drift compensation for tile {} [shape = {}]'.format(tile_index, tile_aligned.shape))
        
    with tile_crop.CodexTileCrop(config) as op:
        crop_tile = op.run(tile_aligned)
        logger.info('Finished crop for tile {} [shape = {}]'.format(tile_index, crop_tile.shape))
        
    # with best_focus.CodexFocalPlaneSelector(config) as op:
    #     focus_tile, best_z, classifications, probabilities = op.run(tile)
    
    with deconvolution.CodexDeconvolution(config, n_iter=25) as op:
        decon_tile = op.run(crop_tile)
        logger.info('Finished deconvolution for tile {} [shape = {}]'.format(tile_index, decon_tile.shape))
        
    res_tile = decon_tile
    
    tx = tile_index // 5
    ty = tile_index % 5
    tile_file = os.path.join(output_dir, 'reg{:03d}_X{:02d}_Y{:02d}.tif'.format(region_index + 1, tx + 1, ty + 1))
    logger.info('Saving result to path "{}" [shape = {}]'.format(tile_file, res_tile.shape))
    imsave(tile_file, res_tile, imagej=True)
    
    logger.info('Processing for tile {} complete'.format(tile_index))
    
    return 0

def main():
    import tensorflow as tf
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    tf.logging.set_verbosity(tf.logging.WARN)
    
    #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    start = timer()
    
    cluster = LocalCluster(n_workers=5, threads_per_worker=1, processes=True, memory_limit=32e9)
    client = Client(cluster)
    logger.info('Created cluster')

    try:
        workers = cluster.workers
        #client.run(init_logging)
        
        ngpus = 2
        args = [(i, i % ngpus) for i in range(25)]
        res = client.map(run_pipeline, args)
        res = [r.result() for r in res]
    finally:
        client.close()
        cluster.close()

    stop = timer()
    
    print('Execution time:', stop - start)
    
if __name__ == '__main__':
    main()
 

