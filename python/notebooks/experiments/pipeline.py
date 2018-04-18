
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

logger = logging.getLogger('PipelineTest')


def run_pipeline(args):
    from codex.ops import op
    tile_index, gpu = args
    
    logger.info('Setting gpu device {}'.format(gpu))
    op.set_gpu_device(gpu)
    
    data_dir = 'F:\\7-7-17-multicycle'
    
    conf = codex_config.load(data_dir)
    
    op = tile_generator.CodexTileGenerator(conf, data_dir, 0, tile_index)
    tile = op.run()
    
    logger.info('Loaded tile {} with shape {}'.format(tile_index, tile.shape))
    
    op = drift_compensation.CodexDriftCompensator(conf).initialize()
    res = op.run(tile)
    
    logger.info('Pipeline result shape {}'.format(res.shape))
    
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
 

