from flowdec import restoration as fd_restoration
from flowdec import data as fd_data
import time
import numpy as np
import logging
logging.basicConfig()
logger = logging.getLogger('NB')

def run_mult(device):
    for i in range(1000):
        np.matmul(np.ones((1000, 1000)), np.ones((1000, 1000)))
    return 1

def run_deconvolution(device):
    import os
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = device
    #acq = fd_data.load_celegans_channel('CY3')
    
    for i in range(10):
        acq = fd_data.bars_25pct()
        algo = fd_restoration.RichardsonLucyDeconvolver(3).initialize()
        res = algo.run(acq, niter=50, session_config=config).data
        time.sleep(5)
    return res

def set_device(device):
    import os
    print('Pid of process: ', os.getpid())
    
def init_logging():
    import logging
    
if __name__ == '__main__':
    from dask.distributed import Client, LocalCluster
    cluster = LocalCluster(n_workers=2, threads_per_worker=1, processes=True, memory_limit=16e9)
    client = Client(cluster)
    logger.info('Created cluster')

    try:
        workers = cluster.workers
        client.run(init_logging)
        client.run(set_device, '0', workers=[workers[0].address])
        client.run(set_device, '1', workers=[workers[1].address])
        print('Done set device')
        res = client.map(run_deconvolution, ['0', '1'])
        #res = client.map(run_mult, ['/cpu:0']*3)
        res = [r.result() for r in res]
        logs = client.get_worker_logs()
    finally:
        client.close()
        cluster.close()
        
    print('Done')
        
  