import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

def init_session(gpu_fraction=0.75):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    KTF.set_session(tf.Session(config=config))