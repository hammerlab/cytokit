import tensorflow as tf
import os


class OpGraph(object):

    def __init__(self, tf_graph, inputs, outputs):
        self.tf_graph = tf_graph
        self.inputs = inputs
        self.outputs = outputs


GPU_DEVICE = None


def set_gpu_device(device):
    global GPU_DEVICE
    GPU_DEVICE = device


def get_gpu_device():
    global GPU_DEVICE
    return GPU_DEVICE


def get_tf_config(cpu_only=False):
    global GPU_DEVICE
    if cpu_only:
        config = tf.ConfigProto(device_count={'GPU': 0}, log_device_placement=False)
        config.gpu_options.visible_device_list = ''
    else:
        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True
        if GPU_DEVICE is not None:
            config.gpu_options.visible_device_list = str(GPU_DEVICE)
    return config


class TensorFlowOp(object):

    def __init__(self, cpu_only=False):
        self.graph = None
        self.primary_dtype = tf.float32
        self.cpu_only = cpu_only

    def initialize(self):
        graph = tf.Graph()
        with graph.as_default():
            inputs, outputs = self._build_graph()
        self.graph = OpGraph(graph, inputs, outputs)
        return self

    def _build_graph(self):
        raise NotImplementedError()

    def args(self, *args, **kwargs):
        raise NotImplementedError()

    def run(self, *args, **kwargs):
        return next(self.flow([self.args(*args, **kwargs)]))

    def flow(self, args_generator):
        if self.graph is None:
            raise ValueError('Must initialize operation before running (via `.initialize` method)')

        with tf.Session(config=get_tf_config(self.cpu_only), graph=self.graph.tf_graph) as sess:
            for args in args_generator:
                args_dict = {self.graph.inputs[k]: v for k, v in args.items() if v is not None}
                res = sess.run(self.graph.outputs, feed_dict=args_dict)
                yield res


class CodexOp(object):

    def __init__(self, config):
        self.config = config

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, type, value, traceback):
        if type:
            raise value
        else:
            self.shutdown()
            return True

    def initialize(self):
        pass

    def shutdown(self):
        pass
