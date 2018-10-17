import os
import re
import cytokit
import tensorflow as tf
from timeit import default_timer as timer
GPU_DEVICE = None


def set_gpu_device(device):
    global GPU_DEVICE
    GPU_DEVICE = device


def get_gpu_device():
    global GPU_DEVICE
    return GPU_DEVICE


def is_cpu_only(op_class):
    cpu_only_ops = os.getenv(cytokit.ENV_CPU_ONLY_OPS, None)
    if cpu_only_ops is None:
        return False
    return op_class.lower() in [op.lower().strip() for op in cpu_only_ops.split(',')]


def get_tf_config(op, cpu_only=None):
    global GPU_DEVICE
    if cpu_only is None:
        cpu_only = is_cpu_only(op.__class__.__name__)
    if cpu_only:
        config = tf.ConfigProto(device_count={'GPU': 0}, log_device_placement=False)
        config.gpu_options.visible_device_list = ''
    else:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.)
        config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        if GPU_DEVICE is not None:
            config.gpu_options.visible_device_list = str(GPU_DEVICE)
    return config


class OpMonitor(object):

    def __init__(self, context):
        self.context = context
        self.data = {}

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if type:
            raise value
        else:
            return True

    def record(self, op, data):
        if op not in self.data:
            self.data[op] = []
        # Merge global context into operation data and add to 
        # list of all records for this operation
        self.data[op].append({**self.context, **data})
        return self


CURRENT_MONITOR = OpMonitor({})


def new_monitor(context):
    """Create a new operation monitor with the given global context

    Args:
        context: Dictionary containing any globally relevent information for this monitored session
    Returns:
        New OpMonitor instance
    """
    global CURRENT_MONITOR
    CURRENT_MONITOR = OpMonitor(context)
    return CURRENT_MONITOR


def add_monitor_data(op, data):
    global CURRENT_MONITOR
    CURRENT_MONITOR.record(op, data)


class MonitorMixin(object):

    def get_op_name(self):
        return self.__class__.__name__

    def add_monitor_data(self, data):
        op = self.get_op_name().lower().strip()
        add_monitor_data(op, data)


class OpGraph(object):

    def __init__(self, tf_graph, inputs, outputs):
        self.tf_graph = tf_graph
        self.inputs = inputs
        self.outputs = outputs


class TensorFlowOp(object):

    def __init__(self):
        self.graph = None
        self.primary_dtype = tf.float32

    def initialize(self):
        graph = tf.Graph()
        with graph.as_default():  # pylint: disable=not-context-manager
            inputs, outputs = self._build_graph()
        self.graph = OpGraph(graph, inputs, outputs)
        return self

    def _build_graph(self):
        raise NotImplementedError()

    def args(self, *args, **kwargs):
        raise NotImplementedError()

    def run(self, *args, **kwargs):
        return self.flow([self.args(*args, **kwargs)])[0]

    def flow(self, args_generator):
        if self.graph is None:
            raise ValueError('Must initialize operation before running (via `.initialize` method)')

        with tf.Session(config=get_tf_config(self), graph=self.graph.tf_graph) as sess:
            results = []
            for args in args_generator:
                args_dict = {self.graph.inputs[k]: v for k, v in args.items() if v is not None}
                res = sess.run(self.graph.outputs, feed_dict=args_dict)
                results.append(res)
            return results


def _to_snake_case(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


class CytokitOp(MonitorMixin):

    def __init__(self, config):
        self.config = config
        self._records = None

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, type, value, traceback):
        if type:
            raise value
        else:
            self.shutdown()
            return True

    @staticmethod
    def get_op_for_class(c):
        return _to_snake_case(c.__name__.replace('Cytokit', ''))

    def get_op_name(self):
        return CytokitOp.get_op_for_class(self.__class__)

    def record(self, data):
        self._records.append(data)

    def _run(self, *args, **kwargs):
        raise NotImplementedError()

    def run(self, *args, **kwargs):
        # Reset monitor record list
        self._records = []
        res = self._run(*args, **kwargs)
        for monitor_record in self._records:
            self.add_monitor_data(monitor_record)
        return res

    def initialize(self):
        pass

    def shutdown(self):
        pass


class CytokitOpSet(object):

    def __init__(self, **ops):
        self.ops = ops
        for k, v in ops.items():
            setattr(self, k, v) 

    def __enter__(self):
        for v in self.ops.values():
            if v is not None:
                v.__enter__()

    def __exit__(self, type, value, traceback):
        for v in self.ops.values():
            if v is not None:
                v.__exit__(type, value, traceback)

