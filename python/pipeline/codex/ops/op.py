import tensorflow as tf


def get_tf_config():
    return tf.ConfigProto()


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

        with tf.Session(config=get_tf_config(), graph=self.graph.tf_graph) as sess:
            for args in args_generator:
                args_dict = {self.graph.inputs[k]: v for k, v in args.items() if v is not None}
                res = sess.run(self.graph.outputs, feed_dict=args_dict)
                yield res


class CodexOp(object):

    def __init__(self, config):
        self.config = config
