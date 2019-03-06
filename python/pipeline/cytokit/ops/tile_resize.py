from cytokit.ops.op import TensorFlowOp
import tensorflow as tf
import logging
logger = logging.getLogger(__name__)


class CytokitTileResize(TensorFlowOp):

    def __init__(self, method='BILINEAR'):
        """Create resizing instance

        Args:
            method: Any option available via tf.image.ResizeMethod including 'BILINEAR', 'AREA', 'BICUBIC' and
                'NEAREST_NEIGHBOR'
        """
        super().__init__()
        # Note that this must be specified outside of the graph since internally an integer equality comparison
        # is done to resolve the method (which doesn't work if method is a scalar tensor)
        self.method = getattr(tf.image.ResizeMethod, method) if isinstance(method, str) else method

    def _build_graph(self):
        images = tf.placeholder(self.primary_dtype, shape=[None] * 4, name='images')
        shape = tf.placeholder(tf.int32, shape=[2], name='shape')
        result = tf.image.resize_images(images, shape, method=self.method, align_corners=True)
        inputs = dict(images=images, shape=shape)
        outputs = dict(result=result)
        return inputs, outputs

    def args(self, images, shape):
        """Get arguments for resize

        Args:
            images: An array with shape (batch, height, width, channels)
            shape: Target shape for resize as two item sequence (height, width)
        """
        if images.ndim != 4:
            raise ValueError('Images must have shape (batch, height, width, channels) not {}'.format(images.shape))
        if len(shape) != 2:
            raise ValueError('Target shape must have 2 elements (height, width), not {}'.format(shape))
        return dict(images=images, shape=shape)
