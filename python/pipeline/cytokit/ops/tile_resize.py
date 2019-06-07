from cytokit.ops.op import TensorFlowOp, CytokitOp
from skimage import transform
import numpy as np
import logging
logger = logging.getLogger(__name__)


class CytokitTileResize(CytokitOp):

    def __init__(self, config):
        super().__init__(config)
        self.resizer = None
        params = config.tile_resize_params
        self.factors = params.get('factors', [1, 1, 1])
        self.implementation = params.get('implementation', 'skimage')

        zrate = config.microscope_params.res_axial_nm / config.microscope_params.res_lateral_nm
        # Get anisotropic sampling factors (often like [.5, 1, 1] for larger z step)
        self.rates = 1. / np.array([zrate, 1., 1.])

    def initialize(self):
        if self.implementation == 'skimage':
            self.resizer = SkimageResizer()
        elif self.implementation == 'tensorflow':
            self.resizer = TensorflowResizer().initialize()
        else:
            raise ValueError(
                'Resizer implementation "{}" not valid (must be "skimage" or "tensorflow")'
                .format(self.implementation)
            )
        return self

    @classmethod
    def resize(cls, tile, factors, shape, rates, resizer):
        ncyc, nch = tile.shape[0], tile.shape[2]

        img_cyc = []
        for icyc in range(ncyc):
            img_ch = []
            for ich in range(nch):
                img = tile[icyc, :, ich]
                if img.ndim != 3:
                    raise ValueError('Expecting 3D image but got shape {}'.format(img.shape))
                img = resizer.resize(img, factors, shape, rates)
                img_ch.append(img)
            # Stack from 3D to (z, ch, h, w)
            img_cyc.append(np.stack(img_ch, axis=1))
        # Stack from 4D to (cyc, z, ch, h, w)
        res = np.stack(img_cyc, axis=0)
        assert res.dtype == tile.dtype, 'Result type {} != expected type {}'.format(res.dtype, tile.dtype)
        return res

    def _run(self, tile, **kwargs):
        # NOOP if rescaling factors imply no change
        if all([np.isclose(1, f) for f in self.factors]):
            logger.debug('Resizing factors all close to 1 so no images will be resized')
            return tile

        # Calculate expected shape and match against resulting shape after resize
        dims = [1, 3, 4]  # Spatial dimensions (z, y, x)
        old_shape = tuple([tile.shape[i] for i in dims])
        new_shape = tuple([max(1, round(v)) for v in np.asarray(self.factors) * np.asarray(old_shape)])

        logger.info(
            'Running tile resize with rescaling factors {} (old shape = {}, new shape = {})'
            .format(self.factors, old_shape, new_shape)
        )
        tile = CytokitTileResize.resize(tile, self.factors, new_shape, self.rates, self.resizer)
        assert tuple([tile.shape[i] for i in dims]) == new_shape
        return tile


class SkimageResizer(object):

    def resize(self, img, factors, shape, rates):
        # Set factors per skimage resize defaults and then scale further by z vs xy sampling
        factors = 1. / np.asarray(factors)
        sigmas = np.maximum(0, (factors - 1) / 2) * rates
        return transform.resize(
            img, shape,
            preserve_range=True, order=1, clip=True, mode='constant',
            anti_aliasing=True, anti_aliasing_sigma=sigmas
        ).astype(img.dtype)


class TensorflowResizer(TensorFlowOp):

    def __init__(self, method='BILINEAR'):
        """Create resizing instance

        Args:
            method: Any option available via tf.image.ResizeMethod including 'BILINEAR', 'AREA', 'BICUBIC' and
                'NEAREST_NEIGHBOR'
        """
        super().__init__()
        # Note that this must be specified outside of the graph since internally an integer equality comparison
        # is done to resolve the method (which doesn't work if method is a scalar tensor)
        import tensorflow as tf
        self.method = getattr(tf.image.ResizeMethod, method) if isinstance(method, str) else method

    def _build_graph(self):
        # Delay tensorflow import until necessary for CPU
        import tensorflow as tf

        # Assume image is 3D (z, y, x)
        image = tf.placeholder(tf.float32, shape=[None] * 3, name='images')
        # Assume shape is target shape (z, y, x)
        shape = tf.placeholder(tf.int32, shape=[3], name='shape')

        # Add unit channel dimension on end (z, y, x, ch)
        img = tf.expand_dims(image, 3)

        # Treat z as batch dimension and resize YX images
        img = tf.image.resize_images(img, tf.stack([shape[1], shape[2]]), method=self.method, align_corners=True)

        # Transpose from (z, y, x, 1) -> (y, z, x, 1)
        img = tf.transpose(img, (1, 0, 2, 3))

        # Treat y as batch dimension and resize ZX images
        img = tf.image.resize_images(img, tf.stack([shape[0], shape[2]]), method=self.method, align_corners=True)

        # Transpose from (y, z, x, 1) -> (z, y, x, 1)
        img = tf.transpose(img, (1, 0, 2, 3))

        # Remove unit-length channel dimension
        img = tf.squeeze(img, axis=3)

        inputs = dict(image=image, shape=shape)
        outputs = dict(result=img)
        return inputs, outputs

    def resize(self, image, factors, shape, rates):
        return super(TensorflowResizer, self).run(image=image, shape=shape)['result'].astype(image.dtype)

    def args(self, image, shape):
        """Get arguments for resize

        Args:
            image: An array with shape (z, y, x)
            shape: Target shape for resize as three item sequence (z, y, x)
        """
        if image.ndim != 3:
            raise ValueError('Images must have shape (z, y, x) not {}'.format(image.shape))
        if len(shape) != 3:
            raise ValueError('Target shape must have 3 elements (z, y, x), not {}'.format(shape))
        return dict(image=image, shape=shape)
