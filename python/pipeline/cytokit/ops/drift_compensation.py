from cytokit.ops.op import TensorFlowOp, CytokitOp
from cytokit.utils import np_utils, tf_utils
import tensorflow as tf
import numpy as np
from flowdec import fft_utils_tf
import logging
logger = logging.getLogger(__name__)


def coord_max(t):
    """Multidimensional argmax

    Returns coordinates of largest element in tensors of any shape

    Args:
        t: Tensor to find location of max element within
    Returns:
        0 or 1-D Tensor of length t.ndims containing coordinates
    """

    # Run 1-D argmax on flattened array
    idx = tf.argmax(tf.reshape(t, [-1]), output_type=tf.int32)

    # Use "unravel_index" which takes any number of indexes within a flattened
    # array and returns the location prior to reshaping
    return tf.squeeze(tf.unravel_index(idx, tf.shape(t)))


class TranslationCalculator(TensorFlowOp):

    def __init__(self, n_dims):
        super().__init__()
        self.n_dims = n_dims

    def _build_graph(self):
        reference_image = tf.placeholder(tf.complex64, shape=[None] * self.n_dims, name='reference_image')
        offset_image = tf.placeholder(tf.complex64, shape=[None] * self.n_dims, name='offset_image')
        shape = tf.shape(reference_image)
        fft_fwd, fft_rev = fft_utils_tf.get_fft_tf_fns(self.n_dims, real_domain_only=False)

        # See https://github.com/scikit-image/scikit-image/blob/master/skimage/feature/register_translation.py#L186
        # for reference implementation
        img_prod = tf.multiply(fft_fwd(reference_image), tf.conj(fft_fwd(offset_image)))
        img_cc = fft_rev(img_prod)
        idx = coord_max(tf.abs(tf.real(img_cc)))
        center = tf.cast(tf.floor(shape / 2), tf.int32)
        translation = tf.where(idx > center, idx - shape, idx)

        inputs = dict(reference_image=reference_image, offset_image=offset_image)
        outputs = dict(translation=translation, center=center, offset=idx)
        return inputs, outputs

    def args(self, reference_image, offset_image):
        """Get arguments for translation calculation

        Args:
            reference_image: Reference image with rank equal to `n_dims` specified on instantiation
            offset_image: Image to calculation translation from reference image for; must have same shape
        """
        return dict(reference_image=reference_image, offset_image=offset_image)


class TranslationApplier(TensorFlowOp):

    def __init__(self, n_dims):
        super().__init__()
        self.n_dims = n_dims
        if self.n_dims not in [2, 3, 4]:
            raise ValueError('Number of dimensions must be 2, 3 or 4 (given {})'.format(self.n_dims))

    def _build_graph(self):
        image = tf.placeholder(self.primary_dtype, shape=[None]*self.n_dims, name='image')
        translation = tf.placeholder(tf.int32, shape=[2], name='translation')

        # tf.contrib.image.translate is given as dx, dy, which is opposite order of usual interpretation in scipy
        # and awkward in 3D so assume all translations are given the usual way (dy, dx) and reverse them here
        shift = tf.reverse(translation, axis=[0])

        # tf.contrib.image.translate also expects floating point shifts so force the cast here since
        # this operation only works on integer shifts
        shift = tf.cast(shift, tf.float32)

        # Apply translation and verify that shape does not change
        result = tf.contrib.image.translate(image, shift)
        with tf.control_dependencies([tf.assert_equal(tf.shape(image), tf.shape(result))]):
                result = tf.identity(result)

        inputs = dict(image=image, translation=translation)
        outputs = dict(result=result)
        return inputs, outputs

    def args(self, image, translation):
        """Get arguments for translation application

        Args:
            image: An array matching any one of the following shapes (see tf.contrib.image.translate docs):
                - (num_images, num_rows, num_columns, num_channels) (NHWC)
                - (num_rows, num_columns, num_channels) (HWC)
                - (num_rows, num_columns) (HW)
            translation: A 2 item array as [dy, dx]
        """
        return dict(image=image, translation=translation)


class CytokitDriftCompensator(CytokitOp):

    def __init__(self, config):
        super(CytokitDriftCompensator, self).__init__(config)
        self.calculator = None
        self.applier = None

        params = config.drift_compensation_params
        self.drift_cycle, self.drift_channel = config.get_channel_coordinates(params['channel'])

    def initialize(self):
        self.calculator = TranslationCalculator(n_dims=3).initialize()
        self.applier = TranslationApplier(n_dims=4).initialize()
        return self

    def _get_translations(self, tile):
        ncyc, nch = self.config.n_cycles, self.config.n_channels_per_cycle

        # Extract reference from tile that should have shape (cycles, z, channel, height, width)
        reference_image = tile[self.drift_cycle, :, self.drift_channel, :, :]

        # Determine set of cycles to be aligned (everything except reference)
        target_cycles = list(set(range(ncyc)) - {self.drift_cycle})

        # Compute translations that need to be applied
        translations = []
        for icyc in target_cycles:
            logger.debug(
                'Calculating drift translation for reference cycle {}, comparison cycle {}'
                .format(self.drift_cycle, icyc)
            )
            offset_image = tile[icyc, :, self.drift_channel, :, :]
            res = self.calculator.run(reference_image, offset_image)
            # Extract dy, dx translation from translation as dz, dy, dx
            translations.append(res['translation'][1:])

        # Add monitor records containing the translation to be applied to each non-reference cycle
        # Note: translations are specified as [dy, dx]
        for i, icyc in enumerate(target_cycles):
            self.record({'target_cycle': icyc, 'translation': translations[i]})

        return translations

    def _apply_translations(self, tile, translations):
        ncyc, nch = self.config.n_cycles, self.config.n_channels_per_cycle

        translation_iter = iter(translations)
        img_cyc = []
        for icyc in range(ncyc):
            # Assign translation if processing non-reference cycle (otherwise use noop translation)
            translation = None if icyc == self.drift_cycle else next(translation_iter)
            logger.debug('Applying translation {} to cycle {}'.format(translation, icyc))

            # Apply translation, if not currently on the reference cycle
            tile_subset = tile[icyc]
            if translation is None:
                img = tile_subset.astype(np.float32)
            else:
                # Transform tile from (nz, nch, ny, nx) as (nz, ny, nx, nch) to comply
                # with TensorFlow image translation function
                img = np.moveaxis(tile_subset, 1, -1)

                # Run translation and convert axes back to original order
                img = self.applier.run(img, translation)['result']
                img = np.moveaxis(img, -1, 1)

                if tile_subset.shape != img.shape:
                    raise AssertionError(
                        'Image after drift compensation application has shape {} instead of expected shape {}'
                        .format(img.shape, tile_subset.shape)
                    )
            img_cyc.append(img)

        # Re-stack along cycle axis
        return np.stack(img_cyc, 0)

    def _run(self, tile, **kwargs):
        # Compute drift between reference cycle and all others
        logger.info('Calculating drift translations')
        translations = self._get_translations(tile)

        # If there is only one cycle, then drift compensation is not possible so return the given tile as-is
        if self.config.n_cycles < 2:
            logger.debug('Experiment has less than 2 cycles so drift compensation operation will result in no changes')
            return tile

        # Apply translations and convert back to original type from float32
        logger.info('Applying drift translations')
        res = self._apply_translations(tile, translations)
        res = np_utils.arr_to_uint(res, tile.dtype)

        # Validate shape of result
        if res.shape != tile.shape:
            raise AssertionError(
                'Tile shape after drift compensation ({}) does not match input shape ({})'
                .format(res.shape, tile.shape)
            )
        return res






