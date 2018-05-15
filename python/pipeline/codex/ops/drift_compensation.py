from codex.ops.op import TensorFlowOp, CodexOp
from codex.utils import np_utils, tf_utils
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
        super(TranslationCalculator, self).__init__()
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
        outputs = dict(translation=translation, center=center, offset=idx, xcor=img_cc)
        return inputs, outputs

    def args(self, reference_image, offset_image):
        return dict(reference_image=reference_image, offset_image=offset_image)


class TranslationApplier(TensorFlowOp):

    def __init__(self, n_dims):
        super(TranslationApplier, self).__init__()
        self.n_dims = n_dims
        if self.n_dims not in [2, 3]:
            raise ValueError('Number of dimensions must be 2 or 3 (given {})'.format(self.n_dims))

    def _build_graph(self):
        image = tf.placeholder(self.primary_dtype, shape=[None]*self.n_dims, name='image')
        translation = tf.placeholder(tf.int32, shape=[self.n_dims], name='translation')

        # tf.contrib.image.translate is given as dx, dy, which is opposite order of usual interpretation in scipy
        # and awkward in 3D so assume all translations are given the usual way and reverse them here
        shift = tf.reverse(translation, axis=[0])

        # tf.contrib.image.translate also expects floating point shifts so force the cast here since
        # this operation only works on integer shifts
        shift = tf.cast(shift, tf.float32)

        # Docs on inputs to tf.contrib.image.translate:
        # images: A tensor of shape (num_images, num_rows, num_columns, num_channels) (NHWC),
        # (num_rows, num_columns, num_channels) (HWC), or (num_rows, num_columns) (HW)
        if self.n_dims == 2:
            result = tf.contrib.image.translate(image, shift)
        elif self.n_dims == 3:
            shp = tf.shape(image)
            n_z = shp[0]

            # tf.contrib.image.translate only does x/y translation so add in z-translation by filling
            # empty planes with zeros and determining how many need to be added on top or bottom
            n_lo, n_hi = tf.maximum(translation[0], 0), tf.abs(tf.minimum(translation[0], 0))
            i_lo, i_hi = n_hi, (n_z - n_lo)
            stack = [
                tf.zeros([n_lo, shp[1], shp[2]], dtype=image.dtype),
                # Expand dims to use (num_images, num_rows, num_columns, num_channels)
                # argument (see translate function docs)
                tf.contrib.image.translate(tf.expand_dims(image, -1), shift)[i_lo:i_hi, :, :, 0],
                tf.zeros([n_hi, shp[1], shp[2]], dtype=image.dtype)
            ]
            result = tf.concat(stack, axis=0)
        else:
            raise ValueError('Number of dimensions {} not supported'.format(self.n_dims))

        inputs = dict(image=image, translation=translation)
        outputs = dict(result=result)
        return inputs, outputs

    def args(self, image, translation):
        return dict(image=image, translation=translation)


class CodexDriftCompensator(CodexOp):

    def __init__(self, config):
        super(CodexDriftCompensator, self).__init__(config)
        self.calculator = None
        self.applier = None

    def initialize(self):
        self.calculator = TranslationCalculator(n_dims=3).initialize()
        self.applier = TranslationApplier(n_dims=3).initialize()
        return self

    def _run(self, tile, **kwargs):
        # Tile should have shape (cycles, z, channel, height, width)
        ncyc, nch = self.config.n_cycles, self.config.n_channels_per_cycle

        # Determine cycle + channel to be used as reference for compensation
        drift_cycle, drift_channel = self.config.drift_compensation_reference
        reference_image = tile[drift_cycle, :, drift_channel, :, :]

        # Determine set of cycles to be aligned (everything except reference)
        target_cycles = list(set(range(ncyc)) - set([drift_cycle]))

        # Compute translations that need to be applied
        def translation_calculations():
            for icyc in target_cycles:
                logger.debug('Calculating drift translation for reference cycle {}, comparison cycle {}'
                             .format(drift_cycle, icyc))
                offset_image = tile[icyc, :, drift_channel, :, :]
                yield self.calculator.args(reference_image, offset_image)

        logger.info('Calculating drift translations')
        translations = self.calculator.flow(translation_calculations())

        # Add monitor records containing the translation to be applied to each non-reference cycle
        # Note: translations are specified as [dz, dy, dx]
        for i, icyc in enumerate(target_cycles):
            self.record({'target_cycle': icyc, 'translation': translations[i]['translation']})
        translations = iter(translations)

        # Apply all computed translations and reassemble result
        noop_translation = np.zeros(3)

        def translation_applications():
            for icyc in range(ncyc):
                translation = noop_translation if icyc == drift_cycle else next(translations)['translation']
                logger.debug('Applying translation {} to cycle {}'.format(translation, icyc))
                for ich in range(nch):
                    img = tile[icyc, :, ich, :, :]
                    yield self.applier.args(img, translation)

        logger.info('Applying drift translations')
        applications = iter(self.applier.flow(translation_applications()))

        img_cyc = []
        for icyc in range(ncyc):
            img_ch = []
            for ich in range(nch):
                img_ch.append(next(applications)['result'])
            # Re-stack along channel axis
            img_cyc.append(np.stack(img_ch, 1))

        # Re-stack along cycle axis and convert back to original type from float32
        return np_utils.arr_to_uint(np.stack(img_cyc, 0), tile.dtype)






