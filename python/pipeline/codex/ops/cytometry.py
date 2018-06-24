from codex.ops import op as codex_op
from codex.cytometry import cytometer
import os
import os.path as osp
import codex
import logging
import numpy as np
logger = logging.getLogger(__name__)


def get_model_path():
    return os.getenv(
        codex.ENV_CYTOMETRY_2D_MODEL_PATH,
        osp.join(os.environ['CODEX_DATA_DIR'], 'modeling', 'cytopy', 'models', 'nuclei', 'v0.3', 'nuclei_model.h5')
    )


def set_keras_session(op):
    import keras.backend.tensorflow_backend as KTF
    import tensorflow as tf
    tf_config = codex_op.get_tf_config(op)
    KTF.set_session(tf.Session(config=tf_config))


def close_keras_session():
    import keras.backend.tensorflow_backend as KTF
    KTF.get_session().close()


class Cytometry(codex_op.CodexOp):

    def __init__(self, config, mode='2D'):
        super(Cytometry, self).__init__(config)
        self.mode = mode
        if self.mode != '2D':
            raise ValueError('Cytometry mode should be one of ["2D"] not {}'.format(self.mode))
        self.nuc_channel_coords, self.mem_channel_coords, self.cytometry_params = config.cytometry_reference
        self.input_shape = (config.tile_height, config.tile_width, 1)
        self.cytometer = None

    def initialize(self):
        # Set the Keras session to have the same TF configuration as other operations
        set_keras_session(self)

        # Load the cytometry model from path to keras model weights
        model_path = get_model_path()
        logger.debug('Initializing cytometry model from path "{}" (input shape = {})'.format(model_path, self.input_shape))
        self.cytometer = cytometer.Cytometer2D(self.input_shape, model_path).initialize()
        return self

    def shutdown(self):
        close_keras_session()
        return self

    def _run_2d(self, tile):
        nuc_cycle = self.nuc_channel_coords[0]
        nuc_channel = self.nuc_channel_coords[1]

        # Tile should have shape (cycles, z, channel, height, width)
        img_nuc = tile[nuc_cycle, :, nuc_channel]
        img_seg, _, _ = self.cytometer.segment(img_nuc, **(self.cytometry_params or {}))

        # Ensure segmentation image is of integer type and >= 0
        assert np.issubdtype(img_seg.dtype, np.integer), \
            'Expecting int segmentation image but got {}'.format(img_seg.dtype)
        assert img_seg.min() >= 0, \
            'Labeled segmentation image contains label < 0 (shape = {}, dtype = {})'\
            .format(img_seg.shape, img_seg.dtype)

        # Check to make sure we did not end up with more than the maximum possible number of labeled cells
        if img_seg.max() > np.iinfo(np.uint16).max:
            raise ValueError(
                'Segmentation resulted in {} cells, a number which is both suspiciously high '
                'and too large to store as the assumed 16-bit format'.format(img_seg.max()))

        stats = self.cytometer.quantify(tile, img_seg, channel_names=self.config.channel_names)

        # Create overlay image of nucleus channel and boundaries and convert to 5D
        # shape to conform with usual tile convention
        img_boundary = _overlay_boundaries(img_nuc, img_seg)
        assert img_boundary.ndim == 3
        img_boundary = img_boundary[np.newaxis, :, np.newaxis, :, :]

        # Convert segmentation image to uint16 (from int32) and also conform to 5D standard
        assert img_seg.ndim == 3
        img_seg = img_seg[np.newaxis, :, np.newaxis, :, :].astype(np.uint16)

        return img_seg, img_boundary, stats

    def _run(self, tile, **kwargs):
        return self._run_2d(tile)


def _overlay_boundaries(img, img_seg):
    """Overlay labeled image into another

    Args:
        img: Any image
        img_seg: A labeled segmentation image (of some integer type) that must be equal in trailing dimensions
            to the `img` array
    """
    from skimage import exposure, segmentation

    # Copy and rescale to uint8 with one entry left at top of range
    res = exposure.rescale_intensity(img.copy(), out_range=(0, 254)).astype(np.uint8)

    # Find boundaries (per z-plane since find_boundaries is buggy in 3D)
    img_border = np.stack([
        segmentation.find_boundaries(img_seg[i], mode='inner', background=img_seg.min())
        for i in range(img_seg.shape[0])
    ], axis=0)

    mask = img_border > 0
    mask = np.broadcast_to(mask, img.shape)
    res[mask] = 255
    return res