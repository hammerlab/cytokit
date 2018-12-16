from cytokit.ops import op as cytokit_op
from cytokit.cytometry import cytometer
from cytokit import io as cytokit_io
from cytokit import math as cytokit_math
from cytokit import data as cytokit_data
import os
import os.path as osp
import cytokit
import logging
import numpy as np
logger = logging.getLogger(__name__)

MASK_CYCLE = 0
BOUNDARY_CYCLE = 1

CHANNEL_COORDINATES = {
    # Map channel names to (cycle, channel) coordinates
    'cell_mask': (MASK_CYCLE, cytometer.CELL_CHANNEL),
    'cell_boundary': (BOUNDARY_CYCLE, cytometer.CELL_CHANNEL),
    'nucleus_mask': (MASK_CYCLE, cytometer.NUCLEUS_CHANNEL),
    'nucleus_boundary': (BOUNDARY_CYCLE, cytometer.NUCLEUS_CHANNEL),
}


def get_channel_coordinates(channel):
    channel = channel.lower().strip()
    if channel not in CHANNEL_COORDINATES:
        raise ValueError(
            'Cytometry channel "{}" is not valid.  Must be one of {}'
            .format(channel, list(CHANNEL_COORDINATES.keys()))
        )
    return CHANNEL_COORDINATES[channel]


def set_keras_session(op):
    import keras.backend.tensorflow_backend as KTF
    import tensorflow as tf
    tf_config = cytokit_op.get_tf_config(op)
    KTF.set_session(tf.Session(config=tf_config))


def close_keras_session():
    import keras.backend.tensorflow_backend as KTF
    KTF.get_session().close()


def _validate_z_plane(z_plane):
    if z_plane not in ['best', 'all'] and not isinstance(z_plane, int):
        raise ValueError(
            'Z plane for cytometry must be either "all", "best", or a 0-based integer index (given = {})'
            .format(z_plane)
        )


def get_op(config):
    mode = config.cytometry_params.get('mode', '2D')
    if mode != '2D':
        raise ValueError('Cytometry mode should be one of ["2D"] not {}'.format(mode))
    return Cytometry2D(config)


class Cytometry2D(cytokit_op.CytokitOp):

    def __init__(self, config, z_plane='best', target_shape=None, segmentation_params=None, quantification_params=None):
        super().__init__(config)

        params = config.cytometry_params
        self.z_plane = params.get('z_plane', z_plane)
        self.target_shape = params.get('target_shape', target_shape)

        self.segmentation_params = params.get('segmentation_params', segmentation_params or {})

        self.quantification_params = params.get('quantification_params', quantification_params or {})
        if 'channel_names' not in self.quantification_params:
            self.quantification_params['channel_names'] = config.channel_names

        self.nuc_channel_coords = config.get_channel_coordinates(params['nuclei_channel_name'])
        self.mem_channel_coords = None if 'membrane_channel_name' not in params else \
            config.get_channel_coordinates(params['membrane_channel_name'])

        self.input_shape = (config.tile_height, config.tile_width, 1)
        self.cytometer = None

        _validate_z_plane(self.z_plane)

    def initialize(self):
        # Set the Keras session to have the same TF configuration as other operations
        set_keras_session(self)

        # Use explicit override for weights for segmentation model (otherwise a default
        # will be used based on cached weights files)
        weights_path = os.getenv(cytokit.ENV_CYTOMETRY_2D_MODEL_PATH)
        logger.debug(
            'Initializing cytometry model for input shape = %s (target shape = %s)',
            self.input_shape, self.target_shape
        )
        self.cytometer = cytometer.Cytometer2D(
            self.input_shape, target_shape=self.target_shape, weights_path=weights_path)
        self.cytometer.initialize()
        return self

    def shutdown(self):
        close_keras_session()
        return self

    def _resolve_z_plane(self, z_plane, best_focus_z_plane):
        if z_plane is None:
            z_plane = self.z_plane
        _validate_z_plane(z_plane)

        if z_plane == 'best' and best_focus_z_plane is None:
            raise ValueError('Best focus plane must be specified when running cytometry for best z planes')

        if z_plane == 'best':
            z_plane = best_focus_z_plane

        assert z_plane is not None, 'Z plane must be set'
        return z_plane

    def _run(self, tile, z_plane=None, best_focus_z_plane=None):
        z_plane = self._resolve_z_plane(z_plane, best_focus_z_plane)
        z_slice = slice(None) if z_plane == 'all' else slice(z_plane, z_plane + 1)

        # Determine coordinates of nucleus channel
        nuc_cycle = self.nuc_channel_coords[0]
        nuc_channel = self.nuc_channel_coords[1]

        # Tile shape = (cycles, z, channel, height, width)
        img_nuc = tile[nuc_cycle, z_slice, nuc_channel]

        # If configured to do so, also extract cell membrane channel to make
        # more precise cell segmentations
        img_memb = None
        if self.mem_channel_coords is not None:
            memb_cycle = self.mem_channel_coords[0]
            memb_channel = self.mem_channel_coords[1]
            img_memb = tile[memb_cycle, z_slice, memb_channel]

        # Fetch segmentation volume as ZCHW where C = 2 and C1 = cell and C2 = nucleus
        img_seg = self.cytometer.segment(img_nuc, img_memb=img_memb, **self.segmentation_params)[0]

        # If using a specific z-plane, conform segmented volume to typical tile
        # shape by adding empty z-planes
        if z_plane != 'all':
            shape = list(img_seg.shape)
            shape[0] = tile.shape[1]
            img_seg_tmp = np.zeros(shape, dtype=img_seg.dtype)
            img_seg_tmp[z_plane] = img_seg
            img_seg = img_seg_tmp

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

        # Run cell cytometry calculations (results given as data frame)
        stats = self.cytometer.quantify(tile, img_seg, **self.quantification_params)

        # Convert size measurements to more meaningful scales and add diameter
        resolution_um = self.config.microscope_params.res_lateral_nm / 1000.
        for c in ['cell', 'nucleus']:
            stats[c + '_size'] = cytokit_math.pixel_area_to_squared_um(stats[c + '_size'].values, resolution_um)
            stats[c + '_diameter'] = stats[c + '_diameter'] * resolution_um

        # Create overlay image of nucleus channel and boundaries and convert to 5D
        # shape to conform with usual tile convention
        img_boundary = np.stack([
            _find_boundaries(img_seg[:, i], as_binary=False)
            for i in range(img_seg.shape[1])
        ], axis=1)
        assert img_boundary.ndim == 4, 'Expecting 4D image, got shape {}'.format(img_boundary.shape)

        # Stack labeled volumes to 5D tiles and convert to uint16
        # * Note that this ordering should align to MASK_CYCLE and BOUNDARY_CYCLE constants
        img_label = np.stack([img_seg, img_boundary], axis=0).astype(np.uint16)

        return img_label, stats

    def save(self, tile_indices, output_dir, data):
        region_index, tile_index, tx, ty = tile_indices
        img_label, stats = data

        # Save label volumes if present
        label_tile_path = None
        if img_label is not None:
            label_tile_path = cytokit_io.get_cytometry_image_path(region_index, tx, ty)
            cytokit_io.save_tile(osp.join(output_dir, label_tile_path), img_label, config=self.config)

        # Append useful metadata to cytometry stats (align these names to those used in config.TileDims)
        # and export as csv
        stats.insert(0, 'tile_y', ty)
        stats.insert(0, 'tile_x', tx)
        stats.insert(0, 'tile_index', tile_index)
        stats.insert(0, 'region_index', region_index)
        stats_path = cytokit_io.get_cytometry_stats_path(region_index, tx, ty)
        cytokit_io.save_csv(osp.join(output_dir, stats_path), stats, index=False)
        return label_tile_path, stats_path


def _find_boundaries(img, as_binary=False):
    """Identify boundaries in labeled image volume

    Args:
        img: A labeled 3D volume with shape (z, h, w)
        as_binary: Flag indicating whether to return binary boundary image or labeled boundaries
    """
    from skimage import segmentation
    assert img.ndim == 3, 'Expecting 3D volume but got image with shape {}'.format(img.shape)

    # Find boundaries (per z-plane since find_boundaries is buggy in 3D)
    bg_value = img.min()
    res = np.stack([
        segmentation.find_boundaries(img[i], mode='inner', background=bg_value)
        for i in range(img.shape[0])
    ], axis=0)

    return res if as_binary else res * img

