from cytokit.ops import op as cytokit_op
from cytokit.ops import tile_crop
from cytokit.cytometry import cytometer
from cytokit import io as cytokit_io
from cytokit import math as cytokit_math
from cytokit import data as cytokit_data
from cytokit import config as cytokit_config
import os
import re
import os.path as osp
import cytokit
import logging
import numpy as np
logger = logging.getLogger(__name__)

OBJECT_CYCLE = 0

CHANNEL_COORDINATES = {
    # Map channel names to (cycle, channel) coordinates
    'cell_mask': (OBJECT_CYCLE, cytometer.CytometerBase.CELL_MASK_CHANNEL),
    'cell_boundary': (OBJECT_CYCLE, cytometer.CytometerBase.CELL_BOUNDARY_CHANNEL),
    'nucleus_mask': (OBJECT_CYCLE, cytometer.CytometerBase.NUCLEUS_MASK_CHANNEL),
    'nucleus_boundary': (OBJECT_CYCLE, cytometer.CytometerBase.NUCLEUS_BOUNDARY_CHANNEL)
}

DEFAULT_CYTOMETER_TYPE = cytometer.Cytometer2D


def get_channel_coordinates(channel):
    channel = channel.lower().strip()

    if channel in CHANNEL_COORDINATES:
        return channel, CHANNEL_COORDINATES[channel]

    # If not a pre-defined coordinate, assume coordinate is either invalid or specified directly
    # Example: "mychannel(0,1)" -> name "mychannel", cycle 0, channel 1 in segmentation result
    coords = re.search(r'^([a-zA-Z0-9_.\-:;]+)\(([0-9 ]+),([0-9 ]+)\)$', channel)
    if not coords or len(coords.groups()) != 3:
        raise ValueError(
            'Cytometry channel specification "{}" is not valid.  Must be one of {} or name+cycle+channel '
            'coordinates as `[name]([cycle],[channel])` (where cycle and channel are 0-based integer indices'
            'and name is the name associated with the channel)'
            .format(channel, list(CHANNEL_COORDINATES.keys()))
        )
    groups = coords.groups()
    return groups[0], tuple(map(int, groups[1:]))


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

    def __init__(self, config, z_plane='best', cytometer_type='2D', crop=False, target_shape=None,
                 segmentation_params=None, quantification_params=None):
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

        self.cytometer_type = params.get('type', cytometer_type)
        self.cytometer = None
        self.input_shape = (config.tile_height, config.tile_width, 1)

        self.cropper = tile_crop.CytokitTileCrop(config) if params.get('crop', crop) else None

        _validate_z_plane(self.z_plane)

    def initialize(self):
        # Construct custom cytometer if a definition for one was provided
        cytometer_class = DEFAULT_CYTOMETER_TYPE
        if hasattr(self.cytometer_type, 'keys'):
            cytometer_class = cytokit_config.get_implementation_class(self.cytometer_type)
            # Skip default initialization if not an override of default implementation
            if not issubclass(cytometer_class, DEFAULT_CYTOMETER_TYPE):
                self.cytometer = cytokit_config.get_implementation_instance(
                    self.cytometer_type, cytometer_class, config=self.config
                )
                self.cytometer.initialize()
                return self
        # Otherwise verify that type for default cytometer is valid (currently only 2D supported)
        else:
            if self.cytometer_type not in ['2D']:
                raise ValueError('Cytometer type "{}" is not valid'.format(self.cytometer_type))

        # Set the Keras session to have the same TF configuration as other operations
        set_keras_session(self)

        # Use explicit override for weights for segmentation model (otherwise a default
        # will be used based on cached weights files)
        weights_path = os.getenv(cytokit.ENV_CYTOMETRY_2D_MODEL_PATH)
        logger.debug(
            'Initializing cytometry model for input shape = %s (target shape = %s)',
            self.input_shape, self.target_shape
        )

        self.cytometer = cytometer_class(
                input_shape=self.input_shape, target_shape=self.target_shape,
                weights_path=weights_path, config=self.config
        )
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
            raise ValueError(
                'Best focus plane must be specified when running cytometry for best z planes '
                '(set `run_best_focus: true` in processor.args of config)'
            )

        if z_plane == 'best':
            z_plane = best_focus_z_plane

        assert z_plane is not None, 'Z plane must be set'
        return z_plane

    def _run(self, tile, z_plane=None, best_focus_z_plane=None, tile_indices=None):
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
        img_seg = self.cytometer.segment(
            img_nuc, img_memb=img_memb, z_plane=z_plane, tile=tile,
            tile_indices=tile_indices, **self.segmentation_params
        )

        # Validate results are 4D (ZCHW)
        assert img_seg.ndim == 4, 'Expecting 4D segmentation image but shape is {}'.format(img_seg.shape)
        assert img_seg.shape[0] == 1 or img_seg.shape[0] == tile.shape[1], \
            'Expecting segmentation image to have either one z-plane or as many as input tile ({}) but got {}'\
            .format(tile.shape[1], img_seg.shape[0])

        # Segmentation results can have one z plane or as many as tile so depending on
        # the z plane used to calculate segmentation, pad the result to required shape (if necessary)
        if img_seg.shape[0] != tile.shape[1]:
            shape = list(img_seg.shape)
            shape[0] = tile.shape[1]
            img_seg_tmp = np.zeros(shape, dtype=img_seg.dtype)
            if z_plane == 'all':
                # Repeat result z times
                img_seg_tmp[:] = img_seg
            else:
                # Set single plane to result and leave all others as 0
                img_seg_tmp[z_plane] = img_seg
            img_seg = img_seg_tmp

        # Ensure segmentation image is of integer type and >= 0
        assert np.issubdtype(img_seg.dtype, np.integer), \
            'Expecting int segmentation image but got {}'.format(img_seg.dtype)
        assert img_seg.min() >= 0, \
            'Labeled segmentation image contains label < 0 (shape = {}, dtype = {})'\
            .format(img_seg.shape, img_seg.dtype)

        # Check to make sure we did not end up with more than the maximum possible number of labeled objects
        if img_seg.max() > np.iinfo(np.uint16).max:
            raise ValueError(
                'Segmentation resulted in {} cells, a number which is both suspiciously high '
                'and too large to store as the assumed 16-bit format'.format(img_seg.max()))

        # Lastly, if configured to crop input tiles and results, apply this cropping
        # after segmentation and before quantification
        if self.cropper is not None:
            tile = self.cropper.run(tile)
            img_seg = self.cropper.run(img_seg)

        # Run cell cytometry calculations (results given as data frame)
        logger.debug('Running segmentation quantification')
        stats = self.cytometer.quantify(
            tile, img_seg, z_plane=z_plane,
            tile_indices=tile_indices, **self.quantification_params
        )

        # Add any statistics or transformations that require the experimental context
        stats = self.cytometer.augment(stats)

        # Convert to 5D tile format (single cycle) and 16-bit
        img_seg = img_seg.astype(np.uint16)[np.newaxis]
        assert img_seg.ndim == 5, 'Expecting 5D image but shape is {}'.format(img_seg.shape)

        return tile, (img_seg, stats)

    def save(self, tile_indices, output_dir, data, compress=6):
        region_index, tile_index, tx, ty = tile_indices
        img_label, stats = data

        # Save label volumes if present (use compression as these are often highly redundant)
        label_tile_path = None
        if img_label is not None:
            label_tile_path = cytokit_io.get_cytometry_image_path(region_index, tx, ty)
            cytokit_io.save_tile(osp.join(output_dir, label_tile_path), img_label,
                                 config=self.config, compress=compress)

        # Save statistics if present
        stats_path = None
        if stats is not None:
            # Append useful metadata to cytometry stats (align these names to those used in config.TileDims)
            # and export as csv
            stats.insert(0, 'tile_y', ty)
            stats.insert(0, 'tile_x', tx)
            stats.insert(0, 'tile_index', tile_index)
            stats.insert(0, 'region_index', region_index)
            stats_path = cytokit_io.get_cytometry_stats_path(region_index, tx, ty)
            cytokit_io.save_csv(osp.join(output_dir, stats_path), stats, index=False)

        return label_tile_path, stats_path

