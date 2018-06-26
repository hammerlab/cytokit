import json
import os
import codex
from codex import tiling
from os import path as osp
from collections import namedtuple


TILING_MODE_SNAKE = 'snake'


def _load_json_config(data_dir, filename):
    f = osp.join(data_dir, filename)
    if not osp.exists(f):
        raise ValueError('Required configuration file "{}" does not exist'.format(f))
    if filename.endswith('.yaml'):
        import yaml
        with open(f, 'r') as fd:
            return yaml.load(fd)
    elif filename.endswith('.json'):
        import json
        with open(f, 'r') as fd:
            return json.load(fd)
    else:
        raise ValueError('Configuration file "{}" has invalid extension (should be json or yaml)'.format(f))


def _load_experiment_config(data_dir, filename):
    return _load_json_config(data_dir, filename)


TileDims = namedtuple('TileDims', ['cycles', 'z', 'channels', 'height', 'width'])
TileIndices = namedtuple('TileIndices', ['region_index', 'tile_index', 'tile_x', 'tile_y'])


class Config(object):

    def register_environment(self):
        # Delegate this registration to the main codex module as it is the only
        # one that should ever set environment variables
        codex.register_environment(self.get_environment())

    @property
    def n_tiles_per_region(self):
        return self.region_width * self.region_height

    @property
    def tile_dims(self):
        """Get tile dimensions as (cycles, z, channels, height, width)"""
        return TileDims(self.n_cycles, self.n_z_planes, self.n_channels_per_cycle, self.tile_height, self.tile_width)

    def get_tile_indices(self):
        """Get coordinates (as indexes) of experiment tiles

        Returns:
            List of TileIndices tuples like (region_index, tile_index, tile_x, tile_y)
        """
        indices = []
        for ireg in self.region_indexes:
            for itile in range(self.n_tiles_per_region):
                tx, ty = self.get_tile_coordinates(itile)
                indices.append(TileIndices(ireg, itile, tx, ty))
        return indices

    def get_tile_coordinates(self, tile_index):
        """Get 0-based X and Y coordinates of a tile using the configured 'tiling_mode'

        Args:
            tile_index: 0-based tile index
        Returns:
            Integer tuple as (X, Y) where 0 <= X < region_width and 0 <= Y < region_height
        """
        if tile_index >= self.n_tiles_per_region:
            raise ValueError(
                'Cannot get coordinates for tile with 0-based index {} when only {} '
                'iles are expected (region width = {}, height = {})'
                    .format(tile_index, self.n_tiles_per_region, self.region_width, self.region_height)
            )
        tiler = tiling.get_tiling_by_name(self.tiling_mode)
        return tiler.coordinates_from_index(tile_index, w=self.region_width, h=self.region_height)

    def get_channel_coordinates(self, channel_name):
        """Get 0-based cycle and per-cycle-channel index coordinates for a channel name

        Args:
            channel_name: String name of channel
        Returns:
            (cycle, channel) - 0-based indexes for cycle and channel
        """
        if channel_name not in self.channel_names:
            raise ValueError('Channel "{}" is not configured channel list {}'.format(channel_name, self.channel_names))
        i = self.channel_names.index(channel_name)
        cycle_index = i // self.n_channels_per_cycle
        ch_index = i % self.n_channels_per_cycle
        return cycle_index, ch_index


class CodexConfigV10(Config):

    def __init__(self, conf):
        self._conf = conf

    def get_environment(self):
        res = {}
        env = self._conf.get('environment', {})
        acq = self._conf.get('acquisition', {})

        if 'index_symlinks' in env:
            res[codex.ENV_RAW_INDEX_SYMLINKS] = str(env['index_symlinks'])
        if 'path_formats' in env:
            res[codex.ENV_PATH_FORMATS] = str(env['path_formats'])
        if 'raw_file_type' in acq:
            res[codex.ENV_RAW_FILE_TYPE] = acq['raw_file_type']

        return res

    @property
    def channel_names(self):
        return self._conf['acquisition']['channel_names']

    @property
    def n_cycles(self):
        return self._conf['acquisition']['num_cycles']

    @property
    def n_z_planes(self):
        return self._conf['acquisition']['num_z_planes']

    @property
    def n_channels_per_cycle(self):
        return len(self._conf['acquisition']['per_cycle_channel_names'])

    @property
    def tile_width(self):
        return self._conf['acquisition']['tile_width']

    @property
    def tile_height(self):
        return self._conf['acquisition']['tile_height']

    @property
    def overlap_x(self):
        return self._conf['acquisition']['tile_overlap_x']

    @property
    def overlap_y(self):
        return self._conf['acquisition']['tile_overlap_y']

    @property
    def region_width(self):
        return self._conf['acquisition']['region_width']

    @property
    def region_height(self):
        return self._conf['acquisition']['region_height']

    @property
    def tiling_mode(self):
        return self._conf['acquisition']['tiling_mode']

    @property
    def region_indexes(self):
        """Get 0-based region index list"""
        return list(range(len(self._conf['acquisition']['region_names'])))

    @property
    def drift_compensation_reference(self):
        """Get reference image configured for drift compensation
        Returns:
            (cycle, channel) - 0-based indexes for cycle and channel
        """
        return self.get_channel_coordinates(self._conf['operation']['drift_compensation']['channel'])

    @property
    def best_focus_reference(self):
        """Get reference image configured for best focus plan selection
        Returns:
            (cycle, channel) - 0-based indexes for cycle and channel
        """
        return self.get_channel_coordinates(self._conf['operation']['best_focus']['channel'])

    def _op_params(self, op):
        return self._conf['operation'].get(op, {})

    @property
    def tile_generator_params(self):
        return self._op_params('tile_generator')

    @property
    def drift_compensation_params(self):
        return self._op_params('drift_compensation')

    @property
    def best_focus_params(self):
        return self._op_params('best_focus')

    @property
    def deconvolution_params(self):
        return self._op_params('deconvolution')

    @property
    def cytometry_params(self):
        return self._op_params('cytometry')

    @property
    def analysis_params(self):
        return self._conf['analysis']

    @property
    def _n_actual_channels(self):
        return len(self.channel_names)

    @property
    def _n_expected_channels(self):
        return self.n_cycles * self.n_channels_per_cycle

    @property
    def microscope_params(self):
        mag = self._conf['acquisition']['magnification']
        na = self._conf['acquisition']['numerical_aperture']
        res_axial_nm = self._conf['acquisition']['axial_resolution']
        res_lateral_nm = self._conf['acquisition']['lateral_resolution']
        objective_type = self._conf['acquisition']['objective_type']
        em_wavelength_nm = self._conf['acquisition']['emission_wavelengths']
        return mag, na, res_axial_nm, res_lateral_nm, objective_type, em_wavelength_nm

    def _validate(self):
        # Ensure that number of channel names equals expected number
        if self._n_actual_channels != self._n_expected_channels:
            raise ValueError(
                'Full list of channel names does not have length equal '
                'to num_cycles * n_channels_per_cycle; '
                'n expected channel names = {}, n actual channel names = {}'
                .format(self._n_expected_channels, self._n_actual_channels)
            )
        return self

    def __str__(self):
        return str(self._conf)

    __repr__ = __str__

    @staticmethod
    def load(data_dir, filename=None):
        """Load all CODEX related configuration files given a primary data directory"""
        conf = _load_experiment_config(data_dir, filename if filename else codex.get_config_default_filename())
        return CodexConfigV10(conf)._validate()


def load(path):
    if not osp.exists(path):
        raise ValueError('Configuration path "{}" does not exist'.format(path))

    # Split path into directory and filename
    if osp.isdir(path):
        dirname, filename = path, None
    else:
        dirname, filename = osp.dirname(path), osp.basename(path)

    # Load configuration based on version specified in environment
    version = codex.get_config_version()
    if version == codex.CONFIG_V10:
        return CodexConfigV10.load(dirname, filename)
    else:
        raise ValueError(
            'Configuration version "{}" not supported (determined by env variable {})'
            .format(version, codex.ENV_CONFIG_VERSION)
        )


def load_example_config(example_name):
    """Load example configuration stored at codex/config/$SCHEMA_VERSION/examples"""
    return load(osp.join(codex.conf_dir, codex.get_config_version(), 'examples', example_name))
