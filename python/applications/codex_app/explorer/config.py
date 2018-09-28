import os
import os.path as osp
import numpy as np
from collections import OrderedDict
from codex import data as codex_data
from codex import config as codex_config

ENV_APP_EXP_NAME = 'APP_EXP_NAME'
ENV_APP_EXP_CONFIG_PATH = 'APP_EXP_CONFIG_PATH'
ENV_APP_CYTO_DATA_PATH = 'APP_CYTO_DATA_PATH'
ENV_APP_EXP_DATA_DIR = 'APP_EXP_DATA_DIR'
ENV_APP_DATA_DIR = 'APP_DATA_DIR'
ENV_APP_PORT = 'APP_PORT'
ENV_APP_HOST_IP = 'APP_HOST_IP'
ENV_APP_MAX_MONTAGE_CELLS = 'APP_MAX_MONTAGE_CELLS'
ENV_APP_MAX_TILE_CELLS = 'APP_MAX_TILE_CELLS'
ENV_APP_MAX_SINGLE_CELLS = 'APP_MAX_SINGLE_CELLS'
ENV_APP_REGION_INDEX = 'APP_REGION_INDEX'
ENV_APP_EXTRACT_NAME = 'APP_EXTRACT_NAME'
ENV_APP_MONTAGE_NAME = 'APP_MONTAGE_NAME'
ENV_APP_MONTAGE_CHANNEL_NAMES = 'APP_MONTAGE_CHANNEL_NAMES'
ENV_APP_MONTAGE_CHANNEL_COLORS = 'APP_MONTAGE_CHANNEL_COLORS'
ENV_APP_MONTAGE_CHANNEL_RANGES = 'APP_MONTAGE_CHANNEL_RANGES'
ENV_APP_EXTRACT_BIT_DEPTH = 'APP_EXTRACT_BIT_DEPTH'
ENV_APP_MONTAGE_CYCLE = 'MONTAGE_CYCLE '
ENV_APP_MONTAGE_Z = 'MONTAGE_Z'
ENV_APP_EXTRACT_CYCLE = 'APP_EXTRACT_CYCLE'
ENV_APP_EXTRACT_Z = 'APP_EXTRACT_Z'
ENV_APP_CELL_IMAGE_WIDTH = 'APP_CELL_IMAGE_WIDTH'
ENV_APP_CELL_IMAGE_HEIGHT = 'APP_CELL_IMAGE_HEIGHT'
ENV_APP_CELL_IMAGE_DISPLAY_WIDTH = 'APP_CELL_IMAGE_DISPLAY_WIDTH'
ENV_APP_GRAPH_POINT_OPACITY = 'APP_GRAPH_POINT_OPACITY'
ENV_APP_MONTAGE_POINT_COLOR = 'APP_MONTAGE_POINT_COLOR'
ENV_APP_TILE_POINT_COLOR = 'APP_TILE_POINT_COLOR'

DEFAULT_APP_DATA_PATH = osp.join(codex_data.get_cache_dir(), 'app', 'explorer')
DEFAULT_APP_HOST_IP = '0.0.0.0'
DEFAULT_MAX_MONTAGE_CELLS = 10000
DEFAULT_MAX_TILE_CELLS = 10000
DEFAULT_MAX_SINGLE_CELLS = 50
DEFAULT_CELL_IMAGE_HEIGHT = 64
DEFAULT_CELL_IMAGE_WIDTH = 64
DEFAULT_GRAPH_POINT_OPACITY = .2
DEFAULT_APP_MONTAGE_POINT_COLOR = 'white'
DEFAULT_APP_TILE_POINT_COLOR = 'white'


class AppConfig(object):

    CYTO_FIELDS = OrderedDict([
        ('cell_diameter', 'Cell Diameter'),
        ('nucleus_diameter', 'Nucleus Diameter'),
        ('cell_size', 'Cell Size'),
        ('cell_solidity', 'Cell Solidity'),
        ('cell_circularity', 'Cell Circularity'),
        ('nucleus_size', 'Nucleus Size'),
        ('nucleus_solidity', 'Nucleus Solidity'),
        ('nucleus_circularity', 'Nucleus Circularity'),
        ('region_index', 'Region Index'),
        ('tile_x', 'Tile X'),
        ('tile_y', 'Tile Y'),
        ('id', 'Cell ID'),
        ('rid', 'Cell ID (In Region)'),
        ('rx', 'Cell X (In Region)'),
        ('ry', 'Cell Y (In Region)'),
        ('x', 'Cell X (In Tile)'),
        ('y', 'Cell Y (In Tile)'),
        ('cg:n_neighbors', 'Cell Num Neighbors (w/ Shared Boundary)'),
        ('cg:adj_bg_pct', 'Cell Percent Bordering Background'),
    ])
    CYTO_HOVER_FIELDS = [
        'id',
        'nucleus_diameter', 'nucleus_solidity',
        'cell_diameter', 'cell_size',
        'cg:n_neighbors', 'cg:adj_bg_pct'
    ]
    CYTO_INT_FIELDS = ['id', 'rid', 'x', 'y', 'rx', 'ry', 'tile_x', 'tile_y', 'region_index', 'cg:n_neighbors']

    def __init__(self):
        self._exp_config = codex_config.load(self.exp_config_path)
        self._exp_config.register_environment()

    @property
    def exp_name(self):
        return os.getenv(ENV_APP_EXP_NAME, '')

    @property
    def exp_config_path(self):
        return os.environ[ENV_APP_EXP_CONFIG_PATH]

    @property
    def cytometry_data_path(self):
        return os.getenv(ENV_APP_CYTO_DATA_PATH)

    @property
    def exp_config(self):
        if not self._exp_config:
            self._exp_config = codex_config.load(self.exp_config_path)
        return self._exp_config

    @property
    def montage_target_shape(self):
        # Return shape as rows, cols
        return 512, 512

    @property
    def montage_shape(self):
        rh = self._exp_config.region_height * self._exp_config.tile_height
        rw = self._exp_config.region_width * self._exp_config.tile_width
        return rh, rw

    @property
    def region_shape(self):
        return self._exp_config.region_height, self._exp_config.region_width

    @property
    def montage_target_scale_factors(self):
        """Montage scaling factors as (scale_y, scale_x)"""
        return tuple(np.array(self.montage_target_shape) / np.array(self.montage_shape))

    @property
    def tile_shape(self):
        return self._exp_config.tile_height, self._exp_config.tile_width

    @property
    def montage_channel_names(self):
        names = os.getenv(ENV_APP_MONTAGE_CHANNEL_NAMES)
        return None if names is None else names.split(',')

    @property
    def montage_channel_colors(self):
        colors = os.getenv(ENV_APP_MONTAGE_CHANNEL_COLORS)
        return None if colors is None else colors.split(',')

    @property
    def montage_channel_ranges(self):
        ranges = os.getenv(ENV_APP_MONTAGE_CHANNEL_RANGES)
        # Split on command and separate into start and stop tuples
        return None if ranges is None else [
            (int(v.split('-')[0]), int(v.split('-')[1]))
            for v in ranges.split(',')
        ]

    @property
    def exp_name(self):
        assert self._exp_config.experiment_name, \
            'Experiment name is empty in experiment configuration (config path = {})'.format(self.exp_config_path)
        return self._exp_config.experiment_name

    @property
    def exp_data_dir(self):
        return os.environ[ENV_APP_EXP_DATA_DIR]

    @property
    def app_data_dir(self):
        return os.getenv(ENV_APP_DATA_DIR, osp.join(DEFAULT_APP_DATA_PATH, self.exp_name))

    @property
    def app_port(self):
        port = os.getenv(ENV_APP_PORT)
        return None if port is None else int(port)

    @property
    def app_host_ip(self):
        return os.getenv(ENV_APP_HOST_IP, DEFAULT_APP_HOST_IP)

    @property
    def region_index(self):
        return int(os.getenv(ENV_APP_REGION_INDEX, 0))

    @property
    def montage_name(self):
        return os.environ[ENV_APP_MONTAGE_NAME]

    @property
    def montage_cycle(self):
        return int(os.getenv(ENV_APP_MONTAGE_CYCLE, '0'))

    @property
    def montage_z(self):
        return int(os.getenv(ENV_APP_MONTAGE_Z, '0'))

    @property
    def montage_grid_enabled(self):
        return True

    @property
    def extract_name(self):
        return os.environ[ENV_APP_EXTRACT_NAME]

    @property
    def extract_cycle(self):
        return int(os.getenv(ENV_APP_EXTRACT_CYCLE, '0'))

    @property
    def extract_z(self):
        return int(os.getenv(ENV_APP_EXTRACT_Z, '0'))

    @property
    def max_montage_cells(self):
        return int(os.getenv(ENV_APP_MAX_MONTAGE_CELLS, DEFAULT_MAX_MONTAGE_CELLS))

    @property
    def max_tile_cells(self):
        return int(os.getenv(ENV_APP_MAX_TILE_CELLS, DEFAULT_MAX_TILE_CELLS))

    @property
    def max_single_cells(self):
        return int(os.getenv(ENV_APP_MAX_SINGLE_CELLS, DEFAULT_MAX_SINGLE_CELLS))

    @property
    def cell_image_size(self):
        """Get patch dimensions for cell image extractions at original resolution

        This should roughly encompass a bounding box in which cells will be centered, so it should
        be big enough to fit most cells, and any cell images exceeding this size (on original scale)
        will be cropped to fit.
        """
        return (
            int(os.getenv(ENV_APP_CELL_IMAGE_HEIGHT, DEFAULT_CELL_IMAGE_HEIGHT)),
            int(os.getenv(ENV_APP_CELL_IMAGE_WIDTH, DEFAULT_CELL_IMAGE_WIDTH))
        )

    @property
    def cell_image_display_width(self):
        """Get cell image display width in pixels (height will scale proportionally)"""
        width = os.getenv(ENV_APP_CELL_IMAGE_DISPLAY_WIDTH)
        if width is None:
            return None
        return int(width)

    @property
    def graph_point_opacity(self):
        return float(os.getenv(ENV_APP_GRAPH_POINT_OPACITY, DEFAULT_GRAPH_POINT_OPACITY))

    @property
    def random_state(self):
        return 1

    @property
    def montage_point_color(self):
        return os.getenv(ENV_APP_MONTAGE_POINT_COLOR, DEFAULT_APP_MONTAGE_POINT_COLOR)

    @property
    def tile_point_color(self):
        return os.getenv(ENV_APP_TILE_POINT_COLOR, DEFAULT_APP_TILE_POINT_COLOR)


cfg = AppConfig()
