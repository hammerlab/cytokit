import os
import os.path as osp
import numpy as np
from codex import data as codex_data
from codex import config as codex_config

ENV_APP_EXP_CONFIG_PATH = 'APP_EXP_CONFIG_PATH'
ENV_APP_EXP_DATA_DIR = 'APP_EXP_DATA_DIR'
ENV_APP_DATA_DIR = 'APP_DATA_DIR'
ENV_APP_PORT = 'APP_PORT'
ENV_APP_HOST_IP = 'APP_HOST_IP'
ENV_APP_REGION_INDEX = 'APP_REGION_INDEX'
ENV_APP_EXTRACT_NAME = 'APP_EXTRACT_NAME'
ENV_APP_MONTAGE_NAME = 'APP_MONTAGE_NAME'
ENV_MONTAGE_CHANNELS = 'APP_MONTAGE_CHANNELS'
ENV_EXTRACT_CHANNELS = 'APP_EXTRACT_CHANNELS'

DEFAULT_APP_DATA_PATH = osp.join(codex_data.get_cache_dir(), 'app', 'explorer')
DEFAULT_APP_HOST_IP = '0.0.0.0'


class AppConfig(object):

    def __init__(self):
        self._exp_config = codex_config.load(self.exp_config_path)
        self._exp_config.register_environment()

    @property
    def exp_config_path(self):
        return os.environ[ENV_APP_EXP_CONFIG_PATH]

    @property
    def exp_config(self):
        if not self._exp_config:
            self._exp_config = codex_config.load(self.exp_config_path)
        return self._exp_config

    @property
    def montage_target_shape(self):
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
        return int(os.environ[ENV_APP_REGION_INDEX])

    @property
    def montage_name(self):
        return os.environ[ENV_APP_MONTAGE_NAME]

    @property
    def extract_name(self):
        return os.environ[ENV_APP_EXTRACT_NAME]

    @property
    def montage_channels(self):
        return sorted(os.environ[ENV_MONTAGE_CHANNELS].split(','))

    @property
    def montage_nchannels(self):
        return len(self.montage_channels)

    @property
    def extract_channels(self):
        return sorted(os.environ[ENV_EXTRACT_CHANNELS].split(','))

    @property
    def extract_nchannels(self):
        return len(self.extract_channels)


cfg = AppConfig()
