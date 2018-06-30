import os
import os.path as osp

ENV_APP_REGION_INDEX = 'APP_REGION_INDEX'
ENV_APP_EXTRACT_NAME = 'APP_EXTRACT_NAME'
ENV_APP_MONTAGE_NAME = 'APP_MONTAGE_NAME'
ENV_MONTAGE_CHANNELS = 'APP_MONTAGE_CHANNELS'
ENV_TILE_CHANNELS = 'APP_TILE_CHANNELS'


def get_region_index():
    return int(os.environ[ENV_APP_REGION_INDEX])


def get_montage_name():
    return os.environ[ENV_APP_MONTAGE_NAME]


def get_extract_name():
    return os.environ[ENV_APP_EXTRACT_NAME]


def get_montage_channels():
    return [int(v) for v in os.getenv(ENV_MONTAGE_CHANNELS, '0').split(',')]


def get_tile_channels():
    return [int(v) for v in os.getenv(ENV_TILE_CHANNELS, '0').split(',')]