import os
import os.path as osp

ENV_FILE_VERSION = 'CODEX_FILE_FORMAT_VERSION'
ENV_CONFIG_VERSION = 'CODEX_CONFIG_VERSION'
ENV_CPU_ONLY_OPS = 'CODEX_CPU_ONLY_OPS'
ENV_RAW_INDEX_SYMLINKS = 'CODEX_RAW_INDEX_SYMLINKS'

# ############################# #
# Configuration Schema Versions #
# ############################# #

CONFIG_V01 = 'v0.1'
CONFIG_VERSIONS = [CONFIG_V01]


def get_config_version():
    return os.getenv(ENV_CONFIG_VERSION, CONFIG_V01)


def set_config_version(version):
    os.environ[ENV_CONFIG_VERSION] = version


# #################### #
# File Format Versions #
# #################### #

FF_V01 = 'v0.1'
FF_V02 = 'v0.2'
FF_VERSIONS = [FF_V01, FF_V02]


def get_file_format_version():
    return os.getenv(ENV_FILE_VERSION, FF_V01)


def set_file_format_version(version):
    os.environ[ENV_FILE_VERSION] = version


# ################### #
# Data File Remapping #
# ################### #


def get_raw_index_symlinks():
    return eval(os.getenv(ENV_RAW_INDEX_SYMLINKS, '{}'))


def set_raw_index_symlinks(links=None):
    os.environ[ENV_RAW_INDEX_SYMLINKS] = links if isinstance(links, str) else str(links or '{}')


# ####################### #
# Project Path Resolution #
# ####################### #

pkg_dir = osp.abspath(osp.dirname(__file__))
conf_dir = osp.normpath(osp.join(pkg_dir, '../../../config'))
