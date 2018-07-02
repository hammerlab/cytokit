import os
import os.path as osp

ENV_PATH_FORMATS = 'CODEX_PATH_FORMATS'
ENV_CONFIG_VERSION = 'CODEX_CONFIG_VERSION'
ENV_CONFIG_DEFAULT_FILENAME = 'CODEX_CONFIG_DEFAULT_FILENAME'
ENV_CPU_ONLY_OPS = 'CODEX_CPU_ONLY_OPS'
ENV_RAW_INDEX_SYMLINKS = 'CODEX_RAW_INDEX_SYMLINKS'
ENV_CYTOMETRY_2D_MODEL_PATH = 'CODEX_CYTOMETRY_2D_MODEL_PATH'
ENV_CYTOMETRY_3D_MODEL_PATH = 'CODEX_CYTOMETRY_3D_MODEL_PATH'

# Raw microscope files are a mixed bag in terms of format
# and structure so this variable can be used to enable
# specific handlers
ENV_RAW_FILE_TYPE = 'CODEX_RAW_FILE_TYPE'


def get_env_vars():
    """Get map of all CODEX environment variables"""
    return {k: v for k, v in os.environ.items() if k.startswith('CODEX_')}


def register_environment(env):
    """Register environment variables if not already set"""
    for k, v in env.items():
        if k not in os.environ:
            os.environ[k] = v


# ####################### #
# Configuration Variables #
# ####################### #

CONFIG_V10 = 'v1.0'
CONFIG_VERSIONS = [CONFIG_V10]


def get_config_version():
    return os.getenv(ENV_CONFIG_VERSION, CONFIG_V10)


def set_config_version(version):
    os.environ[ENV_CONFIG_VERSION] = version


def get_config_default_filename():
    return os.getenv(ENV_CONFIG_DEFAULT_FILENAME, 'experiment.yaml')


def set_config_default_filename(filename):
    os.environ[ENV_CONFIG_DEFAULT_FILENAME] = filename


# #################### #
# File Format Versions #
# #################### #

FF_DEFAULT = 'keyence_multi_cycle_v01'


def get_path_formats():
    return os.getenv(ENV_PATH_FORMATS, FF_DEFAULT)


def set_path_formats(formats):
    os.environ[ENV_PATH_FORMATS] = formats


# ################### #
# Data File Remapping #
# ################### #


def get_raw_index_symlinks():
    if not os.getenv(ENV_RAW_INDEX_SYMLINKS):
        return dict()
    sym = {}
    for k, m in eval(os.getenv(ENV_RAW_INDEX_SYMLINKS)).items():
        sym[k] = {}
        for src, dst in m.items():
            sym[k][int(src)] = int(dst)
    return sym


def set_raw_index_symlinks(links):
    os.environ[ENV_RAW_INDEX_SYMLINKS] = links if isinstance(links, str) else str(links or '{}')


# ############################# #
# Raw Image File Disambiguation #
# ############################# #

FT_GRAYSCALE = 'grayscale'
FT_KEYENCE_RGB = 'keyence_rgb'
FT_KEYENCE_MIXED = 'keyence_mixed'
FT_KEYENCE_REPEAT = 'keyence_repeat'
RAW_FILE_TYPES = [FT_GRAYSCALE, FT_KEYENCE_RGB, FT_KEYENCE_MIXED, FT_KEYENCE_REPEAT]


def get_raw_file_type():
    return os.getenv(ENV_RAW_FILE_TYPE, FT_GRAYSCALE)


# ####################### #
# Project Path Resolution #
# ####################### #

pkg_dir = osp.abspath(osp.dirname(__file__))
conf_dir = osp.normpath(osp.join(pkg_dir, 'configs'))
nb_dir = osp.normpath(osp.join(pkg_dir, '..', '..', 'notebooks'))