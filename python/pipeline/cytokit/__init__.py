import os
import os.path as osp

ENV_PATH_FORMATS = 'CYTOKIT_PATH_FORMATS'
ENV_CONFIG_VERSION = 'CYTOKIT_CONFIG_VERSION'
ENV_CONFIG_DEFAULT_FILENAME = 'CYTOKIT_CONFIG_DEFAULT_FILENAME'
ENV_CPU_ONLY_OPS = 'CYTOKIT_CPU_ONLY_OPS'
ENV_RAW_INDEX_SYMLINKS = 'CYTOKIT_RAW_INDEX_SYMLINKS'
ENV_CYTOMETRY_2D_MODEL_PATH = 'CYTOKIT_CYTOMETRY_2D_MODEL_PATH'
ENV_CYTOMETRY_2D_MODEL_FILE_ID = 'CYTOMETRY_2D_MODEL_FILE_ID'

# Raw microscope files are a mixed bag in terms of format
# and structure so this variable can be used to enable
# specific handlers
ENV_RAW_FILE_TYPE = 'CYTOKIT_RAW_FILE_TYPE'

# Random state for processing operations
SEED = 5523


def get_env_vars():
    """Get map of all Cytokit environment variables"""
    return {k: v for k, v in os.environ.items() if k.startswith('CYTOKIT_')}


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


# ################# #
# Resource Defaults #
# ################# #

DEFAULT_CYTOMETRY_2D_MODEL_URL = 'https://storage.googleapis.com/musc-codex/models/r0.3/nuclei_model.h5'

# #################### #
# File Format Versions #
# #################### #

FF_DEFAULT = 'keyence_multi_cycle_v01'

# Define the default layout for cytokit results on disk
DEFAULT_PATH_FORMATS = dict(
    best_focus_image='best_focus/tile/R{region:03d}_X{x:03d}_Y{y:03d}_Z{z:03d}.tif',
    montage_image='montage/{name}/R{region:03d}.tif',
    cyto_agg='cytometry/data.{extension}',
    cyto_image='cytometry/tile/R{region:03d}_X{x:03d}_Y{y:03d}.tif',
    cyto_stats='cytometry/statistics/R{region:03d}_X{x:03d}_Y{y:03d}.csv',
    illum_func='illumination/function/R{region:03d}.tif',
    proc_data='processor/data.json',
    proc_exec='processor/execution/{date}.json',
    proc_image='processor/tile/R{region:03d}_X{x:03d}_Y{y:03d}.tif',
    extract_image='extract/{name}/R{region:03d}_X{x:03d}_Y{y:03d}.tif'
)


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
test_data_dir = osp.normpath(osp.join(pkg_dir, '..', 'tests', 'data'))
nb_dir = osp.normpath(osp.join(pkg_dir, '..', '..', 'notebooks'))
