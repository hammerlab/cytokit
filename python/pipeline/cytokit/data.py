import os
import requests
from os import path as osp
import logging
import cytokit
logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = osp.join('.cytokit', 'cache')
ENV_CACHE_DIR = 'CYTOKIT_CACHE_DIR'
ENV_DATA_DIR = 'CYTOKIT_DATA_DIR'


def get_data_dir():
    return os.environ[ENV_DATA_DIR]


def get_cache_dir():
    # Use explicit cache location, if set
    if os.getenv(ENV_CACHE_DIR):
        return os.getenv(ENV_CACHE_DIR)
    # Next, use path residing under data directory if set
    if os.getenv(ENV_DATA_DIR):
        return osp.join(os.getenv(ENV_DATA_DIR), DEFAULT_CACHE_DIR)
    # Otherwise, use path under home directory
    return osp.expanduser(osp.join('~', DEFAULT_CACHE_DIR))


def _resolve_cache_path(path):
    return osp.join(get_cache_dir(), path)


def download(url, path):
    import urllib.request
    if not osp.exists(path):
        os.makedirs(osp.dirname(path), exist_ok=True)
        logger.info('Downloading url "{}" to local path "{}"'.format(url, path))
        urllib.request.urlretrieve(url, path)
    return path


def download_file_from_google_drive(id, path, name=None):
    url = "https://docs.google.com/uc?export=download"

    if not osp.exists(path):
        os.makedirs(osp.dirname(path), exist_ok=True)
        logger.info('Downloading google drive file "{}" to local path "{}"'.format(name or id, path))

        session = requests.Session()
        response = session.get(url, params={'id': id}, stream=True)
        token = _get_confirm_token(response)
        if token:
            params = {'id': id, 'confirm': token}
            response = session.get(url, params=params, stream=True)
        _save_response_content(response, path)
    return path


def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def _save_response_content(response, destination):
    chunk_size = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def initialize_best_focus_model():
    from cytokit.miq.constants import REMOTE_MODEL_CHECKPOINT_PATH
    file_extensions = [".index", ".meta", ".data-00000-of-00001"]
    model_path = _resolve_cache_path(osp.join('best_focus', 'model'))
    for extension in file_extensions:
        remote_path = REMOTE_MODEL_CHECKPOINT_PATH + extension
        local_path = osp.join(model_path, osp.basename(remote_path))
        download(remote_path, local_path)

    # Return path to checkpoint, to be fed directory to tensorflow restore operations
    return osp.join(model_path, osp.basename(REMOTE_MODEL_CHECKPOINT_PATH))


def initialize_cytometry_2d_model():
    from cytokit import cytometry
    default_path = _resolve_cache_path(osp.join('cytometry', 'model', 'unet_v2_weights.h5'))
    path = os.getenv(cytokit.ENV_CYTOMETRY_2D_MODEL_PATH, default_path)
    file_id = os.getenv(cytokit.ENV_CYTOMETRY_2D_MODEL_FILE_ID, cytometry.CYTOMETRY_UNET_V2_WEIGHTS_FILE_ID)

    # Return path to model h5 file
    return download_file_from_google_drive(file_id, path, name='UNet Weights')
