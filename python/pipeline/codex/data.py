
import os
from os import path as osp
import logging
logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = '~/.codex_app/cache'
ENV_CACHE_DIR = 'CODEX_CACHE_DIR'

BEST_FOCUS_MODEL = "https://storage.googleapis.com/microscope-image-quality/static/model/model.ckpt-1000042"


def get_cache_dir():
    return os.getenv(ENV_CACHE_DIR, osp.expanduser(DEFAULT_CACHE_DIR))


def _resolve_cache_path(path):
    return osp.join(get_cache_dir(), path)


def download(url, path):
    import urllib.request
    if not osp.exists(path):
        os.makedirs(osp.dirname(path), exist_ok=True)
        logger.debug('Downloading url "{}" to local path "{}"'.format(url, path))
        urllib.request.urlretrieve(url, path)
    return path


def initialize_best_focus_model():
    from codex.miq.constants import REMOTE_MODEL_CHECKPOINT_PATH
    file_extensions = [".index", ".meta", ".data-00000-of-00001"]
    model_path = _resolve_cache_path(osp.join('best_focus', 'model'))
    for extension in file_extensions:
        remote_path = REMOTE_MODEL_CHECKPOINT_PATH + extension
        local_path = osp.join(model_path, osp.basename(remote_path))
        download(remote_path, local_path)

    # Return path to checkpoint, to be fed directory to tensorflow restore operations
    return osp.join(model_path, osp.basename(REMOTE_MODEL_CHECKPOINT_PATH))
