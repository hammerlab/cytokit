from skimage.external.tifffile import imread
from codex import io as codex_io
from codex import config as codex_config
from codex import config as codex_config
from codex import data as codex_data
from explorer import config as app_config
import os
import os.path as osp
import pandas as pd
import logging

logger = logging.getLogger(__name__)

DEFAULT_APP_DATA_PATH = osp.join(codex_data.get_cache_dir(), 'app', 'explorer')

db = None

class Datastore(object):

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def exists(self, group, key):
        raise NotImplementedError()

    def put(self, group, key, value):
        raise NotImplementedError()

    def get(self, group, key, default=None):
        raise NotImplementedError()

    def open(self):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()

    def sput(self, group, key, value):
        if self.exists(group, key):
            return
        self.put(group, key)


class DictDatastore(Datastore):

    def __init__(self, data_dir):
        super(DictDatastore, self).__init(data_dir)
        self.data = {}

    def put(self, group, key, value):
        if group not in self.data:
            self.data[group] = {}
        self.data[group][key] = value

    def get(self, group, key, default=None):
        return self.data.get(group, {}).get(key, default)

    def exists(self, group, key):
        return group in self.data and key in self.data[group]

    def open(self):
        import pickle
        path = osp.join(self.data_dir, 'data.pkl')
        if osp.exists(path):
            with open(path, 'r') as fd:
                self.data = pickle.load(fd)
        return self

    def close(self):
        import pickle
        path = osp.join(self.data_dir, 'data.pkl')
        with open(path, 'w') as fd:
            pickle.dump(self.data, fd)
        return self


def montage_image():
    return db.get('images', 'montage')


def exp_config():
    return db.get('experiment', 'config')


def exp_data_dir():
    return db.get('experiment', 'data_dir')


def cytometry_stats():
    return db.get('stats', 'cytometry')


def _get_cytometry_data(exp_data_dir):
    path = codex_io.get_cytometry_agg_path('csv')
    return pd.read_csv(osp.join(exp_data_dir, path))


def _get_montage_image(exp_data_dir):
    from skimage.transform import resize

    ch = app_config.get_montage_channels()
    if len(ch) != 1:
        raise NotImplementedError(
            'Multi-channel montages not yet supported (must select only one channel (channels configured = {}))'
            .format(ch)
        )
    ch = ch[0]

    montage_name = app_config.get_montage_name()
    path = codex_io.get_montage_image_path(app_config.get_region_index(), montage_name)
    img = codex_io.read_image(osp.join(exp_data_dir, path))
    img = img[..., ch]
    assert img.ndim == 2, 'Expecting 2 dims, image shape = {}'.format(img.shape)

    # Resize the montage image to something much smaller
    img = resize(img, (512, 512), order=0)
    return img


def get_tile_image(tx=0, ty=0):
    ch = app_config.get_tile_channels()
    if len(ch) != 1:
        raise NotImplementedError(
            'Multi-channel tiles not yet supported (must select only one channel (channels configured = {}))'
            .format(ch)
        )
    ch = ch[0]
    path = codex_io.get_extract_image_path(app_config.get_region_index(), tx, ty, app_config.get_extract_name())
    img = codex_io.read_image(osp.join(exp_data_dir(), path))
    img = img[..., ch]
    assert img.ndim == 2, 'Expecting 2 dims, image shape = {}'.format(img.shape)
    return img


def initialize(exp_config_path, exp_data_dir, app_data_dir=None):
    exp_config = codex_config.load(exp_config_path)

    if app_data_dir is None:
        assert exp_config.experiment_name, \
            'Experiment name is empty in experiment configuration (config path = {})'.format(config_path)
        app_data_dir = osp.join(DEFAULT_APP_DATA_PATH, exp_config.experiment_name)

    global db
    db = DictDatastore(app_data_dir)

    db.put('experiment', 'config', exp_config)
    db.put('experiment', 'data_dir', exp_data_dir)

    # Load the montage only if not present (often takes several seconds otherwise)
    if not db.exists('images', 'montage'):
        logger.info('Loading montage image for the first time (this may take a bit but is only necessary once)')
        montage_image = _get_montage_image(exp_data_dir)
        db.put('images', 'montage', montage_image)

    # Reload this and ignore prior existence since it's fast
    db.put('stats', 'cytometry', _get_cytometry_data(exp_data_dir))


