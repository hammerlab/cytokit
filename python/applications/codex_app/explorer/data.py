from skimage.external.tifffile import imread
from codex import io as codex_io
from codex import config as codex_config
from codex import data as codex_data
from codex_app.explorer.config import cfg
import os
import os.path as osp
import pandas as pd
import logging

logger = logging.getLogger(__name__)

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
        super(DictDatastore, self).__init__(data_dir)
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


def get_montage_image():
    return db.get('images', 'montage')


def get_cytometry_stats():
    return db.get('stats', 'cytometry')


def _get_cytometry_data():
    path = codex_io.get_cytometry_agg_path('csv')
    return pd.read_csv(osp.join(cfg.exp_data_dir, path))


def _get_montage_image():
    from skimage.transform import resize

    ch = cfg.montage_channels
    if len(ch) != 1:
        raise NotImplementedError(
            'Multi-channel montages not yet supported (must select only one channel (channels configured = {}))'
            .format(ch)
        )
    ch = ch[0]

    path = codex_io.get_montage_image_path(cfg.region_index, cfg.montage_name)
    img = codex_io.read_image(osp.join(cfg.exp_data_dir, path))
    logger.info('Loaded montage image with shape = %s, dtype = %s', img.shape, img.dtype)
    img = img[ch]
    assert img.ndim == 2, 'Expecting 2 dims, image shape = {}'.format(img.shape)

    # Resize the montage image to something much smaller
    img = resize(img, cfg.montage_target_shape, order=0, mode='constant')
    return img


def get_tile_image(tx=0, ty=0):
    ch = cfg.tile_channels
    if len(ch) != 1:
        raise NotImplementedError(
            'Multi-channel tiles not yet supported (must select only one channel (channels configured = {}))'
            .format(ch)
        )
    ch = ch[0]
    path = codex_io.get_extract_image_path(cfg.region_index, tx, ty, cfg.extract_name)
    img = codex_io.read_image(osp.join(cfg.exp_data_dir, path))
    logger.info('Loaded tile image for tile x = %s, tile y = %s, shape = %s, dtype = %s', tx, ty, img.shape, img.dtype)
    img = img[ch]
    assert img.ndim == 2, 'Expecting 2 dims, image shape = {}'.format(img.shape)
    return img


def initialize():
    global db
    db = DictDatastore(cfg.app_data_dir)

    # Load the montage only if not present (often takes several seconds otherwise)
    if not db.exists('images', 'montage'):
        logger.info('Loading montage image for the first time (this may take a bit but is only necessary once)')
        img = _get_montage_image()
        db.put('images', 'montage', img)

    # Reload this and ignore prior existence since it's fast
    db.put('stats', 'cytometry', _get_cytometry_data())


