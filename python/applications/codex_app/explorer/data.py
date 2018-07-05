from skimage.external.tifffile import imread
from codex import io as codex_io
from codex import config as codex_config
from codex import data as codex_data
from codex_app.explorer.config import cfg
from collections import defaultdict
import os
import os.path as osp
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

db = None
ddict = lambda: defaultdict(ddict)
cache = ddict()


class Datastore(object):

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def exists(self, group, key):
        raise NotImplementedError()

    def put(self, group, key, value):
        raise NotImplementedError()

    def get(self, group, key, default=None):
        raise NotImplementedError()

    def restore(self):
        raise NotImplementedError()

    def save(self):
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

    def restore(self):
        import pickle
        path = osp.join(self.data_dir, 'data.pkl')
        if osp.exists(path):
            with open(path, 'rb') as fd:
                self.data = pickle.load(fd)
        return self

    def save(self):
        import pickle
        if not osp.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
        path = osp.join(self.data_dir, 'data.pkl')

        # Choose groups to save data for (at TOW only app level data is saved)
        dbs = {k: v for k, v in self.data.items() if k in ['app']}
        with open(path, 'wb') as fd:
            pickle.dump(dbs, fd)
        return path


def get_montage_image():
    return db.get('images', 'montage')


def get_cytometry_stats():
    return db.get('stats', 'cytometry')


def _get_cytometry_data():
    path = codex_io.get_cytometry_agg_path('csv')
    return pd.read_csv(osp.join(cfg.exp_data_dir, path))


def _get_montage_image():
    from skimage.transform import resize

    path = codex_io.get_montage_image_path(cfg.region_index, cfg.montage_name)
    path = osp.join(cfg.exp_data_dir, path)
    img = codex_io.read_image(path)
    logger.info('Loaded montage image with shape = %s, dtype = %s', img.shape, img.dtype)
    if img.dtype != np.uint8 and img.dtype != np.uint16:
        raise ValueError('Only 8 or 16 bit images are supported (image type = {})'.format(img.dtype))
    if img.shape[0] != cfg.montage_nchannels:
        raise ValueError(
            'Expecting montage from extraction to have {} channels ("{}") but loaded image at {} has shape {}'
            .format(cfg.montage_nchannels, cfg.montage_channels, path, img.shape)
        )

    # Resize the montage image to something much smaller (resize function expects channels last
    # and preserves them if not specified in target shape)
    img = np.moveaxis(img, 0, -1)
    img = resize(
        img, cfg.montage_target_shape, order=0, mode='constant',
        anti_aliasing=False, preserve_range=True).astype(img.dtype)
    img = np.moveaxis(img, -1, 0)
    # Image is now (C, H, W)
    return img


def get_tile_image(tx=0, ty=0):
    path = codex_io.get_extract_image_path(cfg.region_index, tx, ty, cfg.extract_name)
    path = osp.join(cfg.exp_data_dir, path)
    img = codex_io.read_image(path)
    logger.info('Loaded tile image for tile x = %s, tile y = %s, shape = %s, dtype = %s', tx, ty, img.shape, img.dtype)
    if img.dtype != np.uint8 and img.dtype != np.uint16:
        raise ValueError('Only 8 or 16 bit images are supported (image type = {})'.format(img.dtype))

    if img.shape[0] != cfg.extract_nchannels:
        raise ValueError(
            'Expecting tile images from extraction to have {} channels ("{}") but loaded image at {} has shape {}'
            .format(cfg.extract_nchannels, cfg.extract_channels, path, img.shape)
        )
    return img


def initialize():
    global db
    db = DictDatastore(cfg.app_data_dir).restore()

    # Load the montage only if not present (often takes several seconds otherwise)
    if not db.exists('images', 'montage'):
        logger.info('Loading montage image for the first time (this may take a bit but is only necessary once)')
        img = _get_montage_image()
        db.put('images', 'montage', img)

    # Reload this and ignore prior existence since it's fast
    db.put('stats', 'cytometry', _get_cytometry_data())

