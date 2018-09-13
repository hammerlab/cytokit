from skimage.external.tifffile import imread
from codex import io as codex_io
from codex import config as codex_config
from codex import data as codex_data
from codex_app.explorer.config import cfg
from collections import defaultdict
from collections import OrderedDict
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

    def save(self, groups=None):
        import pickle
        if not osp.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
        path = osp.join(self.data_dir, 'data.pkl')

        dbs = self.data
        if groups is not None:
            dbs = {k: v for k, v in self.data.items() if k in groups}

        with open(path, 'wb') as fd:
            pickle.dump(dbs, fd)
        return path


def get_cytometry_stats():
    return db.get('stats', 'cytometry')


def _get_cytometry_data():
    path = cfg.cytometry_data_path
    if path is None:
        path = osp.join(cfg.exp_data_dir, codex_io.get_cytometry_agg_path('csv'))
    return pd.read_csv(path)


def _load_montage_data():
    from skimage.transform import resize

    path = codex_io.get_montage_image_path(cfg.region_index, cfg.montage_name)
    path = osp.join(cfg.exp_data_dir, path)
    img, meta = codex_io.read_tile(path, return_metadata=True)

    # Select cycle and z plane
    img = img[cfg.montage_cycle, cfg.montage_z]
    labels = list(meta['structured_labels'][cfg.montage_cycle, cfg.montage_z])

    logger.info('Loaded montage image with shape = %s, dtype = %s', img.shape, img.dtype)
    if img.dtype != np.uint8 and img.dtype != np.uint16:
        raise ValueError('Only 8 or 16 bit images are supported (image type = {})'.format(img.dtype))

    # Resize the montage image to something much smaller (resize function expects channels last
    # and preserves them if not specified in target shape)
    img = np.moveaxis(img, 0, -1)
    img = resize(
        img, cfg.montage_target_shape, order=0, mode='constant',
        anti_aliasing=False, preserve_range=True).astype(img.dtype)
    img = np.moveaxis(img, -1, 0)

    # Image is now (C, H, W)
    db.put('images', 'montage', img)
    db.put('channels', 'montage', labels)


def get_montage_image():
    return db.get('images', 'montage')


def get_montage_image_channels():
    return db.get('channels', 'montage')


def _tile_loaded():
    return db.get('images', 'tile') is not None


def _load_tile_data(tx, ty):
    # Do nothing if this tile has already been loaded
    if _tile_loaded() and db.get('coords', 'tile') == (tx, ty):
        return

    path = codex_io.get_extract_image_path(cfg.region_index, tx, ty, cfg.extract_name)
    path = osp.join(cfg.exp_data_dir, path)
    img, meta = codex_io.read_tile(path, return_metadata=True)

    # Select cycle and z plane
    img = img[cfg.extract_cycle, cfg.extract_z]
    labels = list(meta['structured_labels'][cfg.extract_cycle, cfg.extract_z])

    logger.info('Loaded tile image for tile x = %s, tile y = %s, shape = %s, dtype = %s', tx, ty, img.shape, img.dtype)
    if img.dtype != np.uint8 and img.dtype != np.uint16:
        raise ValueError('Only 8 or 16 bit images are supported (image type = {})'.format(img.dtype))

    # Image is now (C, H, W)
    db.put('images', 'tile', img)
    db.put('channels', 'tile', labels)
    db.put('coords', 'tile', (tx, ty))


def get_tile_image(tx=0, ty=0):
    _load_tile_data(tx, ty)
    return db.get('images', 'tile')


def get_tile_image_channels():
    if not _tile_loaded():
        _load_tile_data(0, 0)
    return db.get('channels', 'tile')


def get_channel_dtype_map():
    """Infer the bit depth of each channel by using an arbitrary tile"""
    img = get_tile_image()

    channels = get_tile_image_channels()

    map = OrderedDict()
    for i, ch in enumerate(channels):
        if img[i].max() <= 255:
            map[ch] = np.uint8
        else:
            map[ch] = np.uint16
    return map


def initialize():
    global db
    db = DictDatastore(cfg.app_data_dir).restore()

    # Load the montage only if not present (often takes several seconds otherwise)
    if not db.exists('images', 'montage'):
        logger.info('Loading montage image for the first time (this may take a bit but is only necessary once)')
        _load_montage_data()

    # Reload this and ignore prior existence since it's fast
    db.put('stats', 'cytometry', _get_cytometry_data())

