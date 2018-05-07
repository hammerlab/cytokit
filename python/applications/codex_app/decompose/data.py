from skimage.external.tifffile import imread
from codex import io as codex_io
from codex import config as codex_config
from codex.analyze import expression as codex_expr
import os.path as osp

DATA_DIR = None


def set_data_dir(dir):
    global DATA_DIR
    DATA_DIR = dir


def get_tile_image(ireg, ix, iy):
    path = codex_io.get_processor_img_path(ireg, ix, iy)
    return imread(osp.join(DATA_DIR, path))


def get_experiment_config():
    return codex_config.load(DATA_DIR)


def read_expression_file(config, path):
    return codex_expr.read_expression_file(config, path)

