from skimage.external.tifffile import imread
from codex import io as codex_io
from codex import config as codex_config
import os.path as osp
import pandas as pd

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
    # Load configuration and expression data resulting from CODEX segmentation
    df = pd.read_csv(path, sep='\t')
    df = df.rename(columns={'Filename:Filename': 'tile'})
    cyc_names = list(df.filter(regex='Cyc_').columns.values)
    assert len(cyc_names) == len(config.channel_names)
    df = df.rename(columns=dict(zip(cyc_names, config.channel_names)))
    return df

