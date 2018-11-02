import numpy as np
import scipy as sp
import os.path as osp
from flowdec import data as fd_data
import cytokit
from cytokit import config as cytokit_config


def load_simulated_bars_experiment(blur=False, **kwargs):
    ref_img = fd_data.bars_25pct().data if blur else fd_data.bars_25pct().actual
    # Subset image to not be equal in x and y
    ref_img = ref_img[:, :48, :]
    return experiment_from_img(ref_img, **kwargs)


def load_celegans_experiment(**kwargs):
    ref_img = fd_data.load_celegans_channel('CY3').data
    return experiment_from_img(ref_img, **kwargs)


def get_example_config(example_name='ex1'):
    path = osp.join(cytokit.conf_dir, 'v0.1', 'examples', example_name)
    return cytokit_config.load(path)


def experiment_from_img(img, nch=4, ncyc=3, src_config=None, shift=(0, -5, 15)):
    """Create simulated experiment from image in (z, rows, cols) order"""

    if src_config is None:
        src_config = get_example_config()

    # Add 30 rows of zeros and 15 columns of zeros to simulate tile overlap
    pad_rows, pad_cols = 30, 15
    ref_img = np.pad(img, ((0, 0), (pad_rows, pad_rows), (pad_cols, pad_cols)), mode='constant')

    # Apply a shift to test drift compensation
    offset_img = sp.ndimage.shift(ref_img, shift)

    # Repeat images in a 5D stack shaped like (cycles, z, channel, height, width)
    ref_cycle = np.stack([ref_img]*nch, 1)
    off_cycle = np.stack([offset_img]*nch, 1)
    tile = np.stack([ref_cycle] + [off_cycle]*(ncyc-1), 0)

    # Generate a simulated experiment configuration
    acq_config = dict(
        num_cycles=tile.shape[0],
        num_z_planes=tile.shape[1],
        channel_names=['CH{}'.format(i) for i in range(ncyc * nch)],
        per_cycle_channel_names=['CH{}'.format(i) for i in range(nch)],
        emission_wavelengths=src_config.microscope_params.em_wavelength_nm[0:nch] * ncyc,
        tile_height=img.shape[1],  # Note that this is original image, not image with padding
        tile_width=img.shape[2],
        tile_overlap_x=pad_cols * 2,
        tile_overlap_y=pad_rows * 2,
        region_height=1,
        region_width=1
    )

    # Overwrite acquisition properties in source configuration and remove operator/analysis definitions
    config = dict(src_config._conf)
    config['acquisition'].update(acq_config)
    config['operator'] = []
    config['analysis'] = []
    config['processor']['drift_compensation'] = dict(channel='CH0')
    config = cytokit_config.CytokitConfigV10(config)

    return tile, config, {'shift': shift}
