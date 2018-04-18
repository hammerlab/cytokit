
import numpy as np
import scipy as sp
from flowdec import data as fd_data
from codex import config as codex_config


def load_simulated_bars_experiment(blur=False):
    ref_img = fd_data.bars_25pct().data if blur else fd_data.bars_25pct().actual
    # Subset image to not be equal in x and y
    ref_img = ref_img[:, :48, :]
    return experiment_from_stack(ref_img)


def load_celegans_experiment(blur=False, channel='CY3'):
    ref_img = fd_data.load_celegans_channel('CY3').data
    return experiment_from_stack(ref_img)


def experiment_from_img(img):
    """Create simulated experiment from image in (z, rows, cols) order"""

    # Add 30 rows of zeros and 15 columns of zeros to simulate tile overlap
    pad_rows, pad_cols = 30, 15
    ref_img = np.pad(img, ((0, 0), (pad_rows, pad_rows), (pad_cols, pad_cols)), mode='constant')

    # Apply a shift to test drift compensation
    shift = (10, -5, 15)
    offset_img = sp.ndimage.shift(ref_img, shift)

    # Repeat images in a 5D stack shaped like (cycles, z, channel, height, width)
    nch, ncyc = 4, 3
    ref_cycle = np.stack([ref_img]*nch, 1)
    off_cycle = np.stack([offset_img]*nch, 1)
    tile = np.stack([ref_cycle] + [off_cycle]*(ncyc-1), 0)

    # Generate a simulated experiment configuration
    exp_config = dict(
        num_cycles=tile.shape[0],
        num_z_planes=tile.shape[1],
        channel_names=['CH{}'.format(i) for i in range(tile.shape[2])],
        tile_height=img.shape[1],  # Note that this is original image, not image with padding
        tile_width=img.shape[2],
        driftCompReferenceCycle=1,
        drift_comp_channel=1,
        bestFocusReferenceCycle=1,
        best_focus_channel=1,
        tile_overlap_X=pad_cols*2,
        tile_overlap_Y=pad_rows*2,
        emission_wavelengths=[425, 525, 595, 670],
        objectiveType='air',
        per_pixel_XY_resolution=377.442,
        z_pitch=1500.,
        numerical_aperture=.75,
        magnification=20,

    )
    channel_names = ['CH-{}-{}'.format(i, j) for i in range(ncyc) for j in range(nch)]
    config = codex_config.CodexConfigV1(exp_config=exp_config, processing_options={}, channel_names=channel_names)

    return tile, config, {'shift': shift}
