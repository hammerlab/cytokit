

import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
from cytokit import config as cytokit_config
from cytokit import io as cytokit_io
import pandas as pd
from skimage.filters import rank
from skimage.filters import threshold_local
from skimage.exposure import rescale_intensity
from skimage.filters import threshold_local, rank
from skimage.morphology import disk, watershed
from skimage.measure import regionprops
from skimage.feature import peak_local_max
from scipy import ndimage
from collections import OrderedDict


def get_background(img):
    img_thresh = threshold_local(img, 501, offset=0)
    img_bg = (img < img_thresh).astype(np.int)
    return img_bg


def segment(img, smooth_size, min_radius, compactness):
    assert img.ndim == 2

    img = rank.median(img, disk(smooth_size))
    img = rank.enhance_contrast(img, disk(smooth_size))

    # It's pretty crucial to do the thresholding in small windows/blocks
    img_bin = img > threshold_local(img, block_size=41, offset=0., method='median')

    img_dist = ndimage.distance_transform_edt(img_bin)
    local_maxi = peak_local_max(img_dist, min_distance=min_radius, indices=False, labels=img_bin)

    # Determine connected region labelings
    markers = ndimage.label(local_maxi)[0]
    img_seg = watershed(-img_dist, markers, mask=img_bin, compactness=compactness, watershed_line=True)
    return img, img_dist, img_bin, img_seg


# def segment(img, smooth_size=5, min_radius=4, compactness=0.):
#     assert img.ndim == 2
#
#     img = rank.median(img, disk(smooth_size))
#     img = rank.enhance_contrast(img, disk(smooth_size))
#     #img = img_as_uint(img)
#     #print(img.dtype)
#
#     #thresh = thresh_global
#     #print(describe(img.ravel()))
#     #thresh = threshold_otsu(img)
#     #print(thresh)
#
#     #img_bin = img > thresh
#     img_bin = img > 0.01
#     #img_bin = img > threshold_local(img, block_size=501)
#
#     # Compute distance to nearest 0-element for each pixel
#     img_dist = ndimage.distance_transform_edt(img_bin)
#     local_maxi = peak_local_max(img_dist, min_distance=min_radius, indices=False)
#
#     # Determine connected region labelings
#     markers = ndimage.label(local_maxi)[0]
#     #img_seg = watershed(-img_dist, markers, mask=img_bin, compactness=.01, watershed_line=True)
#     img_seg = watershed(-img, markers, mask=img_bin, compactness=compactness, watershed_line=True)
#     return img, img_dist, img_bin, img_seg


def quantify(cells, signals):
    res = []
    for i, c in enumerate(cells):
        r = {}
        for sig_name, img_sig in signals.items():
            # Get path in signal image matching patch for cell coordinates
            sig_intensities = img_sig[[c.coords[:,0], c.coords[:,1]]]
            r[sig_name] = sig_intensities.mean()
        r = pd.Series(r)
        r['cell'] = c
        r['x'] = c.centroid[1]
        r['y'] = c.centroid[0]
        r['area'] = c.area
        r['solidity'] = c.solidity
        r['eccentricity'] = c.eccentricity
        res.append(r)
    return pd.DataFrame(res)


# def segment_tile(tile, smooth_size=5, min_radius=4, compactness=.01):
#     # Select relevant images
#     z = 3
#     # imgs = OrderedDict({
#     #     'nuc': tile[0, z, 0],
#     #     'cd3': tile[2, z, 2],
#     #     'cd4': tile[1, z, 3],
#     #     'cd8': tile[3, z, 1],
#     #     'cd40': tile[3, z, 2],
#     #     'cd7': tile[0, z, 2],
#     #     'ki67': tile[0, z, 3],
#     #     'cd38': tile[1, z, 1]
#     # })
#     imgs = OrderedDict({
#         'nuc': tile[0, z, 0],
#         'cd3': tile[2, :, 2].mean(axis=0),
#         'cd4': tile[1, :, 3].mean(axis=0),
#         'cd8': tile[3, :, 1].mean(axis=0),
#         'cd40': tile[3, :, 2].mean(axis=0),
#         'cd7': tile[0, :, 2].mean(axis=0),
#         'ki67': tile[0, :, 3].mean(axis=0),
#         'cd38': tile[1, :, 1].mean(axis=0)
#     })
#
#     imgs = OrderedDict({k: rescale_intensity(v, out_range=(0, 1)) for k, v in imgs.items()})
#
#     # Select nucleus stain
#     img = imgs['nuc'].copy()
#
#     # Determine background/foreground
#     img_bg = get_background(img)
#     img_fg = -(img_bg - 1)
#     assert np.all(np.unique(img_bg) == np.array([0, 1]))
#     assert np.all(np.unique(img_fg) == np.array([0, 1]))
#
#     img_sub = img * img_fg
#
#     img_sm, img_dist, img_bin, img_seg = segment(
#         img_sub, smooth_size=smooth_size, min_radius=min_radius, compactness=compactness)
#
#     cells = regionprops(img_seg, intensity_image=img)
#
#     dfq = quantify(cells, imgs)
#
#     return cells, dfq, img_bg, img_fg, img_sub, img_seg


def segment_tile(tile, z, smooth_size, min_radius, compactness):
    imgs = OrderedDict({
        'nuc': tile[0, z, 0],
        'cd3': tile[2, :, 2].mean(axis=0),
        'cd4': tile[1, :, 3].mean(axis=0),
        'cd8': tile[3, :, 1].mean(axis=0),
        'cd40': tile[3, :, 2].mean(axis=0),
        'cd7': tile[0, :, 2].mean(axis=0),
        'ki67': tile[0, :, 3].mean(axis=0),
        'cd38': tile[1, :, 1].mean(axis=0)
    })

    imgs = OrderedDict({k: rescale_intensity(v, out_range=(0, 1)) for k, v in imgs.items()})

    # Select nucleus stain
    img = imgs['nuc'].copy()

    # Determine background/foreground
    img_bg = get_background(img)
    img_fg = -(img_bg - 1)
    assert np.all(np.unique(img_bg) == np.array([0, 1]))
    assert np.all(np.unique(img_fg) == np.array([0, 1]))

    img_sub = img * img_fg

    img_sm, img_dist, img_bin, img_seg = segment(
        img_sub, smooth_size=smooth_size, min_radius=min_radius, compactness=compactness)

    cells = regionprops(img_seg, intensity_image=img)

    dfq = quantify(cells, imgs)

    return cells, dfq, img_bg, img_fg, img_sub, img_seg