import numpy as np
from skimage import filters
from skimage import measure
from skimage import exposure
from skimage import feature
from skimage import morphology
from skimage import segmentation
from skimage import img_as_float
from cytokit import math as ck_math
from cytokit.cytometry import cytometer
from scipy.ndimage.morphology import distance_transform_edt
import logging
logger = logging.getLogger(__name__)


class SpheroidCytometer(cytometer.Cytometer):
    
    def __init__(self, **kwargs):
        pass
    
    def initialize(self):
        pass
    
    def segment(self, img, **kwargs):
        assert img.ndim == 3, 'Expecting 3D image, got shape %s' % img.shape
        
        # Run max-z projection to create single 2D image (but record initial number of z planes)
        nz = img.shape[0]
        logger.info('Max projecting z dimension from %s planes to 1', nz)
        img = img.max(axis=0)
        
        # Verify 8 or 16 bit type before forcing to 8 bit (for median filtering)
        assert img.dtype in [np.uint8, np.uint16], 'Expecting 8 or 16 bit image but got type %s' % img.dtype
        if img.dtype == np.uint16:
            img = exposure.rescale_intensity(img, out_range='uint8').astype(np.uint8)
        assert img.dtype == np.uint8
        img_raw = img
        
        # Preprocess to remove outliers and blur
        img = filters.median(img, selem=morphology.square(3))
        img = filters.gaussian(img, sigma=3)
        
        # Median requires 8 or 16 bit images (or a warning is thrown) but gaussian filter
        # will convert result per img_as_float conventions leaving 0-1 image (verify that
        # and stretch to 0-1)
        assert img.min() >= 0 and img.max() <= 1, \
            'Expecting 0-1 image but got range %s - %s' % (img.min(), img.max())
        img = exposure.rescale_intensity(img, out_range=(0, 1))
        assert img.min() == 0 and img.max() == 1

        # Run open-close morphological reconstruction
        img_seed = morphology.erosion(img, selem=morphology.disk(8))
        img = morphology.reconstruction(img_seed, img, method='dilation')
        img_seed = morphology.dilation(img, selem=morphology.disk(8))
        img = morphology.reconstruction(img_seed, img, method='erosion')

        # Compute gradient, blur, and threshold to give reasonable outlines of objects
        img_grad = filters.sobel(img)
        img_grad = filters.gaussian(img_grad, sigma=1)
        img_grad_bin = img_grad > filters.threshold_li(img_grad)

        # Fill in outlines by performing large radius closing
        img_mask = morphology.remove_small_objects(img_grad_bin, min_size=64)
        img_mask = morphology.binary_closing(img_mask, selem=morphology.disk(8))
        img_mask = morphology.remove_small_holes(img_mask, area_threshold=2048)

        # Determine seed for primary object segmentation as peak local max in mask dist
        img_mask_dist = distance_transform_edt(img_mask)
        img_mask_dist = filters.gaussian(img_mask_dist, sigma=5) # Large sigma helps join nearby peaks
        img_markers = feature.peak_local_max(img_mask_dist, indices=False, exclude_border=False, min_distance=64)
        img_markers = measure.label(img_markers)
        img_basin = -img_mask_dist

        # Segment larger objects (i.e. spheroids)
        img_seg = segmentation.watershed(img_basin, img_markers, mask=img_mask).astype(np.uint16)

        # Convert (h, w) -> required (z, (cell, nucleus, ...[others]), h, w) format 
        img_seg = np.stack([
            img_seg, img_seg,
            exposure.rescale_intensity(img_grad, out_range='uint16').astype(np.uint16),
            img_grad_bin.astype(np.uint16),
            exposure.rescale_intensity(img_basin, out_range='uint16').astype(np.uint16),
            img_markers.astype(np.uint16)
        ])
        return np.stack([img_seg]*nz)
    
    def quantify(self, tile, segmentation, **kwargs):
        return cytometer.Cytometer2D.quantify(None, tile, segmentation, **kwargs)

    def augment(self, df, config):
        df = cytometer.Cytometer2D.augment(None, df, config)
        # Attempt to sum live + dead intensities if both channels are present
        for agg_fun in ['mean', 'sum']:
            cols = df.filter(regex='ci:(LIVE|DEAD):{}'.format(agg_fun)).columns.tolist()
            if len(cols) == 2:
                df['ci:LIVE+DEAD:{}'.format(agg_fun)] = df[cols[0]] + df[cols[0]]
        return df