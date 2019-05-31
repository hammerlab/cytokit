import numpy as np
import pandas as pd
from skimage import util
from skimage import draw
from skimage import filters
from skimage import measure
from skimage import exposure
from skimage import transform
from skimage import feature
from skimage import morphology
from skimage import segmentation
from skimage import img_as_float
from skimage import io as sk_io
from skimage.feature.blob import _prune_blobs
from centrosome import propagate
from cytokit import math as ck_math
from cytokit.cytometry import cytometer
from scipy import ndimage as ndi
import logging
logger = logging.getLogger(__name__)

####################
# 20X Implementation
####################


class SpheroidCytometer20x(cytometer.Cytometer):
    
    def __init__(self, config, sampling=None):
        """Cytometer Initialization

        Args:
            config: Experiment configuration
            sampling: Sampling distance proportions as 3-item list-like with order (z, y, x)
        """
        super().__init__(config)
        
        if sampling is not None:
            assert len(sampling) == 3, \
                'Sampling factors must be 3 item list-like'
            assert all([s > 0 for s in sampling]), \
                'All sampling factors must be > 0 (given {})'.format(sampling)
            self.sampling = np.array(sampling)
        else:
            # Relative voxel sizes (typically around [10, 1, 1])
            self.sampling = np.array([config.axial_sampling_ratio, 1.0, 1.0])
            
        # Multiplicative factors for rescaling isotropic arguments
        self.factors = 1.0 / self.sampling
        logger.debug('Cytometer initialized with sampling rates %s', tuple(self.sampling))
    
    def initialize(self):
        pass
        
        
    def get_primary_object_mask(self, img, img_pk):
        assert img.ndim == 3, 'Expecting 3D image, got shape {}'.format(img.shape)
        assert img_pk.dtype == np.bool
        # Remove frequencies above scale of individual cells (this results in clusters near spheroid centers)
        img = np.abs(ndi.gaussian_filter(img, sigma=1*self.factors) - ndi.gaussian_filter(img, sigma=8*self.factors))
        img = img.max(axis=0)
        img = img > filters.threshold_otsu(img)
        img = img | img_pk # Merge threshold mask with given peaks/markers
        img = morphology.binary_closing(img, selem=morphology.disk(8))
        img = ndi.morphology.binary_fill_holes(img)
        img = morphology.binary_opening(img, selem=morphology.disk(8))
        # Not necessary with larger opening (min size = ~227 w/ 8 radius disk)
        # img = morphology.remove_small_objects(img, min_size=256) 
        return img
    
    def segment(self, img, include_intermediate_results=False, **kwargs):
        assert img.ndim == 3, 'Expecting 3D image, got shape {}'.format(img.shape)
        img = ndi.median_filter(img, size=(1, 3, 3))
        img = img_as_float(img)
        img = util.invert(img)
        
        img_mz = img.max(axis=0)
        img_mz = exposure.rescale_intensity(img_mz, out_range=(0, 1))
        peaks, img_dog, sigmas = blob_dog(img_mz, min_sigma=8, max_sigma=128, sigma_ratio=1.6, overlap=.25, threshold=1.75)
        
        img_pk = np.zeros(img_mz.shape, dtype=bool)
        img_pk[peaks[:,0].astype(int), peaks[:,1].astype(int)] = True
        img_pk = morphology.label(img_pk)
        
        # Get mask to conduct segmentation over
        img_pm = self.get_primary_object_mask(img, morphology.binary_dilation(img_pk > 0, morphology.disk(32)))
        
        
        #img_dt = ndi.distance_transform_edt(img_pm)
        #img_obj = segmentation.watershed(-img_dt, img_pk, mask=img_pm).astype(np.uint16)
        img_obj = propagate.propagate(img_mz, img_pk, img_pm, .01)[0].astype(np.uint16)
        img_bnd = img_obj * segmentation.find_boundaries(img_obj, mode='inner', background=0)
        
        img_seg = [img_obj, img_obj, img_bnd, img_bnd]
        if include_intermediate_results:
            to_uint16 = lambda im: exposure.rescale_intensity(im, out_range='uint16').astype(np.uint16)
            img_seg += [
                to_uint16(img_mz), 
                to_uint16(img_dog[0]),
                to_uint16(img_dog[1]),
                img_pm.astype(np.uint16),
                img_pk.astype(np.uint16)
            ]
            
        # Stack and add new axis to give to (z, ch, h, w)
        img_seg = np.stack(img_seg)[np.newaxis]
        assert img_seg.dtype == np.uint16, 'Expecting 16bit result, got type {}'.format(img_seg.dtype)
        assert img_seg.ndim == 4, 'Expecting 4D result, got shape {}'.format(img_seg.shape)
        return img_seg

    def quantify(self, tile, segments, **kwargs):
        assert tile.ndim == 5
        # Run max-z projection across all channels and insert new axis where z dimension was
        tile = tile.max(axis=1)[:, np.newaxis]
        assert tile.ndim == 5, 'Expecting result after max-z projection to be 5D but got shape {}'.format(tile.shape)
        assert tile.shape[0] == tile.shape[1] == 1
        return cytometer.CytometerBase.quantify(tile, segments, **kwargs)

    def augment(self, df):
        df = cytometer.CytometerBase.augment(df, self.config.microscope_params)
        # Attempt to sum live + dead intensities if both channels are present
        for agg_fun in ['mean', 'sum']:
            cols = df.filter(regex='ci:(LIVE|DEAD):{}'.format(agg_fun)).columns.tolist()
            if len(cols) == 2:
                df['ci:LIVE+DEAD:{}'.format(agg_fun)] = df[cols[0]] + df[cols[0]]
        return df
    