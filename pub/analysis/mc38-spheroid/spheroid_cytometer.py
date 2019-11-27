import numpy as np
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
        
        
        img_dt = ndi.distance_transform_edt(img_pm)
        
        # Use propogation rather than watershed as it often captures a much more accurate boundary
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
    
    
###################
# 2X Implementation
###################
        

def get_circle_mask(radius, shape, center=None, translation=None):
    center = np.asarray(shape)//2 if center is None else np.asarray(center)
    if translation is not None:
        center += np.asarray(translation).astype(int)
    rr, cc = draw.circle(*center, radius=radius, shape=shape)
    arr = np.zeros(shape, dtype=bool)
    arr[rr, cc] = 1
    return arr.astype(bool)

class SpheroidCytometer2x(cytometer.Cytometer):

    def segment(self, img, well_radius=800, well_mask_radius=765, include_intermediate_results=False, **kwargs):
        # Assume image is single plane z-stack and grab first 2D image to process
        assert img.ndim == 3
        assert img.shape[0] == 1
        img = img[0]
        
        logger.debug(
            'Running 2x segmentation on image with shape %s, type %s (args: well_radius = %s, well_mask_radius = %s, include_intermediate_results=%s)',
            img.shape, img.dtype, well_radius, well_mask_radius, include_intermediate_results
        )

        # Remove outliers, convert to float
        img = ndi.median_filter(img, size=(3, 3))
        img = img_as_float(img)

        # Apply bandpass and compute gradients
        img_bp = ndi.gaussian_filter(img, sigma=6) - ndi.gaussian_filter(img, sigma=10)
        img_gr = ndi.generic_gradient_magnitude(img_bp, ndi.sobel)

        # Get and apply well mask translation
        img_well = get_circle_mask(well_radius, img_gr.shape)
        shifts = feature.register_translation(img_gr, img_well)[0]
        img_well = get_circle_mask(well_mask_radius, img_gr.shape, translation=shifts)
        img_gm = img_gr * img_well

        # Apply local threshold and cleanup binary result
        img_bm = img_gm > filters.threshold_local(img_gm, 255)
        img_bm = ndi.binary_fill_holes(img_bm, structure=morphology.disk(1))
        img_bm = morphology.binary_opening(img_bm, selem=morphology.disk(8))

        # Run segmentation
        img_dt = ndi.distance_transform_edt(img_bm)
        img_dt = ndi.gaussian_filter(img_dt, sigma=1)
        img_pk = morphology.label(feature.peak_local_max(img_dt, indices=False, min_distance=8))
        img_obj = segmentation.watershed(-img_dt, img_pk, mask=img_bm).astype(np.uint16)
        img_bnd = img_obj * segmentation.find_boundaries(img_obj, mode='inner', background=0)

        # Compile list of object image results (and append intermediates if necessary)
        img_seg = [img_obj, img_obj, img_bnd, img_bnd]
        if include_intermediate_results:
            to_uint16 = lambda im: exposure.rescale_intensity(im, out_range='uint16').astype(np.uint16)
            img_seg += [
                to_uint16(img_bp), 
                segmentation.find_boundaries(img_well, mode='inner', background=0).astype(np.uint16),
                to_uint16(img_gm),
                to_uint16(img_dt), 
                img_pk.astype(np.uint16)
            ]
            
        # Stack and add new axis to give to (z, ch, h, w)
        img_seg = np.stack(img_seg)[np.newaxis]
        assert img_seg.dtype == np.uint16, 'Expecting 16bit result, got type {}'.format(img_seg.dtype)
        assert img_seg.ndim == 4, 'Expecting 4D result, got shape {}'.format(img_seg.shape)
        return img_seg
        
    def quantify(self, tile, segments, **kwargs):
        return cytometer.CytometerBase.quantify(tile, segments, **kwargs)
    
    def augment(self, df):
        df = cytometer.CytometerBase.augment(df, self.config.microscope_params)
        # Attempt to sum live + dead intensities if both channels are present
        for agg_fun in ['mean', 'sum']:
            cols = df.filter(regex='ci:(LIVE|DEAD):{}'.format(agg_fun)).columns.tolist()
            if len(cols) == 2:
                df['ci:LIVE+DEAD:{}'.format(agg_fun)] = df[cols[0]] + df[cols[0]]
        return df
    
    
###################
# Utility Functions
###################

def blob_dog(image, min_sigma=1, max_sigma=50, sigma_ratio=1.6, threshold=2.0,
             overlap=.5, *, exclude_border=False):
    r"""Lift from https://github.com/scikit-image/scikit-image/blob/2962429237988cb60b9b317aa020ca3bab100b7f/skimage/feature/blob.py#L168
    
    Modifications are added here to return more intermediate results
    """
    image = img_as_float(image)

    # if both min and max sigma are scalar, function returns only one sigma
    scalar_sigma = np.isscalar(max_sigma) and np.isscalar(min_sigma)

    # Gaussian filter requires that sequence-type sigmas have same
    # dimensionality as image. This broadcasts scalar kernels
    if np.isscalar(max_sigma):
        max_sigma = np.full(image.ndim, max_sigma, dtype=float)
    if np.isscalar(min_sigma):
        min_sigma = np.full(image.ndim, min_sigma, dtype=float)

    # Convert sequence types to array
    min_sigma = np.asarray(min_sigma, dtype=float)
    max_sigma = np.asarray(max_sigma, dtype=float)

    # k such that min_sigma*(sigma_ratio**k) > max_sigma
    k = int(np.mean(np.log(max_sigma / min_sigma) / np.log(sigma_ratio) + 1))

    # a geometric progression of standard deviations for gaussian kernels
    sigma_list = np.array([min_sigma * (sigma_ratio ** i)
                           for i in range(k + 1)])

    gaussian_images = [ndi.gaussian_filter(image, s) for s in sigma_list]

    # computing difference between two successive Gaussian blurred images
    # multiplying with average standard deviation provides scale invariance
    dog_images = [(gaussian_images[i] - gaussian_images[i + 1])
                  * np.mean(sigma_list[i]) for i in range(k)]

    image_cube = np.stack(dog_images, axis=-1)

    # local_maxima = get_local_maxima(image_cube, threshold)
    local_maxima = feature.peak_local_max(image_cube, threshold_abs=threshold,
                                  footprint=np.ones((3,) * (image.ndim + 1)),
                                  threshold_rel=0.0,
                                  exclude_border=exclude_border)
    # Catch no peaks
    if local_maxima.size == 0:
        return np.empty((0, 3)), dog_images, sigma_list

    # Convert local_maxima to float64
    lm = local_maxima.astype(np.float64)

    # translate final column of lm, which contains the index of the
    # sigma that produced the maximum intensity value, into the sigma
    sigmas_of_peaks = sigma_list[local_maxima[:, -1]]

    if scalar_sigma:
        # select one sigma column, keeping dimension
        sigmas_of_peaks = sigmas_of_peaks[:, 0:1]

    # Remove sigma index and replace with sigmas
    lm = np.hstack([lm[:, :-1], sigmas_of_peaks])

    # See: https://github.com/scikit-image/scikit-image/blob/master/skimage/feature/blob.py#L129
    #return lm, _prune_blobs(lm, overlap), sigma_list, dog_images
    return _prune_blobs(lm, overlap), dog_images, sigma_list