import numpy as np
from skimage import filters
from skimage import measure
from skimage import exposure
from skimage import transform
from skimage import feature
from skimage import morphology
from skimage import segmentation
from skimage import img_as_float
from cytokit import math as ck_math
from cytokit.cytometry import cytometer
from scipy import ndimage as ndi
import logging
logger = logging.getLogger(__name__)

def flat_disk(radius):
    # Return default 3D structuring element with all connections outside center plane gone
    n = 2*radius + 1
    return np.stack([
        np.zeros((n, n), bool), 
        morphology.disk(radius).astype(bool), 
        np.zeros((n, n), bool)
    ])

def flat_square(size):
    return np.stack([
        np.zeros((size, size), bool), 
        morphology.square(size).astype(bool), 
        np.zeros((size, size), bool)
    ])

def flat_ball(r, rz):
    return np.stack([
        np.pad(morphology.disk(rz), r-rz, 'constant'),
        morphology.disk(r),
        np.pad(morphology.disk(rz), r-rz, 'constant'),
    ])

class SpheroidCytometer(cytometer.Cytometer):
    
    def __init__(self, config):
        """Cytometer Initialization

        Args:
        """
        super().__init__(config)
    
    def initialize(self):
        pass
        
    def segment(
        self, img, rescale_factor=None, z_sample_factor=None, 
        min_object_size=1024, min_peak_dist=200,
        sigmas=(1, 2, 5), include_intermediate_results=False, max_peaks=75, **kwargs):
        
        # Validate arguments
        assert img.dtype in [np.uint8, np.uint16], \
            'Expecting 8 or 16 bit image but got type %s' % img.dtype
        assert img.ndim == 3, \
            'Expecting 3D image, got shape %s' % img.shape
        assert len(sigmas) == 3, \
            'Sigmas must be 3 element sequence'
        if z_sample_factor is None:
            z_sample_factor = 1.
        assert 0 < z_sample_factor <= 1, \
            'Z-sampling factor must be in (0, 1]'
        if rescale_factor is None:
            rescale_factor = 1
        assert 0 < rescale_factor <= 1, \
            'XY rescale factor must be in (0, 1]'
        shape = img.shape
        
        if 0 < rescale_factor < 1:
            logger.debug('Reducing image XY dimensions by factor %s', rescale_factor)
            img = transform.rescale(
                img, (1, rescale_factor, rescale_factor), 
                multichannel=False, preserve_range=True, 
                anti_aliasing=True, mode='reflect'
            ).astype(img.dtype)
        
        # Run small XY median filter to remove outliers (and convert to float)
        img = ndi.median_filter(img, size=(1, 3, 3))
        img = img_as_float(img)
        
        
        def get_sigma(s):
            return (z_sample_factor*s, s, s)
        
        #img_log = ndi.gaussian_laplace(img, sigma=get_sigma(sigmas[0]), mode='reflect')

        logger.debug('Running high pass filter (sigma = %s)', get_sigma(sigmas[0]))
        img_hp = img - ndi.gaussian_filter(img, sigma=get_sigma(sigmas[0]), mode='reflect')
        img_hp = exposure.rescale_intensity(img_hp, out_range=(0, 1))
        
        logger.debug('Computing gradients (sigma = %s)', get_sigma(sigmas[1]))
        img_grad = ndi.generic_gradient_magnitude(img_hp, ndi.sobel)
        img_grad = ndi.gaussian_filter(img_grad, sigma=get_sigma(sigmas[1]), mode='constant')
        img_grad = exposure.rescale_intensity(img_grad, out_range=(0, 1))

        threshold = filters.threshold_otsu(img_grad)
        logger.debug('Computing gradient mask (threshold = %s)', threshold)
        img_mask = img_grad > threshold

        # Make sure to transform mask only after running distance transorm
        logger.debug('Applying morphological smoothings to mask')
        img_mask = ndi.binary_fill_holes(img_mask, structure=flat_disk(1))
        img_border_mask = morphology.binary_closing(img_mask, selem=ndi.generate_binary_structure(3, 1))
        img_marker_mask = morphology.binary_erosion(img_border_mask, selem=flat_ball(15, 4))
        img_marker_mask = img_marker_mask & img_border_mask
        img_center_mask = morphology.binary_dilation(img_marker_mask, selem=flat_ball(15, 4))
        img_marker_mask = img_center_mask & img_border_mask
        
        img_marker_labl = morphology.label(img_marker_mask)

        logger.debug('Computing mask distance transform (sigma = %s)', get_sigma(sigmas[2]))
        img_dist = ndi.distance_transform_edt(img_marker_mask, sampling=(1/z_sample_factor, 1, 1))
        img_dist = ndi.gaussian_filter(img_dist, sigma=get_sigma(sigmas[2]), mode='constant')

        logger.debug('Finding local maxima (min peak dist = %s)', min_peak_dist)
        # Compute local maxima in gradient intensity making sure to isolate min_distance filters
        # to different objects in the mask, otherwise disconnected objects close together will
        # only get one peak if that distance is < min_peak_dist
        img_pks = feature.peak_local_max(
            img_dist, min_distance=min_peak_dist, labels=img_marker_labl,
            indices=False, exclude_border=False,
            num_peaks=max_peaks
        )
        img_pks = morphology.label(img_pks)
        

        logger.debug('Running watershed transform')
        #img_seg = segmentation.random_walker(img_grad, img_pks - (~img_border_mask).astype(img_pks.dtype)).astype(np.uint16)
        #img_seg = segmentation.watershed(-img_dist, img_pks, mask=img_border_mask).astype(np.uint16)
        img_basin = ndi.distance_transform_edt(~(img_pks>0), sampling=(1/z_sample_factor, 1, 1))
        img_seg = segmentation.watershed(img_basin, img_pks, mask=img_border_mask).astype(np.uint16)
        img_center_lbl = img_center_mask.astype(np.uint16) * img_seg

        # Convert (h, w) -> required (z, (cell, nucleus, ...[others]), h, w) format 
        images = ([img_seg, img_center_lbl]) + ([
            exposure.rescale_intensity(img_grad, out_range='uint16').astype(np.uint16),
            img_marker_labl.astype(np.uint16),
            exposure.rescale_intensity(img_dist, out_range='uint16').astype(np.uint16),
            exposure.rescale_intensity(img_basin, out_range='uint16').astype(np.uint16)
            #img_pks.astype(np.uint16),
        ] if include_intermediate_results else [])
        img_seg = np.stack(images, 1)
        assert img_seg.dtype == np.uint16
        assert img_seg.ndim == 4, 'Expecting 4D result, got shape {}'.format(img_seg.shape)
        
        # Rescale back to original size if necessary
        if 0 < rescale_factor < 1:
            # Transpose to (h, w, ...) since resize will operation on batches like this
            img_seg = np.transpose(img_seg, axes=(2, 3, 0, 1))
            tgt_shape = shape[-2:] + img_seg.shape[2:]
            logger.debug('Rescaling results from shape %s back to %s', img_seg.shape, tgt_shape)
            assert len(tgt_shape) == img_seg.ndim
            img_seg = transform.resize(
                img_seg, tgt_shape, mode='constant', order=0,
                preserve_range=True, anti_aliasing=False, clip=True
            ).astype(img_seg.dtype)
            # Transpose back to original (z, (cell, nucleus, ...[others]), h, w) format
            img_seg = np.transpose(img_seg, axes=(2, 3, 0, 1))
        
        return img_seg
    
    def quantify(self, tile, segments, sigma=None, **kwargs):
        # TODO: Fix this to use rescaling factors
        sigma = sigma if sigma is not None else (.1, 1, 1)
        
        # Compute LoG image using nucleus channel
        ch_name = self.config.cytometry_params['nuclei_channel_name']
        ch_coords = self.config.get_channel_coordinates(ch_name)
        logger.debug('Computing LoG image on channel "%s" (sigma = %s)', ch_name, sigma)
        img = img_as_float(tile[ch_coords[0], :, ch_coords[1]])
        img = ndi.gaussian_laplace(img, sigma=sigma, mode='reflect')
        img = exposure.rescale_intensity(img, out_range=str(tile.dtype)).astype(tile.dtype)
        
        logger.debug('Appending LoG image and running quantification')
        # Stack LoG image onto tile as new channel (at end of array)
        assert img.ndim == 3
        assert img.dtype == tile.dtype
        # Stack (1, z, 1, h, w) onto tile (cyc, z, ch, h, w)
        tile = np.append(tile, img[np.newaxis, :, np.newaxis, :, :], axis=2)
        assert tile.ndim == 5
        if 'channel_names' in kwargs:
            kwargs['channel_names'] = kwargs['channel_names'] + ['laplofgauss']
        return cytometer.Base2D.quantify(tile, segments, **kwargs)

    def augment(self, df):
        df = cytometer.Base2D.augment(df, self.config.microscope_params)
        # Attempt to sum live + dead intensities if both channels are present
        for agg_fun in ['mean', 'sum']:
            cols = df.filter(regex='ci:(LIVE|DEAD):{}'.format(agg_fun)).columns.tolist()
            if len(cols) == 2:
                df['ci:LIVE+DEAD:{}'.format(agg_fun)] = df[cols[0]] + df[cols[0]]
        return df