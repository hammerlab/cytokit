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

def _flat_selem():
    # Return default 3D structuring element with all connections outside center plane gone
    return np.stack([
        np.zeros((3, 3), bool), 
        ndi.generate_binary_structure(2, 1), 
        np.zeros((3, 3), bool)
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
        sigmas=(1, 2, 5), include_intermediate_results=False, **kwargs):
        
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

        threshold = filters.threshold_li(img_grad)
        logger.debug('Computing gradient mask (threshold = %s)', threshold)
        img_mask = img_grad > threshold

        selem = _flat_selem()
        img_mask = ndi.binary_closing(img_mask, structure=selem, iterations=1)
        img_mask = ndi.binary_fill_holes(img_mask, structure=selem)
        img_mask = morphology.remove_small_objects(img_mask, min_size=min_object_size)
        img_mask_lab = ndi.label(img_mask)[0]

        logger.debug('Computing mask distance transform (sigma = %s)', get_sigma(sigmas[2]))
        img_dist = ndi.distance_transform_edt(img_mask, sampling=(1/z_sample_factor, 1, 1))
        img_dist = ndi.gaussian_filter(img_dist, sigma=get_sigma(sigmas[2]), mode='constant')

        logger.debug('Finding local maxima (min peak dist = %s)', min_peak_dist)
        # Compute local maxima in gradient intensity making sure to isolate min_distance filters
        # to different objects in the mask, otherwise disconnected objects close together will
        # only get one peak if that distance is < min_peak_dist
        img_pks = feature.peak_local_max(
            img_dist, min_distance=min_peak_dist, labels=img_mask_lab,
            indices=False, exclude_border=False
        )
        img_pks = ndi.label(img_pks)[0]

        logger.debug('Running watershed transform')
        img_seg = segmentation.watershed(-img_dist, img_pks, mask=img_mask).astype(np.uint16)

        # Convert (h, w) -> required (z, (cell, nucleus, ...[others]), h, w) format 
        images = ([img_seg] * 2) + ([
            exposure.rescale_intensity(img_grad, out_range='uint16').astype(np.uint16),
            img_mask.astype(np.uint16),
            img_pks.astype(np.uint16),
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
                img_seg, tgt_shape, mode='constant',
                preserve_range=True, anti_aliasing=False, clip=True
            ).astype(img_seg.dtype)
            # Transpose back to original (z, (cell, nucleus, ...[others]), h, w) format
            img_seg = np.transpose(img_seg, axes=(2, 3, 0, 1))
        
        return img_seg
    
    def quantify(self, tile, segments, sigma=None, **kwargs):
        # Compute LoG image using nucleus channel
        logger.debug('Computing LoG image (sigma = %s)', sigma)
        ch_name = self.config.cytometry_params['nuclei_channel_name']
        ch_coords = self.config.get_channel_coordinates(nuc_channel_name)
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