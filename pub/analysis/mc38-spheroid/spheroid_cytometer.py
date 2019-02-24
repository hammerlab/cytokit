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
    ]).astype(bool)

def get_boundaries(label_img):
    assert label_img.ndim == 3, 'Expecting 3D volume but got image with shape {}'.format(label_img.shape)
    return np.stack([
        label_img[i] * segmentation.find_boundaries(label_img[i], mode='inner', background=0)
        for i in range(label_img.shape[0])
    ])

class SpheroidCytometer(cytometer.Cytometer):
    
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
            rz = config.microscope_params.res_axial_nm
            rxy = config.microscope_params.res_lateral_nm
            # Relative voxel sizes (typically around [10, 1, 1])
            self.sampling = np.array([rz / rxy, 1.0, 1.0])
            
        # Multiplicative factors for rescaling isotropic arguments
        self.factors = 1.0 / self.sampling
        logger.debug('Cytometer initialized with sampling rates %s', tuple(self.sampling))
    
    def initialize(self):
        pass
        
    def segment(
        self, img, rescaling=None, min_peak_dist=100, max_peaks=75,
        sigmas=(1, 3, 2), basin_type='dist', include_intermediate_results=False, **kwargs):
        
        # Validate arguments
        assert img.dtype in [np.uint8, np.uint16], \
            'Expecting 8 or 16 bit image but got type %s' % img.dtype
        assert img.ndim == 3, \
            'Expecting 3D image, got shape %s' % img.shape
        assert len(sigmas) == 3, \
            'Sigmas must be 3 element sequence'
        assert basin_type in ['diff', 'dist', 'inverse'], \
            'Basin type must be one of "diff", "dist", "inverse" (not {})'.format(basin_type)
        
        if rescaling is None:
            rescaling = (1, 1, 1)
        if np.isscalar(rescaling):
            rescaling = (1, float(rescaling), float(rescaling))
        assert len(rescaling) == 3, \
            'Rescaling factors must be 3 item list-like'
        assert all([0 < r <= 1 for r in rescaling]), \
            'All rescaling factors must be in (0, 1]'
        shape = img.shape
        
        logger.debug('Rescaling sigma values with sampling factors = %s, resize factors = %s', tuple(self.factors), rescaling)
        sigmas = [
            tuple(self.factors * np.array(rescaling) * np.array([s]*len(self.factors))) 
            for s in sigmas
        ]
        
        if any([r < 1 for r in rescaling]):
            logger.debug('Reducing image XY dimensions by factor %s', rescaling)
            img = transform.rescale(
                img, rescaling, 
                multichannel=False, preserve_range=True, 
                anti_aliasing=True, clip=True, mode='reflect'
            ).astype(img.dtype)
        
        # Run small XY median filter to remove outliers (and convert to float)
        img = ndi.median_filter(img, size=(1, 3, 3))
        img = img_as_float(img)

        logger.debug('Running high pass filter (sigma = %s)', sigmas[0])
        img_hp = img - ndi.gaussian_filter(img, sigma=sigmas[0], mode='reflect')
        img_hp = exposure.rescale_intensity(img_hp, out_range=(0, 1))
        
        logger.debug('Computing gradients (sigma = %s)', sigmas[1])
        img_grad = ndi.generic_gradient_magnitude(img_hp, ndi.sobel)
        img_grad = ndi.gaussian_filter(img_grad, sigma=sigmas[1], mode='constant')
        img_grad = exposure.rescale_intensity(img_grad, out_range=(0, 1))

        threshold = filters.threshold_otsu(img_grad)
        logger.debug('Computing gradient mask (threshold = %s)', threshold)
        img_mask = img_grad > threshold

        logger.debug('Applying morphological smoothings to mask')
        img_mask = ndi.binary_fill_holes(img_mask, structure=flat_disk(1))
        img_border_mask = morphology.binary_closing(img_mask, selem=flat_ball(1, 0))
        # Open to give objects enclosing highly smoothed centers of objects (break into erosion + dilation
        # and fetch intermediate result to work with eroded centers)
        img_marker_mask = morphology.binary_opening(img_border_mask, selem=flat_ball(15, 4))
        img_marker_mask = img_marker_mask & img_border_mask
        img_marker_labl = morphology.label(img_marker_mask)

        logger.debug('Computing mask distance transform (sigma = %s, sampling = %s)', sigmas[2], tuple(self.sampling))
        img_dist = ndi.distance_transform_edt(img_marker_mask, sampling=self.sampling)
        img_dist = ndi.gaussian_filter(img_dist, sigma=sigmas[2], mode='constant')

        logger.debug('Finding local maxima (min peak dist = %s, max peaks = %s)', min_peak_dist, max_peaks)
        # Compute local maxima in gradient intensity making sure to isolate min_distance filters
        # to different objects in the mask, otherwise disconnected objects close together will
        # only get one peak if that distance is < min_peak_dist
        img_pks = feature.peak_local_max(
            img_dist, min_distance=min_peak_dist, labels=img_marker_labl,
            indices=False, exclude_border=False,
            num_peaks=max_peaks
        )
        img_pks, num_pks = morphology.label(img_pks, return_num=True)
        

        logger.debug('Running watershed transform (sampling = %s, num peaks = %s)', tuple(self.sampling), num_pks)
        # img_seg = segmentation.random_walker(img_grad, img_pks - (~img_border_mask).astype(img_pks.dtype)).astype(np.uint16)
        
        # If "inverting" the distance basin, use distance to nearest peak as basin (aka delaunay triangulation distance); 
        # otherwise, use distance to nearest background from markers
        if basin_type == 'dist':
            img_basin = -img_dist
        elif basin_type == 'inverse':
            img_basin = ndi.distance_transform_edt(~(img_pks>0), sampling=self.sampling)
        elif basin_type == 'diff':
            img_basin = ndi.distance_transform_edt(~(img_pks>0), sampling=self.sampling) - img_dist
        else:
            raise ValueError('Basin type "{}" not valid'.format(basin_type))
        img_seg = segmentation.watershed(img_basin, img_pks, mask=img_border_mask).astype(np.uint16)
        
        # Compute boundaries
        # img_center_lbl = img_center_mask.astype(np.uint16) * img_seg
        img_seg_bnd = get_boundaries(img_seg)
        assert img_seg_bnd.dtype == img_seg.dtype
        assert img_seg_bnd.shape == img_seg.shape

        # Convert (h, w) -> required (z, (cell_mask, nucleus_mask, cell_boundary, nucleus_boundary, ...[others]), h, w) format 
        img_seg = ([img_seg, img_seg, img_seg_bnd, img_seg_bnd]) + ([
                exposure.rescale_intensity(img_grad, out_range='uint16').astype(np.uint16),
                exposure.rescale_intensity(img_dist, out_range='uint16').astype(np.uint16),
                exposure.rescale_intensity(img_basin, out_range='uint16').astype(np.uint16),
                img_marker_labl.astype(np.uint16),
                img_pks.astype(np.uint16)
            ] if include_intermediate_results else [])
        img_seg = np.stack(img_seg, axis=1)
        assert img_seg.dtype == np.uint16
        assert img_seg.ndim == 4, 'Expecting 4D result, got shape {}'.format(img_seg.shape)
        
        # Rescale back to original size if necessary
        if any([r < 1 for r in rescaling]):
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
    
    def quantify(self, tile, segments, sigma=1, **kwargs):
        sigma = tuple(self.factors * np.array([sigma]*len(self.factors))) 
        
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