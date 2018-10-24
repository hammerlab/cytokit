from cytokit.ops.op import CytokitOp, get_tf_config
from cytokit.utils import np_utils
from flowdec import restoration as fd_restoration
from flowdec import psf as fd_psf
from flowdec import data as fd_data
import numpy as np
import logging

logger = logging.getLogger(__name__)


def get_immersion_ri(immersion):
    """Get refractive index for an immersion type"""
    if immersion == 'air':
        return 1.0
    elif immersion == 'water':
        return 1.33
    elif immersion == 'oil':
        return 1.5115
    else:
        raise ValueError('Immersion "{}" is not valid (must be air, water, or oil)'.format(immersion))


def generate_psfs(config):
    mag, na, res_axial_nm, res_lateral_nm, objective_type, em_wavelength_nm = config.microscope_params
    args = dict(
        # Set psf dimensions to match volumes
        size_x=config.tile_width,
        size_y=config.tile_height,
        size_z=config.n_z_planes,

        # Magnification factor
        m=mag,

        # Numerical aperture
        na=na,

        # Axial resolution in microns (nm in akoya config)
        res_axial=res_axial_nm / 1000.,

        # Lateral resolution in microns (nm in akoya config)
        res_lateral=res_lateral_nm / 1000.,

        # Immersion refractive index
        ni0=get_immersion_ri(objective_type),

        # Set "particle position" in Gibson-Lannie to 0 which gives a
        # Born & Wolf kernel as a degenerate case
        pz=0.,

        # Lower this parameter as it was shown to produce results more
        # closely resembling PSFGenerator, especially for higher energy light
        min_wavelength=0.35
    )

    logger.debug('Generating PSFs from experiment configuration file')
    # Specify a psf for each emission wavelength in microns (nm in cytokit_app config)
    return [
        fd_psf.GibsonLanni(**{**args, **{'wavelength': w/1000.}}).generate()
        for w in em_wavelength_nm
    ]


def rescale_stack(tile, stack, scale_factor):
    """Restore mean intensity of z-stack

    This is transformation used in the Nolanlab code to rescale means
    of deconvolution results back to the original (they're not usually
    off by much though).  scale_factor is then a tunable way to lower or
    raise the intensity values so that when clipping to uint type (with
    no scaling) there is less saturation.
    """
    mean_ratio = tile.mean() / np_utils.arr_to_uint(stack, tile.dtype).mean()
    logger.debug('Mean ratio of original stack to deconvolved stack = {}'.format(mean_ratio))
    return stack * (mean_ratio * scale_factor), mean_ratio


class CytokitDeconvolution(CytokitOp):

    def __init__(self, config, n_iter=25, scale_factor=.5):
        super().__init__(config)

        params = config.deconvolution_params
        self.n_iter = params.get('n_iter', n_iter)
        self.scale_factor = params.get('scale_factor', scale_factor)
        self.algo = None
        self.psfs = None

    def initialize(self):
        self.algo = fd_restoration.RichardsonLucyDeconvolver(n_dims=3).initialize()
        self.psfs = generate_psfs(self.config)
        return self

    def _run(self, tile, **kwargs):
        if not np.issubdtype(tile.dtype, np.unsignedinteger):
            raise ValueError(
                'Only unsigned integer images supported; '
                'type given = {}'.format(tile.dtype)
            )

        # Tile should have shape (cycles, z, channel, height, width)
        dims = self.config.tile_dims
        if dims != tile.shape:
            raise AssertionError('Given tile with shape {} does not match expected shape {}'.format(tile.shape, dims))
        ncyc, nz, nch, nh, nw = dims

        img_cyc = []
        for icyc in range(ncyc):
            img_ch = []
            for ich in range(nch):
                acq = fd_data.Acquisition(tile[icyc, :, ich, :, :], kernel=self.psfs[ich])
                logger.debug('Running deconvolution for cycle {}, channel {} [dtype = {}]'.format(icyc, ich, acq.data.dtype))
                res = self.algo.run(acq, self.n_iter, session_config=get_tf_config(self)).data

                # Restore mean intensity if a scale factor was given
                if self.scale_factor is not None:
                    res, mean_ratio = rescale_stack(acq.data, res, self.scale_factor)
                    self.record({'mean_ratio': mean_ratio, 'cycle': icyc, 'channel': ich})

                # Clip float32 and convert to type of original image (i.e. w/ no scaling)
                res = np_utils.arr_to_uint(res, acq.data.dtype)

                img_ch.append(res)
            img_cyc.append(np.stack(img_ch, 1))
        return np.stack(img_cyc, 0)
