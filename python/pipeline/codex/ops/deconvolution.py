from codex.ops.op import CodexOp, get_tf_config
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
    args = dict(
        # Set psf dimensions to match volumes
        size_x=config.exp_config['tile_width'],
        size_y=config.exp_config['tile_height'],
        size_z=config.exp_config['num_z_planes'],

        # Magnification factor
        m=config.exp_config['magnification'],

        # Numerical aperture
        na=config.exp_config['numerical_aperture'],

        # Axial resolution in microns (nm in akoya config)
        res_axial=config.exp_config['z_pitch'] / 1000.,

        # Lateral resolution in microns (nm in akoya config)
        res_lateral=config.exp_config['per_pixel_XY_resolution'] / 1000.,

        # Immersion refractive index
        ni0=get_immersion_ri(config.exp_config['objectiveType']),

        # Set "particle position" in Gibson-Lannie to 0 which gives a
        # Born & Wolf kernel as a degenerate case
        pz=0.
    )

    logger.debug('Generating PSFs from experiment configuration file')
    # Specify a psf for each emission wavelength in microns (nm in codex config)
    return [
        fd_psf.GibsonLanni(**{**args, **{'wavelength': w/1000.}}).generate()
        for w in config.exp_config['emission_wavelengths']
    ]


class CodexDeconvolution(CodexOp):

    def __init__(self, config, n_iter=25):
        super(CodexDeconvolution, self).__init__(config)
        self.n_iter = n_iter
        self.algo = None

    def initialize(self):
        self.algo = fd_restoration.RichardsonLucyDeconvolver(n_dims=3).initialize()
        return self

    def run(self, tile):
        # Tile should have shape (cycles, z, channel, height, width)
        ncyc, nw, nh, nz, nch = self.config.tile_dims()

        psfs = generate_psfs(self.config)
        img_cyc = []
        for icyc in range(ncyc):
            img_ch = []
            for ich in range(nch):
                logger.debug('Running deconvolution for cycle {}, channel {}'.format(icyc, ich))
                acq = fd_data.Acquisition(tile[icyc, :, ich, :, :], kernel=psfs[ich])
                res = self.algo.run(acq, self.n_iter, session_config=get_tf_config()).data
                img_ch.append(res)
            img_cyc.append(np.stack(img_ch, 1))
        return np.stack(img_cyc, 0)
