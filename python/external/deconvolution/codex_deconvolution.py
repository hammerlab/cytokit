"""CODEX Deconvolution

This CLI script is intended to be used to deconvolve images after CODEXProcessor
and before CODEXSegm.  This script will use the same "Experiment.json"
and "channelNames.txt" files used by other CODEX applications in the process of 
determining how many cycles/channels to expect as well as PSF parameters.

The general outline for the process executed here is:
1. Copy metadata files from both a raw input directory and an intermediate directory 
    containing results from CODEXProcessor to the configured output directory
2. Load PSF images from a provided directory containing the PSF corresponding to 
    each channel (a format for filename must be provided that includes the channel number)
3. Load all images in the CODEXProcessor output directory and run deconvolution
4. Save results to the configured output directory

For general guidelines on parameters, try:

> python codex_deconvolution.py --help
"""

import os
import logging
import numpy as np
import tensorflow as tf
from os import path as osp
from shutil import copyfile
from scipy.stats import describe
from timeit import default_timer as timer
from flowdec import restoration as fd_restoration
from flowdec import data as fd_data
from flowdec import psf as fd_psf
from argparse import ArgumentParser
from skimage.external.tifffile import imread, imsave
from deconvolution import utils
from deconvolution import config as akoya_config

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('CODEXDeconCLI')


def make_arg_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        required=True,
        metavar='RAW',
        help="Path to original data directory containing acquisitions"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        metavar='INPUT',
        help="Path to directory containing images stacks from CODEX processor"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        metavar='OUTPUT',
        help="Path to directory to contain results"
    )
    parser.add_argument(
        "--psf-dir",
        required=False,
        default=None,
        metavar='PSFDIR',
        help="Optional path to directory containing psf stacks; if not given PSFs will be generated "
        "based on experiment configuration (which is almost always what you want)"
    )
    parser.add_argument(
        "--psf-pattern",
        required=False,
        default=None,
        metavar='PSFPATTERN',
        help="Optional PSF file naming pattern; e.g. 'psf-ch{channel_id}.tif' "
        "where channel_id is 1-based index (must be given if --psf-dir is also provided)"
    )
    parser.add_argument(
        "--pad-dims",
        required=False,
        default="0,0,6",
        metavar='PADDIMS',
        help="Amount by which to pad a single z-stack as a 'x,y,z' string; e.g. '0,0,6' "
        "for no x or y padding and at least 6 units of padding in z-direction (6 units "
        "in z-direction would correspond to 3 slices on top and 3 on bottom)"
    )
    parser.add_argument(
        "--pad-mode",
        required=False,
        default="log2",
        metavar='PADMODE',
        help="Either 'log2' or 'none' indicating whether or not to stretch dimension lengths "
        "out to those optimal for FFT"
    )
    parser.add_argument(
        "--scale-factor",
        required=False,
        default=.5,
        help="Each z-stack will be multiplied by this number after matching "
        "its mean intensity with that of the original image.  "
        "One reason to do this is to minimize saturation, as seems to "
        "be the original intention in the Akoya codebase (which uses the "
        "value 1/2, which is the default value for the parameter)."
    )
    parser.add_argument(
        "--scale-mode",
        required=False,
        default='stack',
        choices=['stack', 'slice'],
        help="One of 'stack' or 'slice' indicating whether or not scaling should be "
        "applied to whole z-stacks or to individual slices"
    )
    parser.add_argument(
        "--observer-dir",
        required=False,
        default=None,
        help="Directory in which to save per-iteration images (useful for determining proper "
        "iteration counts"
    )
    parser.add_argument(
        "--observer-coords",
        required=False,
        default=None,
        help="Coordinates of single 2D images to save per-iteration views on, a feature "
        "helpful for choosing the number of iterations to use; should be specified in "
        "'<tile>,<cycle>,<channel>,<z>' format where each is a one-based index"
    )
    parser.add_argument(
        "--n-iter",
        required=False,
        default=25,
        help="Number of Richardson-Lucy iterations to execute (defaults to 25)"
    )
    parser.add_argument(
        "--dry-run",
        required=False,
        action='store_true',
        help="Flag indicating to only show inputs and proposed outputs"
    )
    return parser


def init_output(args):
    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)


def copy_meta_files(args):
    files = [
        osp.join(mf[0], mf[1]) for mf in [
            (args.raw_dir, 'Experiment.json'),
            (args.raw_dir, 'channelNames.txt'),
            (args.input_dir, 'tileMap.txt'),
        ]
    ]
    logger.debug('Metadata files copied to output:')
    for f in files:
        dest = osp.join(args.output_dir, os.path.basename(f))
        logger.debug('\t{} -> {}'.format(f, dest))
        if not args.dry_run:
            copyfile(f, dest)
    return files


def generate_psfs(args, config):
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
        ni0=utils.get_immersion_ri(config.exp_config['objectiveType']),

        # Set "particle position" in Gibson-Lannie to 0 which gives a
        # Born & Wolf kernel as a degenerate case
        pz=0.
    )

    logger.debug('Generating PSFs from experiment configuration file')
    # Specify a psf for each emission wavelength in microns (nm in akoya config)
    return [
        fd_psf.GibsonLanni(**{**args, **{'wavelength': w/1000.}}).generate()
        for w in config.exp_config['emission_wavelengths']
    ]


def load_psfs(args, config):
    psfs = []
    for i in range(config.n_channels_per_cycle()):
        f = osp.join(args.psf_dir, args.psf_pattern.format(i+1))
        if not osp.exists(f):
            raise ValueError(
                'Expected PSF file "{}" does not exist; '
                'Num channels expected to have PSFs for = {}'
                .format(f, config.n_channels_per_cycle())
            )
        psfs.append((f, imread(f)))

    logger.debug('PSF stacks loaded from directory {}:'.format(args.psf_dir))
    for f, psf in psfs:
        logger.debug('\t{} --> shape = {}'.format(f, psf.shape))
    return [psf[1] for psf in psfs]


def resolve_psfs(args, config):
    # If no PSF dir was given, generate per-channel kernels based on experiment configuration
    if args.psf_dir is None:
        if args.psf_pattern is not None:
            raise ValueError('Must not set "psf-pattern" parameter when not also setting "psf-dir"')
        return generate_psfs(args, config)
    # Otherwise, load PSFs using the path and file pattern provided
    else:
        if args.psf_pattern is None:
            raise ValueError('Must set "psf-pattern" parameter when setting "psf-dir"')
        return load_psfs(args, config)


def img_generator(files):
    for f in files:
        yield f, imread(f)

IMG_ID = dict(tile=0, channel=0, cycle=0)


def get_iteration_observer_fn(log_dir, coords):
    """Build function to be used within deconvolution process to export intermediate results"""
    logger.debug('Initializing observer dir "{}"'.format(log_dir))
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    coords = [int(v) for v in coords.split(',')]
    t_tile, t_cyc, t_ch, t_z = coords
    def _observer_fn(*args):
        global IMG_ID
        img, i = args

        # Do nothing if this is not the target part of the stack to make per-iteration
        # summaries for (a costly operation)
        if IMG_ID['tile'] != t_tile or IMG_ID['cycle'] != t_cyc or IMG_ID['channel'] != t_ch:
            return
        ldir = osp.join(log_dir, 'img-tile{:02d}-cyc{:02d}-ch{:02d}-z{:02d}'.format(
            IMG_ID['tile'], IMG_ID['cycle'], IMG_ID['channel'], t_z
        ))
        if not osp.exists(ldir):
            os.makedirs(ldir)
        lfile = osp.join(ldir, 'iter{:04d}.tif'.format(i))
        logger.debug(
            'Iteration Observer: Saving z-slice for iteration {} to file "{}"'
            .format(i, lfile)
        )
        imsave(lfile, img[t_z-1])
    return _observer_fn


def run_deconvolution(args, psfs, config):
    global IMG_ID
    files = utils.get_files(args.input_dir, '.*\.tif$')
    times = []
    mean_ratios = []
    scale_factor = float(args.scale_factor)

    # Tone down TF logging, though only the first setting below actually
    # seems to make any difference
    utils.disable_tf_logging()
    session_config = tf.ConfigProto(log_device_placement=False)

    n_iter = int(args.n_iter)
    pad_dims = np.array([int(p) for p in args.pad_dims.split(',')])
    if args.observer_dir is not None and not args.dry_run:
        if args.observer_coords is None:
            raise ValueError('Must set "observer-coords" property when using observer')
        observer_fn = get_iteration_observer_fn(args.observer_dir, args.observer_coords)
    else:
        observer_fn = None
    algo = fd_restoration.RichardsonLucyDeconvolver(
        n_dims=3, pad_mode=args.pad_mode, pad_min=pad_dims, 
        epsilon=1e-6, observer_fn=observer_fn
    ).initialize()
    
    # Stacks load as (cycles, z, channel, height, width)
    imgs = img_generator(files)
    img_dtypes = set()
    for i, (f, img) in enumerate(imgs):
        logger.debug(
            '{} tile "{}" ({} of {}) --> shape = {}, dtype = {}'
            .format(
                'Would deconvolve' if args.dry_run else 'Deconvolving', 
                f, i+1, len(files), img.shape, img.dtype
            )
        )
        img_dtypes.add(img.dtype)
        if len(img_dtypes) > 1:
            raise ValueError(
                'Image has conflicting dtype with prior images; '
                'all dtypes seen = {}'.format(list(img_dtypes))
            )
        if not np.issubdtype(img.dtype, np.unsignedinteger):
            raise ValueError(
                'Only unsigned integer images supported; '
                'type given = {}'.format(img.dtype)
            )
        if img.min() < 0:
            raise ValueError('Image to deconvolve cannot have negative values')
        
        utils.validate_stack_shape(img, config)
        ncyc, nz, nch, nh, nw = img.shape

        # Loop through each cycle and channel so that for each, a single 3D z-stack
        # can be extracted for deconvolution
        res_stack = []
        for icyc in range(ncyc):
            res_ch =[]
            for ich in range(nch):
                acq = fd_data.Acquisition(data=img[icyc,:,ich,:,:], kernel=psfs[ich])

                if args.dry_run:
                    continue
                IMG_ID = dict(tile=i + 1, channel=ich + 1, cycle=icyc + 1)

                # Results have shape (nz, nh, nw)
                start_time = timer()
                res = algo.run(acq, niter=n_iter, session_config=session_config).data
                end_time = timer()
                times.append({'cycle': icyc+1, 'channel': ich+1, 'time': end_time - start_time})

                # This is a transformation used in the Nolanlab code to rescale means
                # of deconvolution results back to the original (they're not usually
                # off by much though).  scale_factor is then a tunable way to lower or 
                # raise the intensity values so that when clipping to uint type (with
                # no scaling) there is less saturation
                if args.scale_mode == 'stack':
                    mean_ratio = acq.data.mean() / utils.arr_to_uint(res, img.dtype).mean()
                    mean_ratios.append({'cycle': icyc+1, 'channel': ich+1, 'ratio': mean_ratio})
                    res *= (mean_ratio * scale_factor)
                elif args.scale_mode == 'slice':
                    for iz in range(nz):
                        mean_ratio = acq.data[iz].mean() / utils.arr_to_uint(res[iz], img.dtype).mean()
                        mean_ratios.append({'cycle': icyc+1, 'channel': ich+1, 'ratio': mean_ratio, 'z': iz+1})
                        res[iz] = res[iz] * (mean_ratio * scale_factor)
                else:
                    raise ValueError('Scale mode "{}" not valid'.format(args.scale_mode))

                # Clip float32 and convert to type of original image (i.e. w/ no scaling)
                res = utils.arr_to_uint(res, img.dtype)

                res_ch.append(res)

            if args.dry_run:
                continue

            # Stack (nz, nh, nw) results to (nz, nch, nh, nw)
            res_ch = np.stack(res_ch, 1)

            if list(res_ch.shape) != [nz, nch, nh, nw]:
                raise ValueError(
                    'Stack across channels has wrong shape --> expected = {}, actual = {}'
                    .format([nz, nch, nh, nw], list(res_ch.shape))
                )
            res_stack.append(res_ch)
 
        if args.dry_run:
            continue

        # Stack (nz, nch, nh, nw) results along first axis to match input
        # like (ncyc, nz, nch, nh, nw)
        res_stack = np.stack(res_stack, 0)

        # Validate resulting data type
        if res_stack.dtype != img.dtype:
            raise ValueError(
                'Final stack has wrong dtype --> expected = {}, actual = {}'
                .format(img.dtype, res_stack.dtype)
            )

        # Validate resulting shape matches the input
        if list(res_stack.shape) != list(img.shape):
            raise ValueError(
                'Final stack has wrong shape --> expected = {}, actual = {}'
                .format(list(img.shape), list(res_stack.shape))
            )

        res_file = osp.join(args.output_dir, osp.basename(f))
        logger.debug(
            'Saving deconvolved tile to "{}" --> shape = {}, dtype = {}'
            .format(res_file, res_stack.shape, res_stack.dtype)
        )
        # See tiffwriter docs at http://scikit-image.org/docs/dev/api/skimage.external.tifffile
        # .html#skimage.external.tifffile.TiffWriter for more info on how scikit-image
        # handles imagej formatting -- the docs aren't very explicit but they do mention
        # that with 'imagej=True' it can handle arrays up to 6 dims in TZCYXS order
        imsave(res_file, res_stack, imagej=True)
    return times, mean_ratios


if __name__ == "__main__":
    # Parse arguments
    parser = make_arg_parser()
    args = parser.parse_args()

    logger.info('Beginning Stack Deconvolution')
    logger.debug('Arguments:')
    for arg in [
        'input_dir', 'output_dir', 'psf_dir', 'psf_pattern', 
        'pad_dims', 'pad_mode', 'scale_factor', 'scale_mode', 'observer_dir',
        'observer_coords', 'n_iter', 'dry_run']:
        logger.debug('\t{}="{}"'.format(arg, getattr(args, arg)))
 
    
    logger.info('Initializing output directory')
    init_output(args)

    logger.info('Initializing metadata files')
    copy_meta_files(args)

    logger.info('Loading experiment configuration')
    config = akoya_config.load_config(args)

    logger.info('Initializing PSF volumes')
    psfs = resolve_psfs(args, config)

    logger.info('Running deconvolution')
    times, mean_ratios = run_deconvolution(args, psfs, config)

    logger.info('Deconvolution Complete')
    logger.info('-'*30)
    logger.info('Execution Time Summary: {}'.format(
        describe([t['time'] for t in times])
    ))
    logger.info('Z-Stack Mean Ratio Summary: {}'.format(
        describe([m['ratio'] for m in mean_ratios])
    ))
