"""Montage Image Generation Utilities"""
from codex import io as codex_io
from codex.ops import tile_crop
import numpy as np
import os.path as osp

def get_tile_montage(config, image_dir, hyperstack, icyc=0, iz=0, ich=0, ireg=0, bw=0, bv_fn=None, allow_missing=False,
                     imread_fn=None):
    """Generate a montage image for a specific cycle, z-plane, channel, and region

    This function supports both raw, flattened 2D images as well as consolidated, 5D 
    hyperstacks (as determined by `hyperstack` argument)
    Args:
        config: Experiment configuration
        image_dir: Location of tiled images; These should include all z-planes, cycles, and channels in 
            individual tif files (e.g. the output of the pre-processing or segmentation pipelines)
        hyperstack: Flag indicating whether or not images are 5D hyperstacks or flattened 2D images:
            - Hyperstacks are typically results from any sort of processing or segmentation step
            - Flattened 2D images are typically raw files generated directly from a microscope
        icyc: 0-based cycle index
        iz: 0-based z-plane index
        ich: 0-based channel index
        ireg: 0-based region index
        bw: Border width (in pixels) to add to each tile in the montage image, which useful for determining
            tile location within the montage; If <= 0, this parameter will do nothing
        bv_fn: Border value function with signature `fn(tile_x, tile_y) --> float`; if not given all
            border values are assigned a value of 0
        allow_missing: Flag indicating whether or not to allow missing tiles into the montage; defaults
            to false and is generally only useful when debugging missing data
        imread_fn: When not using 5D hyperstacks (i.e. reading raw image files) this can be useful for cases when,
            for example, raw, single-channel files are actually 3 channel files with the first two channels blank
            (this happens w/ Keyence somehow).  This function will take an image path and must return a single 2D
            image with shape (rows, cols)
    Returns:
        A (usually very large) 2D array containing all tiles stitched together
    """
    tile_indexes = list(range(config.n_tiles_per_region))
    tw, th = config.tile_width, config.tile_height
    tiles = []
    for itile in tile_indexes:
        tx, ty = config.get_tile_coordinates(itile)

        # If operating on a hyperstack, extract the appropriate slice to add to the montage
        if hyperstack:
            path = codex_io.get_processor_img_path(ireg, tx, ty)
            path = osp.join(image_dir, path)
            if not osp.exists(path) and allow_missing:
                tile = np.zeros((th, tw))
            else:
                tile = codex_io.read_tile(path, config)
            tile = tile[icyc, iz, ich, :, :]
        # Otherwise, assume raw acquisition files are to be loaded and then cropped before being added
        else:
            path = codex_io.get_raw_img_path(ireg, itile, icyc, ich, iz)
            path = osp.join(image_dir, path)
            if not osp.exists(path) and allow_missing:
                tile = np.zeros((th, tw))
            else:
                tile = codex_io.read_image(path) if imread_fn is None else imread_fn(path)
                if tile.ndim != 2:
                    raise ValueError(
                        'Expecting 2D image at path "{}" but shape found is {}.  Consider using the '
                        '`imread_fn` argument to specify a custom function to open files or if already using it, '
                        'make sure that results are 2D'
                        .format(path, tile.shape)
                    )
                tile = tile_crop.apply_slice(tile, tile_crop.get_slice(config))
        
        # Highlight borders, if configured to do so
        if bw > 0:
            bv = 0 if bv_fn is None else bv_fn(tx, ty)
            tile[0:bw, :] = bv
            tile[-bw:, :] = bv
            tile[:, 0:bw] = bv
            tile[:, -bw:] = bv
        
        # Add to montage
        tiles.append(tile)

    return montage(tiles, config)


def montage(tiles, config):
    """Montage a list of tile images

    Args:
        tiles: A list of images having at least 2 dimensions (rows, cols) though any number of leading dimensions
            is also supported (for cycles, channels, z-planes, etc); must have length equal to region width * region height
        config: Experiment configuration
    Returns:
        An array of same data type as tiles with all n-2 dimensions the same as individual tiles, but with the last
        two dimensions expanded as a "montage" of all 2D images contained within the tiles
    """
    rw, rh = config.region_width, config.region_height
    
    # Determine shape/type of individual tiles by checking the first one 
    # (and assume all others will be equal)
    dtype_proto, shape_proto = tiles[0].dtype, tiles[0].shape
    if len(shape_proto) < 2:
        raise ValueError('Tiles must all be at least 2D (shape given = {})'.format(shape_proto))
    shape_rc, shape_ex = shape_proto[-2:], shape_proto[:-2]
    th, tw = shape_rc

    # Preserve leading dimensions and assume 2D image for each will be the 
    # same size multiplied by the number of tiles in row or column axis directions
    img_montage = np.zeros(np.concatenate((shape_ex, shape_rc * np.array([rh, rw]))).astype(np.int), dtype=dtype_proto)

    for itile, tile in enumerate(tiles):
        if tile.shape != shape_proto:
            raise ValueError('All tiles must have the same shape (found {}, expected {})'.format(tile.shape, shape_proto))
        tx, ty = config.get_tile_coordinates(itile)
        idx = [slice(None) for _ in shape_ex] + [slice(ty * th, (ty + 1) * th), slice(tx * tw, (tx + 1) * tw)]
        img_montage[tuple(idx)] = tile
    return img_montage
