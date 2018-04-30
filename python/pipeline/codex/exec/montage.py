"""Montage Image Generation Utilities"""
from codex import io as codex_io
from codex.ops import tile_crop
import numpy as np
import os.path as osp

def get_tile_montage(config, image_dir, hyperstack, icyc=0, iz=0, ich=0, ireg=0, bw=0, bv_fn=None):
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
    Returns:
        A (usually very large) 2D array containing all tiles stitched together
    """
    tile_indexes = list(range(config.n_tiles_per_region))
    tw, th = config.tile_width, config.tile_height
    rw, rh = config.region_width, config.region_height
    img = np.zeros((rh * th, rw * tw))
    for itile in tile_indexes:
        tx, ty = config.get_tile_coordinates(itile)

        # If operating on a hyperstack, extract the appropriate slice to add to the montage
        if hyperstack:
            path = codex_io.get_processor_img_path(ireg, tx, ty)
            tile = codex_io.read_tile(osp.join(image_dir, path), config)
            tile = tile[icyc, iz, ich, :, :]
        # Otherwise, assume raw acquisition files are to be loaded and then cropped before being added
        else:
            path = codex_io.get_raw_img_path(ireg, itile, icyc, ich, iz)
            tile = codex_io.read_image(osp.join(image_dir, path))
            tile = tile_crop.apply_slice(tile, tile_crop.get_slice(config))
        
        # Highlight borders, if configured to do so
        if bw > 0:
            bv = 0 if bv_fn is None else bv_fn(tx, ty)
            tile[0:bw, :] = bv
            tile[-bw:, :] = bv
            tile[:, 0:bw] = bv
            tile[:, -bw:] = bv
        
        # Add to montage
        img[(ty * th):((ty + 1) * th), (tx * tw):((tx + 1) * tw)] = tile
    return img

