import numpy as np


def area_to_diameter(area):
    return 2 * np.sqrt(area / np.pi)


def volume_to_diameter(volume):
    return 2 * ((3 * volume) / (4 * np.pi)) ** (1. / 3)


def pixel_area_to_squared_um(pixel_area, resolution_um):
    """Convert area in pixels to area in squared microns based on (lateral) microscope resolution"""
    return pixel_area * (resolution_um ** 2)


def pixel_area_to_diameter_um(pixel_area, resolution_um):
    """Convert area in pixels to diameter in microns based on (lateral) microscope resolution"""
    return area_to_diameter(pixel_area) * resolution_um


def pixel_volume_to_diameter_um(pixel_volume, resolution_um):
    """Convert volume in pixels to diameter in microns based on (lateral) microscope resolution"""
    return volume_to_diameter(pixel_volume) * resolution_um


def circularity(area, perimeter):
    """Compute circularity of an object based on area and perimeter

    Resulting score is 1 for a perfect circle and closer to 0 for non-circular objects. See here for
    more details: https://en.wikipedia.org/wiki/Shape_factor_(image_analysis_and_microscopy)#Circularity
    """
    # Default tiny objects to perfect circularity
    res = 1.0 if np.isclose(perimeter, 0) else (4 * np.pi * area) / perimeter ** 2
    return np.clip(res, 0, 1)
