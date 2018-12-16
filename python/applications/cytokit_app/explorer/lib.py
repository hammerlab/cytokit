from skimage import exposure
from skimage import measure
from skimage import morphology
from scipy.stats import gaussian_kde
from cytokit.cli.operator import CH_SRC_CYTO
from PIL import Image
import pandas as pd
import numpy as np
from io import BytesIO
import base64
import dash_core_components as dcc
from cytokit.image import ops as cvops
import logging

logger = logging.getLogger(__name__)


def get_encoded_image(img):
    if img.dtype != np.uint8:
        img = exposure.rescale_intensity(img, out_range=np.uint8).astype(np.uint8)
    im = Image.fromarray(img)
    bio = BytesIO()
    im.save(bio, format='PNG')
    return base64.b64encode(bio.getvalue()).decode()


def get_interactive_image_layout(img=None, shape=None):
    if img is None and shape is None:
        raise ValueError('At least one of image or shape must be provided')
    if img is not None and shape is not None:
        if img.shape != shape:
            raise ValueError('Given image (shape = {}) does not match given shape {}'.format(img.shape, shape))
    if img is not None and shape is None:
        shape = img.shape

    nh, nw = shape[:2]
    rngx = [0, nw]
    rngy = [0, nh]

    image_def = {
        'xref': 'x',
        'yref': 'y',
        'x': rngx[0],
        'y': rngy[1],
        'sizex': rngx[1] - rngx[0],
        'sizey': rngy[1] - rngy[0],
        'sizing': 'stretch',
        'layer': 'below'
    }
    if img is not None:
        image_def['source'] = 'data:image/png;base64,{}'.format(get_encoded_image(img))

    return {
        'xaxis': {
            'autorange': False,
            'showgrid': False,
            'zeroline': False,
            'range': rngx,
        },
        'yaxis': {
            'autorange': False,
            'showgrid': False,
            'zeroline': False,
            'range': rngy,
            'scaleanchor': 'x',
            'scaleratio': 1
        },
        'hovermode': 'closest',
        'images': [image_def],
        'margin': dict(l=0, t=0, r=0, b=0, pad=0),
        'dragmode': 'select'  # or 'lasso'
    }


def get_interactive_image(id, layout, style=None):
    return dcc.Graph(
        id=id,
        figure={'data': [], 'layout': layout},
        style=(style or {}),
        config={
            'scrollZoom': True, 'showLink': False, 'displaylogo': False, 'linkText': '',
            # Remove spike lines and selector tools
            'modeBarButtonsToRemove': ['toggleSpikelines', 'lasso2d', 'select2d']
        }
    )


class ImageProcessor(object):

    def __init__(self, n_channels, ranges=None, colors=None):
        self.n_channels = n_channels
        self.colors = colors
        self.ranges = ranges

    def run(self, img):
        assert img.shape[0] == self.n_channels, \
            'Expecting {} channels but got image with shape {}'.format(self.n_channels, img.shape)

        # print('in', img.dtype, img.shape, img.min(), img.max())
        img = cvops.constrain_image_channels(img, ranges=self.ranges, dtype=np.uint8)
        # print('mid', img.dtype, img.shape, img.min(), img.max())
        img = cvops.blend_image_channels(img, colors=self.colors)
        # print('out', img.dtype, img.shape, img.min(), img.max())
        assert img.ndim == 3 and img.shape[-1] == 3, \
            'Expecting RGB result (image shape = {})'.format(img.shape)
        return img


def get_sorted_boundary_coords(prop):
    """Returns coordinates of skimage measure region property sorted by angle from centroid

    See: https://plot.ly/python/polygon-area (PolygonSort)
    Args:
        prop: Property for image object; must be for a BOUNDARY object already, not a filled object
    Returns:
        A 2D array like (row, col) with each entry sorted counterclockwise
    """
    cr, cc = prop.centroid
    angles = []
    coords = prop.coords
    for r, c in coords:
        angle = (np.arctan2(r - cr, c - cc) + 2.0 * np.pi) % (2.0 * np.pi)
        angles.append(angle)
    o = np.argsort(angles)
    return coords[np.array(o)]


def get_single_cell_data(df, raw_tile, display_tile, channels, cell_image_size=None,
                         object_type='cell_boundary', apply_mask=True):
    if df is None:
        return None

    cell_boundary_channel = CH_SRC_CYTO + '_' + object_type
    if cell_boundary_channel not in channels:
        logger.warning('Cannot generate single cell images because extract does not contain cell boundary channel')
        return None

    # Fetch raw tile image with original channels, and extract cell boundaries
    cell_tile = raw_tile[channels.index(cell_boundary_channel)].copy()

    # Eliminate cell objects not in sample
    cell_tile[~np.isin(cell_tile, df['id'].values)] = 0

    logger.debug(
        'Single cell tile shape = %s (%s), cell boundary image shape = %s (%s)',
        raw_tile.shape, raw_tile.dtype, cell_tile.shape, cell_tile.dtype
    )

    # Extract regions in RGB image (display tile) corresponding to cell labelings
    cell_data = extract_single_cell_data(
        cell_tile, display_tile, is_boundary=True,
        patch_shape=cell_image_size,
        apply_mask=apply_mask, fill_value=0)

    # Return list of dictionaries where each represents one cell (with at least an id and image)
    return cell_data


def extract_single_cell_data(cell_image, target_image, patch_shape=None, is_boundary=True,
                             apply_mask=True, fill_value=0):
    """Extract single cell images from a target image

    Args:
        cell_image: 2D label image containing cell objects (each with different id)
        target_image: Image from which to extract patches around cells; must be at least 2D
            in format HW[D1, D2, ...]
        patch_shape: Target shape of individual cell images; If None (default) no cropping/padding
            will occur but if set, this value should be a 2 item sequence [rows, cols] and cell image patches
            will be conformed to this shape by either cropping or padding out from the center
        is_boundary: Whether or not cell image is of boundary or masks (default True)
        apply_mask: Whether or not to set pixels outside of cell binary image to `fill_value` (default True)
        fill_value: Pixel values for parts of cell image outside cell object (default 0)
    """
    if target_image.shape[:2] != cell_image.shape[:2]:
        raise ValueError(
            'Cell label image (shape = {}) must have same HW dimensions as target image (shape = {})'
            .format(cell_image.shape, target_image.shape)
        )

    if patch_shape is not None and len(patch_shape) != 2:
        raise ValueError('Target patch shape should be a 2 item sequence (given = {})'.format(patch_shape))

    cells = []
    props = measure.regionprops(cell_image)
    for p in props:

        # Extract bounding box offsets for extraction
        min_row, min_col, max_row, max_col = p.bbox

        # Extract patch from target image (make sure to copy for subsequent mutations)
        patch = target_image[min_row:max_row, min_col:max_col].copy()

        # Remove off-target pixels, if necessary
        if apply_mask:
            # Set mask containing which pixels in patch to keep
            if is_boundary:
                mask = p.convex_image
            else:
                mask = p.filled_image

            # Set value outside of mask to provided fill value
            patch[~mask] = fill_value

        # Resize if necessary (without transforming original image content)
        if patch_shape is not None:
            patch = cvops.resize_image_with_crop_or_pad(
                patch, tuple(patch_shape) + patch.shape[2:],
                constant_values=fill_value)

        cells.append(dict(id=p.label, properties=p, image=patch))
    return cells


def get_kde_estimate(x, y, max_cells=None, random_state=None):
    """Return density estimate for X and Y vectors of a 2D scatterplot

    Args:
        x: 1D array-like
        y: 1D array-like; same length as x
        max_cells: Maximum number of cells to incorporate in estimation; defaults to no
            limit though performance will suffer for samples larger than ~10k rows
        random_state: Sampling random state; only applies when max_cells is not None and
            length of X/Y vectors is greater than max_cells
    """
    if x.size != y.size:
        raise ValueError('X and Y must be of same length for KDE estimation')

    # Apply sampling if necessary
    M = np.vstack([x, y])
    S = M
    if max_cells is not None and x.size > max_cells:
        idx = pd.Series(np.arange(x.size)).sample(n=max_cells, random_state=random_state)
        x = np.array(x)[idx.values]
        y = np.array(y)[idx.values]
        S = np.vstack([x, y])

    # Compute density estimate for each row
    return gaussian_kde(S)(M)


def get_density_scatter_plot_data(x, y, max_kde_cells, asinh_color_scale=False, **kwargs):
    """Get scatterplot with points colored by density"""
    col = get_kde_estimate(x, y, max_cells=max_kde_cells, random_state=1)
    if asinh_color_scale:
        col = np.arcsinh(col)
    fig_data = [
        dict(
            x=x,
            y=y,
            mode='markers',
            marker={**kwargs, **{'color': col}},
            type='scattergl',
            name='Cells'
        )
    ]
    return fig_data


def get_density_overlay_plot_data(x, y, **kwargs):
    """Get scatterplot (with uniform color) overlay on 2D contour histogram"""
    fig_data = [
        dict(
            x=x,
            y=y,
            # See: https://github.com/plotly/plotly.py/blob/master/plotly/colors.py
            colorscale='Portland',
            type='histogram2dcontour',
            opacity=1,
            contours={'coloring': 'fill'}
        ),
        dict(
            x=x,
            y=y,
            mode='markers',
            marker=kwargs,
            type='scattergl',
            name='Cells'
        )
    ]
    return fig_data


def _apply_fn(fn, v):
    if v is None:
        return v
    if np.isscalar(v):
        return fn(np.array([v]))[0]
    return fn(np.array(v))


class Transform(object):

    def apply(self, v):
        return _apply_fn(self._apply, v)

    def invert(self, v):
        return _apply_fn(self._invert, v)


class LinearTransform(Transform):

    def _apply(self, v):
        return v

    def _invert(self, v):
        return v


class Log10Transform(Transform):

    def _apply(self, v):
        return np.log10(v)

    def _invert(self, v):
        return np.power(10, v)


class AsinhTransform(Transform):

    def _apply(self, v):
        return np.arcsinh(v)

    def _invert(self, v):
        return np.sinh(v)


TRANSFORMS = {
    'linear': LinearTransform(),
    'log10': Log10Transform(),
    'asinh': AsinhTransform()
}


def get_transform_by_name(transform):
    if transform not in TRANSFORMS:
        raise ValueError('Transform "{}" is not valid (must be one of {})'.format(transform, list(TRANSFORMS.keys())))
    return TRANSFORMS[transform]