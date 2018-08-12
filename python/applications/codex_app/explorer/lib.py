from skimage import exposure
from skimage import measure
from skimage import morphology
from PIL import Image
import numpy as np
from io import BytesIO
import base64
import dash_core_components as dcc
from cvutils import ops as cvops


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


def extract_single_cell_images(
        cell_image, target_image,
        patch_shape=None, is_boundary=True, apply_mask=False, fill_value=0):
    """Extract single cell images from a target image

    Args:
        cell_mask_image: 2D label image containing cell masks (each with different id)
        target_image: Image from which to extract patches around cells; must be at least 2D
            in format HW[D1, D2, ...]
        patch_shape: Target shape of individual cell images; If None (default) no cropping/padding
            will occur but if set, this value should be a 2 item sequence [rows, cols] and cell image patches
            will be conformed to this shape by either cropping or padding out from the center
        is_boundary: Whether or not cell image is of boundary or masks
    """
    from skimage.measure import regionprops

    if target_image.shape[:2] != cell_image.shape[:2]:
        raise ValueError(
            'Cell label image (shape = {}) must have same HW dimensions as target image (shape = {})'
            .format(cell_image.shape, target_image.shape)
        )

    if patch_shape is not None and len(patch_shape) != 2:
        raise ValueError('Target patch shape should be a 2 item sequence (given = {})'.format(patch_shape))

    cells = []
    props = regionprops(cell_image)
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

