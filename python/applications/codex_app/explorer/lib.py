from skimage.exposure import rescale_intensity
from PIL import Image
import numpy as np
from io import BytesIO
import base64
import dash_core_components as dcc
from cvutils import ops as cvops


def get_encoded_image(img):
    if img.dtype != np.uint8:
        img = rescale_intensity(img, out_range=np.uint8).astype(np.uint8)
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
    return dcc.Graph(id=id, figure={'data': [], 'layout': layout}, style=(style or {}))


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





# def InteractiveImage(id, img):
#     return interactive_image(id, get_interactive_image_layout(img))
