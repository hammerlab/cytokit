from skimage.exposure import rescale_intensity
from PIL import Image
import numpy as np
from io import BytesIO
import base64
import dash_core_components as dcc


def get_encoded_image(img):
    im = Image.fromarray(rescale_intensity(img, out_range=(0, 255)).astype(np.uint8))
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

    nh, nw = shape
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
            'range': rngx,
        },
        'yaxis': {
            'autorange': False,
            'showgrid': False,
            'range': rngy,
            'scaleanchor': 'x',
            'scaleratio': 1
        },
        # 'height': 1000,
        # 'width': 1000,
        'images': [image_def],
        'margin': dict(l=0, t=0, r=0, b=0, pad=0),
        'dragmode': 'select'  # or 'lasso'
    }


def get_interactive_image(id, layout, style=None):
    return dcc.Graph(id=id, figure={'data': [], 'layout': layout}, style=(style or {}))




# def InteractiveImage(id, img):
#     return interactive_image(id, get_interactive_image_layout(img))
