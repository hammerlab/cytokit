import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

def init_session(gpu_fraction=0.75):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    KTF.set_session(tf.Session(config=config))

    
DEFAULT_IDS = {
    'ex1': '0ea221716cf13710214dcd331a61cea48308c3940df1d28cfc7fd817c83714e1',
    'ex2': '547ef286ee5f4e5dce533e982e6992ada67b7d727fdd3cfa6576f24c631a7ae6',
    'ex3': 'cb4df20a83b2f38b394c67f1d9d4aef29f9794d5345da3576318374ec3a11490',
    'ex4': 'a102535b0e88374bea4a1cfd9ee7cb3822ff54f4ab2a9845d428ec22f9ee2288'
}

def load_images(data_dir, ids=DEFAULT_IDS):
    from skimage import io
    from skimage import color
    from skimage import exposure
    import os.path as osp
    import numpy as np
    
    if not isinstance(ids, dict):
        ids = {i: id for i, id in enumerate(ids)}
        
    def load(id):
        img = io.imread(osp.join(data_dir, id, 'images', id + '.png'))
        if img.ndim == 3:
            img = color.rgb2gray(img)
        img = exposure.rescale_intensity(img, out_range=np.uint8).astype(np.uint8)
        return img

    return {k:load(v) for k, v in ids.items()}


def visualize_augmentation(img, augmentation, n, **kwargs):
    from cvutils import visualize
    visualize.display_images([img] + [augmentation.augment_image(img) for _ in range(n)], **kwargs)
    
    
def visualize_segmentation_augmentation(image, mask, augmentation, n, mask_channels=None, **kwargs):
    """Visualize augmentation on an image and its associated masks
    
    Args:
        image: Any 2 or 3D uint8 image array in format HW or HWC
        mask: Any 3D mask array in format HWC (can be boolean or integer)
    """
    from cvutils import visualize
    from skimage import color
    from cvutils.augmentation import imgaug as imgaug_utils
    import numpy as np
    
    if mask_channels is None:
        mask_channels = np.arange(mask.shape[-1])
    mask_viz = np.stack([mask[..., i] for i in mask_channels], -1)
    
    images = []
    for i in range(n):
        image_aug, mask_aug = imgaug_utils.apply_augmentation(image, mask_viz, augmentation)
        images.append(image.squeeze())
        images.append(image_aug.squeeze())
        images.extend([mask_aug[..., i] for i in mask_channels])
    visualize.display_images(images, cols=2 + len(mask_channels), **kwargs)
    
    
def visualize_model_history(history, keys=None, ncol=2, width=8, height=2, window=None):
    import plotnine as pn
    import pandas as pd
    import numpy as np
    
    if keys is None:
        keys = sorted(list(history.keys()))
    n = len(keys)
    nrow = int(np.ceil(n / ncol))
    figsize = (ncol * width, nrow * height)
        
    if window is None:
        window = slice(None)
    df = pd.concat([
        pd.DataFrame(dict(epoch=np.array(range(len(h)))[window], value=h[window])).assign(key=k)
        for k, h in history.items()
    ])
    return (
        pn.ggplot(df, pn.aes(x='epoch', y='value')) +
        pn.geom_line() + 
        pn.facet_wrap('~key', ncol=ncol, scales='free') +
        pn.theme(figure_size=figsize)
    )