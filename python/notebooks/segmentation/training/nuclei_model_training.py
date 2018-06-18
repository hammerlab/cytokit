import os
import os.path as osp
import numpy as np
from skimage import io
from skimage import transform
from skimage import segmentation
from skimage import color
from skimage import exposure
from imgaug import augmenters as iaa
import tensorflow as tf
import keras
import tqdm
seed = 8392
class_names = ['bg', 'nucleus_interior', 'nucleus_boundary']
    
def add_boundary_class(mask):
    assert mask.dtype == np.bool, 'Expecting boolean but got {}'.format(mask.dtype)
    
    # Calulcate nuclei boundaries
    mask_nucb = segmentation.find_boundaries(mask, mode='inner')
    
    # Calculate nuclei interior and ensure that the boundary is contained within original mask
    mask_nuci = mask.astype(np.uint8) - mask_nucb.astype(np.uint8)
    assert np.all(mask_nuci >= 0)
    
    # Compute background mask
    mask_bg = np.ones(mask.shape, dtype=np.uint8) - mask.astype(np.uint8)
    
    # Stack everything together and convert back to boolean
    mask = np.stack([mask_bg, mask_nuci, mask_nucb], -1)
    assert np.all(mask) >= 0 and np.all(mask) <= 1
    return mask.astype(np.bool)

def load_images(data_dir, target_shape, ids=None, has_masks=True, add_boundaries=True):

    if ids is None:
        ids = os.listdir(data_dir)
    assert len(target_shape) == 3
    
    n = len(ids)
    n_ch = target_shape[-1]
    
    # If not adding boundaries, a single mask channel is produced 
    # otherwise 3 separate channels (each boolean) is created for 
    # bg, nuclei, and boundary
    n_class = 3 if add_boundaries else 1
    
    if n_ch not in [1, 3]:
        raise ValueError('Number of channels should be 1 or 3 not {}'.format(n_ch))
        
    rc_shape = target_shape[:-1]
    X = np.zeros((n,) + target_shape, dtype=np.uint8)
    Y = None
    if has_masks:
        Y = np.zeros((n,) + rc_shape + (n_class,), dtype=np.bool)
        
    for i, id in tqdm.tqdm(enumerate(ids), total=n):
        path = osp.join(data_dir, id)
        img_path = osp.join(path, 'images', id + '.png')
        
        # If one channel requested, ensure that image is read as 2D greyscale
        if n_ch == 1:
            img = io.imread(img_path, as_gray=True)
            assert img.ndim == 2
        # Otherwise, ensure result is 3 channel RGB
        else:
            img = io.imread(img_path)
            assert img.ndim == 3
            assert img.shape[-1] in [3, 4]
            if img.shape[-1] == 4:
                img = color.rgba2rgb(img)
        
        # Resize to desired shape
        img = transform.resize(img, rc_shape, mode='constant', preserve_range=True, anti_aliasing=False)
        if img.ndim == 2:
            img = np.expand_dims(img, -1)
            
        # Convert whatever the image is at this point to uint8 (note that this will rescale the range to min/max)
        img = exposure.rescale_intensity(img, in_range='image', out_range=np.uint8).astype(np.uint8)
        
        assert img.dtype == np.uint8, 'Expecting uint8 image but got {}'.format(img.dtype)
        X[i] = img

        if not has_masks:
            continue

        # Aggregate all masks as booleans
        mask = np.zeros(rc_shape, dtype=np.bool)
        for mask_file in next(os.walk(osp.join(path, 'masks')))[2]:
            m = io.imread(osp.join(path, 'masks', mask_file))
            m = transform.resize(m, rc_shape, mode='constant', preserve_range=True, anti_aliasing=False)
            assert m.ndim == 2
            mask = np.logical_or(mask, m.astype(np.bool))
            
        # If requested, convert to 3 class mask
        if add_boundaries:
            mask = add_boundary_class(mask)
        else:
            mask = np.expand_dims(mask, -1)

        assert mask.ndim == n_class
        assert mask.dtype == np.bool, 'Expecting bool mask but got {}'.format(mask.dtype)
        Y[i] = mask

    return X, Y


opt_conv = {"activation": "relu", "padding": "same", 'kernel_initializer': 'he_normal'}
#opt_conv = {'activation': 'elu', 'padding': 'same', 'kernel_initializer': 'he_normal'}
opt_bn = {'momentum': 0.9}

def get_model_core(input_shape, p=[6,7,8,9], ks=(3, 3), batch_norm=True, dropout=None):
    n = len(p)
    if dropout is not None and np.isscalar(dropout):
        dropout = [dropout] * n
        
    if dropout is not None and len(dropout) != n:
        raise ValueError('Must provide {} dropout probabilities (one for each layer) not {}'.format(n, len(dropout)))
        
    x = keras.layers.Input(shape=input_shape)

    blocks = []
    for i in range(n):
        # Initialize to either input or max pool of last layer (note that this
        # means the final layer added has no pooling)
        l = x if len(blocks) == 0 else keras.layers.MaxPooling2D()(blocks[-1])
        
        # First conv layer
        l = keras.layers.Conv2D(2**p[i], ks, **opt_conv)(l)
        if batch_norm:
            l = keras.layers.BatchNormalization(**opt_bn)(l)
        if dropout is not None:
            l = keras.layers.Dropout(dropout[i])(l)
            
        # Second conv layer
        l = keras.layers.Conv2D(2**p[i], ks, **opt_conv)(l)
        if batch_norm:
            l = keras.layers.BatchNormalization(**opt_bn)(l)

        blocks.append(l)
        
    # Loop from second to last power back to first
    for i in range(n-1)[::-1]:
        
        # Prefer Conv2DTranspose instead of upsampling (e.g. l = keras.layers.UpSampling2D()(blocks[-1]))
        l = keras.layers.Conv2DTranspose(2**p[i], (2, 2), strides=(2, 2), padding='same') (blocks[-1])
        l = keras.layers.merge.concatenate([l, blocks[i]])
        
        # First conv layer
        l = keras.layers.Conv2D(2**p[i], ks, **opt_conv)(l)
        if batch_norm:
            l = keras.layers.BatchNormalization(**opt_bn)(l)
        if dropout is not None:
            l = keras.layers.Dropout(dropout[i])(l)
            
        # Second conv layer
        l = keras.layers.Conv2D(2**p[i], ks, **opt_conv)(l)
        if batch_norm:
            l = keras.layers.BatchNormalization(**opt_bn)(l)
            
        blocks.append(l)
        
    return [x, blocks[-1]]
            

# def get_model_core(input_shape, p, ks=(3, 3)):
#     if len(p) != 4:
#         raise ValueError('Expecting 4 power of 2 layer depths, given power list = {}'.format(p))
#     
#     x = keras.layers.Input(shape=input_shape)
#         
#     a = keras.layers.Conv2D(2**p[0], ks, **opt_conv)(x)
#     a = keras.layers.BatchNormalization(**opt_bn)(a)
# 
#     a = keras.layers.Conv2D(2**p[0], ks, **opt_conv)(a)
#     a = keras.layers.BatchNormalization(**opt_bn)(a)
# 
#     y = keras.layers.MaxPooling2D()(a)
# 
#     b = keras.layers.Conv2D(2**p[1], ks, **opt_conv)(y)
#     b = keras.layers.BatchNormalization(**opt_bn)(b)
# 
#     b = keras.layers.Conv2D(2**p[1], ks, **opt_conv)(b)
#     b = keras.layers.BatchNormalization(**opt_bn)(b)
# 
# 
#     y = keras.layers.MaxPooling2D()(b)
# 
#     c = keras.layers.Conv2D(2**p[2], ks, **opt_conv)(y)
#     c = keras.layers.BatchNormalization(**opt_bn)(c)
# 
#     c = keras.layers.Conv2D(2**p[2], ks, **opt_conv)(c)
#     c = keras.layers.BatchNormalization(**opt_bn)(c)
# 
# 
#     y = keras.layers.MaxPooling2D()(c)
# 
#     d = keras.layers.Conv2D(2**p[3], ks, **opt_conv)(y)
#     d = keras.layers.BatchNormalization(**opt_bn)(d)
# 
#     d = keras.layers.Conv2D(2**p[3], ks, **opt_conv)(d)
#     d = keras.layers.BatchNormalization(**opt_bn)(d)
# 
# 
#     # UP
# 
#     d = keras.layers.UpSampling2D()(d)
# 
#     y = keras.layers.merge.concatenate([d, c], axis=3)
#     #y = keras.layers.merge([d, c], concat_axis=3, mode="concat")
# 
#     e = keras.layers.Conv2D(2**p[2], ks, **opt_conv)(y)
#     e = keras.layers.BatchNormalization(**opt_bn)(e)
# 
#     e = keras.layers.Conv2D(2**p[2], ks, **opt_conv)(e)
#     e = keras.layers.BatchNormalization(**opt_bn)(e)
# 
#     e = keras.layers.UpSampling2D()(e)
# 
# 
#     y = keras.layers.merge.concatenate([e, b], axis=3)
#     #y = keras.layers.merge([e, b], concat_axis=3, mode="concat")
# 
#     f = keras.layers.Conv2D(2**p[1], ks, **opt_conv)(y)
#     f = keras.layers.BatchNormalization(**opt_bn)(f)
# 
#     f = keras.layers.Conv2D(2**p[1], ks, **opt_conv)(f)
#     f = keras.layers.BatchNormalization(**opt_bn)(f)
# 
#     f = keras.layers.UpSampling2D()(f)
# 
#     y = keras.layers.merge.concatenate([f, a], axis=3)
#     #y = keras.layers.merge([f, a], concat_axis=3, mode="concat")
# 
#     y = keras.layers.Conv2D(2**p[0], ks, **opt_conv)(y)
#     y = keras.layers.BatchNormalization(**opt_bn)(y)
# 
#     y = keras.layers.Conv2D(2**p[0], ks, **opt_conv)(y)
#     y = keras.layers.BatchNormalization(**opt_bn)(y)
# 
#     return [x, y]


def get_model(nclass, input_shape, activation='softmax', **kwargs):

    [x, y] = get_model_core(input_shape, **kwargs)

    y = keras.layers.Conv2D(nclass, (1, 1), **opt_conv)(y)

    if activation is not None:
        y = keras.layers.Activation(activation)(y)

    model = keras.models.Model(x, y)

    return model

def class_weighted_softmax_cross_entropy_loss(weights):
    w = weights
    # Class values are: bg, nucleus interior, nucleus border
    def loss(y_true, y_pred):
        class_weights = tf.constant([[[w]]])
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred)
        weights = tf.reduce_sum(class_weights * y_true, axis=-1)
        weighted_losses = weights * unweighted_losses
        return tf.reduce_mean(weighted_losses)
    return loss


def weighted_pixelwise_crossentropy(weights):
    """Get pixel-weighted sigmoid cross entropy loss"""
    
    # Assume that weights could cover any dimension after the 
    # batch dimension (i.e. the first one) -- from there, it
    # should simply be broadcastable to the dimensions of a single
    # prediction/label (which for images may be 3D)
    w = [weights]
    
    def loss(y_true, y_pred):
        _epsilon = 10e-8
        y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)
        return - tf.reduce_mean(y_true * w *  tf.log(y_pred) + (1 - y_true)  *  tf.log(1 - y_pred))
    return loss


def apply_augmentation(image, mask, augmentation):
    # See: https://github.com/aleju/imgaug/issues/41
    # https://github.com/matterport/Mask_RCNN/blob/4129a27275c48c672f6fd8c6303a88ba1eed643b/mrcnn/model.py
    import imgaug
    
    # Augmentors that are safe to apply to masks
    # Some, such as Affine, have settings that make them unsafe
    MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                       "Fliplr", "Flipud", "CropAndPad",
                       "Affine", "PiecewiseAffine"]
    def hook(images, augmenter, parents, default):
        """Determines which augmenters to apply to masks."""
        return (augmenter.__class__.__name__ in MASK_AUGMENTERS)
 

    if image.dtype != np.uint8:
        raise ValueError('Image must be of type uint8 for augmentation (given {})'.format(image.dtype))
    if mask.dtype != np.bool:
        raise ValueError('Mask must be of type boolean for augmentation (given {})'.format(mask.dtype))

    # Store shapes before augmentation to compare
    image_shape = image.shape
    mask_shape = mask.shape
    
    # Make augmenters deterministic to apply similarly to images and masks
    det = augmentation.to_deterministic()
    image = det.augment_image(image)

    # Change mask to np.uint8 because imgaug doesn't support np.bool
    mask = det.augment_image(mask.astype(np.uint8),
                             hooks=imgaug.HooksImages(activator=hook))

    # Verify that shapes didn't change
    assert image.shape == image_shape, "Augmentation shouldn't change image size"
    assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
    # Change mask back to bool
    mask = mask.astype(np.bool)

    return image, mask
    

    

class BaseDataGenerator(keras.utils.Sequence):

    def __init__(self, ids, loader, batch_size=32, shuffle=True):
        'Initialization'
        self.ids = ids
        self.loader = loader
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        
    @staticmethod
    def from_arrays(X, Y, **kwargs):
        if len(X) != len(Y):
            raise ValueError('X and Y should have same length; shape X = {}, shape Y = {}'.format(X.shape, Y.shape))
        ids = np.arange(len(X))
        def loader(sample_id):
            return X[sample_id], Y[sample_id]
        return BaseDataGenerator(ids, loader, **kwargs)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        batch_ids = [self.ids[k] for k in indexes]

        # Generate data
        return self.__data_generation(batch_ids)

    def on_epoch_end(self):
        """Update indexes after each epoch"""
        self.indexes = np.arange(len(self.ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_ids):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = []
        Y = []

        # Generate data
        for i, sample_id in enumerate(batch_ids):
            x, y = self.loader(sample_id)
            X.append(x)
            Y.append(y)

        return np.stack(X, 0), np.stack(Y, 0)
