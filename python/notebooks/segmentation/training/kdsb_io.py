import os
import os.path as osp
import pandas as pd
import numpy as np
from skimage import io
from skimage import transform
from skimage import segmentation
from skimage import color
from skimage import exposure
from skimage import measure
from skimage import draw
from skimage import morphology
from skimage.util import view_as_blocks
import tqdm


def get_image_metadata(classes_csv_path, data_dir=None):
    df = pd.read_csv(classes_csv_path)
    
    df['fileid'] = df['filename'].str.replace('.png', '')
    
    if data_dir is not None:
        df['localdir'] = df['fileid'].apply(lambda f: osp.join(data_dir, f))
        df['localpath'] = df.apply(lambda r: osp.join(r['localdir'], 'images', r['filename']), axis=1)
        df['exists'] = df['localdir'].apply(lambda f: osp.exists(f))
        df['nmasks'] = df.apply(lambda r: None if not r['exists'] else len(os.listdir(osp.join(r['localdir'], 'masks'))), axis=1)

        # Read in images and fetch certain helpful properties
        img_info = []
        for i, r in df.iterrows():
            if not r['exists']:
                img_info.append(dict(shape=None, dtype=None))
            else:
                img = io.imread(r['localpath'])
                img_info.append(dict(shape=img.shape, dtype=img.dtype))
                
        df['ndim'] = [len(i['shape']) if i['shape'] else None for i in img_info]
        df['height'] = [i['shape'][0] if i['shape'] else None for i in img_info]
        df['width'] = [i['shape'][1] if i['shape'] else None for i in img_info]
        df['channels'] = [i['shape'][2] if i['shape'] else None for i in img_info]
        df['dtype'] = [i['dtype'] for i in img_info]
        assert df['ndim'].notnull().sum() == df['exists'].sum()
    
    return df

def load_images(data_dir, target_shape, ids=None, resize_mode='crop'):

    if ids is None:
        ids = os.listdir(data_dir)
    assert len(target_shape) == 3
    
    n = len(ids)
    n_ch = target_shape[-1]
    if n_ch not in [1, 3]:
        raise ValueError('Number of channels should be 1 or 3 not {}'.format(n_ch))

    X = []
    I = []
    Y = [] 
    classes = None
    
    for i, id in tqdm.tqdm(enumerate(ids), total=n):
        path = osp.join(data_dir, id)
        img_path = osp.join(path, 'images', id + '.png')
        
        ###############
        # Image Loading
        ###############
        # Load in the raw image for this example and convert to 
        # either RGB or gray
        
        # If one channel requested, ensure that image is read as 2D greyscale
        if n_ch == 1:
            img = io.imread(img_path, as_gray=True)
            assert img.ndim == 2
            img = np.expand_dims(img, -1)
            
        # Otherwise, ensure result is 3 channel RGB
        else:
            img = io.imread(img_path)
            assert img.ndim == 3
            assert img.shape[-1] in [3, 4]
            if img.shape[-1] == 4:
                img = color.rgba2rgb(img)
                
        # All images should be 3D at this point
        assert img.ndim == 3
            
        #################
        # Mask Processing
        #################
        # Create training targets using masks on original scales
        
        mask_files = next(os.walk(osp.join(path, 'masks')))[2]
        n_mask = len(mask_files)
        masks = np.zeros((n_mask,) + img.shape[:2], dtype=np.bool)
        for mi, mask_file in enumerate(mask_files):
            m = io.imread(osp.join(path, 'masks', mask_file))
            assert m.shape == img.shape[:2], 'Expecting shape {} but got {}'.format(img.shape[:2], m.shape)
            masks[mi] = m.astype(np.bool)
        
        # Convert masks from multiple instances to single instances with multiple classes
        masks, classes = get_objectives(masks)
        n_class = len(classes)
            
        ##########
        # Resizing
        ##########
        # Conform shapes of images to target by either rescaling them or cropping them
        # over windows (i.e. one image turns into many when cropping out windows)
        
        if resize_mode == 'crop':
            imgs, masks = crop_data(img, masks, n_class, target_shape)
        elif resize_mode == 'rescale':
            imgs, masks = rescale_data(img, masks, n_class, target_shape)
        else:
            raise ValueError('Resize mode must be one of "crop" or "rescale" not "{}"'.format(resize_mode))
        assert len(imgs) == len(masks)
      
        X.extend(imgs)
        Y.extend(masks)
        I.extend([id]*len(imgs))

    # Make sure that number of images matches number of correlated ids
    assert len(I) == len(X)
    
    # Validate images before strack and perform image scaling
    for i in range(len(X)):
        x = X[i]
        assert x.shape == target_shape, \
            'Sample shape {} not equal to expected {} (id={})'.format(x.shape, target_shape, I[i])
        # Convert whatever the image is at this point to uint8 (note that this will rescale the range to min/max)
        X[i] = exposure.rescale_intensity(x, in_range='image', out_range=np.uint8).astype(np.uint8)
    X = np.stack(X, 0)
    assert X.dtype == np.uint8, 'Expecting uint8 images but got {}'.format(X.dtype)

    # Validate masks
    for i in range(len(Y)):
        y = Y[i]
        expected_shape = target_shape[:2] + (len(classes),)
        assert y.shape == expected_shape, \
            'Objectives shape {} not equal to expected {} (id={})'.format(y.shape, expected_shape, I[i])
        assert y.dtype == np.bool, \
            'Objectives type {} not boolean (id={})'.format(y.dtype, I[i])
    Y = np.stack(Y, 0)
    assert Y.dtype == np.bool, 'Expecting bool objectives but got {}'.format(Y.dtype)
    
    return np.array(I), X, Y, classes


def crop_data(img, masks, n_class, target_shape):
    assert img.ndim == 3
    assert len(target_shape) == 3
    
    imgs = list(img_to_blocks(img, target_shape))
    
    # Loop through masks and create a block view for each class
    mb = []
    for ci in range(n_class):
        m = masks[..., ci]
        mb.append(list(img_to_blocks(m, target_shape[:2])))
    
    # Now because the blocks were specific to a class and not a sample,
    # regroup them by sample horizontally (imagine mb as a 2D n_class x n_blocks
    # grid that need to be restacked to be a list of length n_blocks)
    y = []
    for i in range(len(imgs)):
        yib = np.stack([mb[ci][i] for ci in range(n_class)], -1)
        assert yib.shape[-1] == n_class
        assert yib.shape[:2] == target_shape[:2]
        y.append(yib)
            
    assert len(y) == len(imgs)
    assert len(imgs) < 250, 'More than 250 cropped images created ({} created) which is likely a mistake'.format(len(imgs))
    return imgs, y

def img_to_blocks(img, target_shape):
    
    # Compute the grid shape where each grid location indicates a spot for a target_shape sized image
    grid_shape = np.ceil(np.array(img.shape) / np.array(target_shape)).astype(int)
    
    # Compute total size of non-partitioned result
    box_shape = grid_shape * np.array(target_shape)
    
    diff_shape = box_shape - np.array(img.shape)
    
    # Pad image out to the left and below to match box shape
    img_pad = np.pad(img, [(0, p) for p in diff_shape], mode='constant')
    assert img_pad.shape == tuple(box_shape)
    
    # Partition image into equally sized blocks 
    blocks = view_as_blocks(img_pad, target_shape)
    
    # Restack blocks along axis 0
    return np.reshape(blocks, (-1,) + target_shape) 


def rescale_data(img, masks, n_class, target_shape):
    assert img.ndim == 3
    assert len(target_shape) == 3
    
    # Resize to desired shape
    img = transform.resize(img, target_shape[:2], mode='constant', preserve_range=True, anti_aliasing=False)
    if img.ndim == 2:
        img = np.expand_dims(img, -1)
    
    # Loop through generated masks and resize them
    y = np.zeros(target_shape[:2] + (n_class,), dtype=np.bool)
    for ci in range(n_class):
        m = masks[..., ci]
        m = transform.resize(m, target_shape[:2], mode='constant', preserve_range=True, anti_aliasing=False)
        assert m.ndim == 2
        y[..., ci] = m.astype(np.bool)

    return [img], [y]

def get_objectives(masks, adjacent_margin=2):
    """Generative predictive objectives from all masks for a single instance
    
    Args:
        masks: A (n_mask, height, width) array containing all masks for a single instance
        adjacent_margin: Distance (in pixels) between boundaries to be counted as adjacent
    """
    assert adjacent_margin >= 1
    assert masks.ndim == 3, 'Expecting 3D masks array; shape = {}'.format(masks.shape)
    assert masks.dtype == np.bool, 'Expecting boolean masks but got {}'.format(masks.dtype)
    n_mask = masks.shape[0]
    img_shape = masks.shape[1:]
    n_class = 3
    assert n_mask > 0, 'Must have at least one mask to process'
    
    m_shape = img_shape + (n_class,)
    res = np.zeros(img_shape + (n_class,), dtype=np.bool)
    
    m_bgrd = np.zeros(img_shape, dtype=np.bool)  # Background
    m_nuci = np.zeros(img_shape, dtype=np.bool)  # Nuclei interior
    m_nucb = np.zeros(img_shape, dtype=np.bool)  # Nuclei boundary
    m_nucn = np.zeros(img_shape, dtype=np.uint16)  # Nuclei shared boundary (neighbors)
    m_nucc = np.zeros(img_shape, dtype=np.bool)  # Nuclei center markers
    
    for i in range(n_mask):
        mask = masks[i]
        
        props = measure.regionprops(mask.astype(np.uint8))
        assert len(props) == 1, 'Expecting single object in mask but found {}'.format(len(props))
        
        m_bgrd = np.logical_or(m_bgrd, mask)
        m_nuci = np.logical_or(m_nuci, morphology.erosion(mask))
        m_nucb = np.logical_or(m_nucb, segmentation.find_boundaries(mask, mode='inner'))
        m_nucc = np.logical_or(m_nucc, get_center_marker(mask, props[0]))
        
        # Dilate the mask to make it possible to detect overlap
        # *Note that at least one dilation is necessary to catch adjacent boundaries
        mask_dilated = mask.copy()
        for j in range(adjacent_margin):
            mask_dilated = morphology.dilation(mask_dilated)
        m_nucn += mask_dilated.astype(np.uint16)
        
    # Choose adjacent boundaries as those pixels with more than 1 nuclei boundary
    m_nucn = (m_nucn > 1)
    
    # Negate to give background
    m_bgrd = np.logical_not(m_bgrd)
    
    classes = ['bg', 'nuc_interior', 'nuc_boundary', 'nuc_partition', 'nuc_center']
    return np.stack([m_bgrd, m_nuci, m_nucb, m_nucn, m_nucc], -1), classes
    

# def add_center_marker(mask, prop, radius=4):
#    r, c = prop.centroid
#    rr, cc = draw.circle(int(r), int(c), radius=radius, shape=mask.shape)
#    mask[rr, cc] = True
#    return mask

def get_center_marker(mask, prop, radius=2, shrink=8):
    # Shrink the mask and if there's anything left of it, return result
    mask_shrink = morphology.erosion(mask, morphology.disk(shrink))
    if mask_shrink.sum() > 0:
        return mask_shrink
    
    # Otherwise, use the mask centroid
    r, c = prop.centroid
    m = np.zeros_like(mask)
    rr, cc = draw.circle(int(r), int(c), radius=radius, shape=m.shape)
    m[rr, cc] = True
    return m
