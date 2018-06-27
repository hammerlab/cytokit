import numpy as np
import pandas as pd
from skimage import segmentation
from skimage import morphology
from skimage import measure
from skimage import filters
from scipy import ndimage

DEFAULT_BATCH_SIZE = 8
CELL_CHANNEL = 0
NUCLEUS_CHANNEL = 1


class Cytometer(object):

    def __init__(self, input_shape, model_path):
        self.input_shape = input_shape
        self.model_path = model_path
        self.initialized = False
        self.model = None

    def initialize(self):
        self.model = self._get_model(self.input_shape)
        self.model.load_weights(self.model_path)
        self.initialized = True
        return self


def _get_unet_v1_model(input_shape):
    from codex.cytometry.models import unet_v1
    import keras  # Load this as late as possible
    # conv_activation = lambda l: keras.layers.LeakyReLU(alpha=.3)(l)
    conv_activation = lambda l: keras.layers.Activation('elu')(l)
    return unet_v1.get_model(3, input_shape, 'sigmoid', conv_activation=conv_activation)


def _get_flat_ball(size):
    struct = morphology.ball(size)

    # Ball structs should always be of odd size and double given radius pluse one
    assert struct.shape[0] == size * 2 + 1
    assert struct.shape[0] % 2 == 1

    # Get middle index (i.e. position 2 (0-index 1) for struct of size 3)
    mid = ((struct.shape[0] + 1) // 2) - 1

    # Flatten the ball so there is no connectivity in the z-direction
    struct[(mid + 1):] = 0
    struct[:(mid)] = 0

    return struct


class Cytometer2D(Cytometer):

    def _get_model(self, input_shape):
        return _get_unet_v1_model(input_shape)

    def prepocess(self, img, thresh, min_size):
        img = img > thresh
        # img = morphology.remove_small_holes(img, area_threshold=6)
        img = morphology.remove_small_objects(img, min_size=min_size)
        return img

    def get_segmentation_mask(self, img_bin_nuci, img_memb=None, dilation_factor=0):
        if dilation_factor > 0:
            img_bin_nuci = morphology.dilation(img_bin_nuci, selem=morphology.disk(dilation_factor))
        if img_memb is None:
            return img_bin_nuci

        # Construct mask as thresholded membrane image OR binary nucleus mask
        img_bin_memb = img_memb > filters.threshold_otsu(img_memb)
        img_bin_memb = img_bin_memb | img_bin_nuci
        return morphology.remove_small_holes(img_bin_memb, 64)

    def segment(self, img_nuc, img_memb=None, nucleus_dilation=4, proba_threshold=.5, min_size=6,
                batch_size=DEFAULT_BATCH_SIZE, return_masks=False):
        if not self.initialized:
            self.initialize()

        if img_nuc.dtype != np.uint8:
            raise ValueError('Must provide uint8 image not {}'.format(img_nuc.dtype))

        # Add batch dimension if not present
        if img_nuc.ndim == 2:
            img = np.expand_dims(img_nuc, 0)
        if img_nuc.ndim != 3:
            raise ValueError('Must provide image as NHW or HW (image shape w/ batch dimension = {})'.format(img_nuc.shape))

        # Make predictions for each image in batch
        img_pred = self.model.predict(np.expand_dims(img_nuc, -1) / 255., batch_size=batch_size)
        assert img_pred.shape[0] == img_nuc.shape[0], \
            'Expecting {} predictions (shape = {})'.format(img_nuc.shape[0], img_pred.shape)
        assert img_pred.shape[-1] == 3, \
            'Expecting 3 outputs in predictions (shape = {})'.format(img_pred.shape)

        img_seg_list, img_bin_list = [], []
        nz = img_nuc.shape[0]
        for i in range(nz):
            # Extract prediction channels
            img_bin_nuci, img_bin_nucb, img_bin_nucm = [
                self.prepocess(img_pred[i, ..., j], proba_threshold, min_size) for j in range(3)]

            # Form watershed markers as marker class intersection with nuclei class, minus boundaries
            img_bin_nucm = img_bin_nucm & img_bin_nuci & ~img_bin_nucb

            # Label the markers and create the basin to segment (+boundary, -nucleus interior)
            img_bin_nucm_label = morphology.label(img_bin_nucm)
            img_bin_nuci_basin = ndimage.distance_transform_edt(img_bin_nuci)
            img_bin_nucb_basin = ndimage.distance_transform_edt(img_bin_nucb)
            img_basin = img_bin_nucb_basin - img_bin_nuci_basin

            # Determine the overall mask to segment across by dilating nuclei as an approximation for cytoplasm/membrane
            img_bin_mask = self.get_segmentation_mask(
                img_bin_nuci, img_memb=img_memb[i] if img_memb is not None else None,
                dilation_factor=nucleus_dilation)

            # Run cell segmentation and append to results
            img_cell_seg = segmentation.watershed(img_basin, img_bin_nucm_label, mask=img_bin_mask)

            # Generate nucleus segmentation based on cell segmentation and nucleus mask
            # (with the same integer labels as in the cell segmentation)
            img_nuc_seg = ((img_cell_seg > 0) & img_bin_nuci) * img_cell_seg

            # Add labeled images to results
            assert img_cell_seg.dtype == img_nuc_seg.dtype, \
                'Cell segmentation dtype {} != nucleus segmentation dtype {}'\
                .format(img_cell_seg.dtype, img_nuc_seg.dtype)
            img_seg_list.append(np.stack([img_cell_seg, img_nuc_seg], axis=0))

            # Add mask images to results
            if return_masks:
                img_bin_list.append(np.stack([img_bin_nuci, img_bin_nucb, img_bin_nucm, img_bin_mask], axis=0))

        assert img_nuc.shape[0] == len(img_seg_list)
        if return_masks:
            assert img_nuc.shape[0] == len(img_bin_list)

        # Stack final segmentation image as (z, c, h, w)
        img_seg = np.stack(img_seg_list, axis=0)
        img_bin = np.stack(img_bin_list, axis=0) if return_masks else None
        assert img_seg.ndim == 4, 'Expecting 4D segmentation image but shape is {}'.format(img_seg.shape)

        # Return (in this order) labeled volumes, prediction volumes, mask volumes
        return img_seg, img_pred, img_bin

    def quantify(self, tile, img_seg, channel_names=None, channel_name_prefix='ch:'):
        ncyc, nz, _, nh, nw = tile.shape

        # Move cycles and channels to last axes (in that order)
        tile = np.moveaxis(tile, 0, -1)
        tile = np.moveaxis(tile, 1, -1)

        # Collapse tile to ZHWC (instead of cycles and channels being separate)
        tile = np.reshape(tile, (nz, nh, nw, -1))
        nch = tile.shape[-1]

        if channel_names is None:
            channel_names = ['{}{:03d}'.format(channel_name_prefix, i) for i in range(nch)]
        else:
            channel_names = [channel_name_prefix + c for c in channel_names]
        if nch != len(channel_names):
            raise ValueError(
                'Data tile contains {} channels but channel names list contains only {} items '
                '(names given = {}, tile shape = {})'
                    .format(nch, len(channel_names), channel_names, tile.shape))

        res = []
        for iz in range(nz):
            cell_props = measure.regionprops(img_seg[iz][CELL_CHANNEL], cache=False)
            nuc_props = measure.regionprops(img_seg[iz][NUCLEUS_CHANNEL], cache=False)
            assert len(cell_props) == len(nuc_props), \
                'Expecting cell and nucleus properties to have same length (nuc props = {}, cell props = {})'\
                .format(len(nuc_props), len(cell_props))

            for i in range(len(cell_props)):
                cell_prop, nuc_prop = cell_props[i], nuc_props[i]
                assert cell_prop.label == nuc_prop.label, \
                    'Expecting equal labels for cell and nucleus (nuc label = {}, cell label = {})'\
                    .format(nuc_prop.label, cell_prop.label)

                # Get a (n_pixels, n_channels) array of intensity values associated with
                # this region and then average across n_pixels dimension
                intensities = tile[iz][cell_prop.coords[:, 0], cell_prop.coords[:, 1]].mean(axis=0)
                assert intensities.ndim == 1
                assert len(intensities) == nch
                row = [
                    cell_prop.label, cell_prop.centroid[1], cell_prop.centroid[0], iz,
                    cell_prop.area, cell_prop.solidity, nuc_prop.area, nuc_prop.solidity
                ]
                row += list(intensities)
                res.append(row)

        columns = ['id', 'x', 'y', 'z', 'cell_size', 'cell_solidity', 'nucleus_size', 'nucleus_solidity']
        return pd.DataFrame(res, columns=columns + channel_names)


# class Cytometer3D(Cytometer):
#
#     def _get_model(self, input_shape):
#         return _get_unet_v1_model(input_shape)
#
#     def prepocess(self, img, thresh, min_size):
#         img = img > thresh
#         if min_size > 0:
#             # img = morphology.remove_small_holes(img, area_threshold=min_size)
#             img = np.stack([morphology.remove_small_objects(img[i], min_size=min_size) for i in range(img.shape[0])])
#         return img
#
#     def get_segmentation_mask(self, img_bin_nuci, dilation_factor=0):
#         if dilation_factor > 0:
#             return morphology.dilation(img_bin_nuci, selem=_get_flat_ball(dilation_factor))
#         else:
#             return img_bin_nuci
#
#     def segment(self, img, nucleus_dilation=4, proba_threshold=.5, min_size=6, batch_size=DEFAULT_BATCH_SIZE):
#         if not self.initialized:
#             self.initialize()
#
#         if img.dtype != np.uint8:
#             raise ValueError('Must provide uint8 image not {}'.format(img.dtype))
#         if img.squeeze().ndim != 3:
#             raise ValueError(
#                 'Must provide single, 3D grayscale (or an image with other unit dimensions) '
#                 'image but not image with shape {}'.format(img.shape))
#         img = img.squeeze()
#         nz = img.shape[0]
#
#         img_pred = self.model.predict(np.expand_dims(img, -1) / 255., batch_size=batch_size)
#         assert img_pred.shape[0] == nz, \
#             'Expecting {} predictions but got result with shape {}'.format(nz, img_pred.shape)
#
#         # Extract prediction channels
#         img_bin_nuci, img_bin_nucb, img_bin_nucm = [self.prepocess(img_pred[..., i], proba_threshold, min_size) for i in range(3)]
#
#         # Form watershed markers as marker class intersection with nuclei class, minus boundaries
#         img_bin_nucm = img_bin_nucm & img_bin_nuci & ~img_bin_nucb
#
#         # Label the markers and create the basin to segment (+boundary, -nucleus interior)
#         img_bin_nucm_label = morphology.label(img_bin_nucm)
#         img_bin_nuci_basin = ndimage.distance_transform_edt(img_bin_nuci)
#         img_bin_nucb_basin = ndimage.distance_transform_edt(img_bin_nucb)
#         img_basin = -img_bin_nuci_basin + img_bin_nucb_basin
#
#         # Determine the overall mask to segment across by dilating nuclei as an approximation for cytoplasm/membrane
#         seg_mask = self.get_segmentation_mask(img_bin_nuci, dilation_factor=nucleus_dilation)
#
#         # Run segmentation and return results
#         img_seg = segmentation.watershed(img_basin, img_bin_nucm_label, mask=seg_mask)
#
#         return img_seg, img_pred, np.stack([img_bin_nuci, img_bin_nucb, img_bin_nucm], axis=-1)
#
#     def quantify(self, tile, cell_segmentation, channel_names=None, channel_name_prefix='ch:'):
#         ncyc, nz, _, nh, nw = tile.shape
#
#         # Move cycles and channels to last axes (in that order)
#         tile = np.moveaxis(tile, 0, -1)
#         tile = np.moveaxis(tile, 1, -1)
#
#         # Collapse tile to ZHWC (instead of cycles and channels being separate)
#         tile = np.reshape(tile, (nz, nh, nw, -1))
#         nch = tile.shape[-1]
#
#         if channel_names is None:
#             channel_names = ['{}{:03d}'.format(channel_name_prefix, i) for i in range(nch)]
#         else:
#             channel_names = [channel_name_prefix + c for c in channel_names]
#         if nch != len(channel_names):
#             raise ValueError(
#                 'Data tile contains {} channels but channel names list contains only {} items '
#                 '(names given = {}, tile shape = {})'
#                     .format(nch, len(channel_names), channel_names, tile.shape))
#
#         res = []
#         props = measure.regionprops(cell_segmentation)
#         for i, prop in enumerate(props):
#             # Get a (n_pixels, n_channels) array of intensity values associated with
#             # this region and then average across n_pixels dimension
#             intensities = tile[prop.coords[:, 0], prop.coords[:, 1], prop.coords[:, 0]].mean(axis=0)
#             assert intensities.ndim == 1
#             assert len(intensities) == nch
#             row = [prop.label, prop.centroid[2], prop.centroid[1], prop.centroid[0], prop.area, prop.solidity]
#             row += list(intensities)
#             res.append(row)
#
#         return pd.DataFrame(res, columns=['id', 'x', 'y', 'z', 'volume', 'solidity'] + channel_names)