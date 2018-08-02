import cv2
import numpy as np
import pandas as pd
import os.path as osp
from skimage import segmentation
from skimage import morphology
from skimage import measure
from skimage import filters
from skimage import exposure
from scipy import ndimage
from codex import math as codex_math
from codex import data as codex_data

DEFAULT_BATCH_SIZE = 1
CELL_CHANNEL = 0
NUCLEUS_CHANNEL = 1
DEFAULT_CHANNEL_PREFIX = 'ch:'


class Cytometer(object):

    def __init__(self, input_shape, weights_path=None):
        self.input_shape = input_shape
        self.weights_path = weights_path
        self.initialized = False
        self.model = None

    def initialize(self):
        self.model = self._get_model(self.input_shape)
        self.model.load_weights(self.weights_path or self._get_weights_path())
        self.initialized = True
        return self

    def _get_model(self, input_shape):
        raise NotImplementedError()

    def _get_weights_path(self):
        raise NotImplementedError()


def _to_uint8(img, name):
    if img.dtype != np.uint8 and img.dtype != np.uint16:
        raise ValueError(
            'Image must be 8 or 16 bit for segmentation (image name = {}, dtype = {}, shape = {})'
            .format(name, img.dtype, img.shape)
        )
    if img.dtype == np.uint16:
        img = exposure.rescale_intensity(img, in_range=np.uint16, out_range=np.uint8).astype(np.uint8)
    return img


class Cytometer2D(Cytometer):

    def _get_model(self, input_shape):
        # Load this as late as possible to avoid premature keras backend initialization
        from codex.cytometry.models import unet_v2 as unet_model
        return unet_model.get_model(3, input_shape)

    def _get_weights_path(self):
        # Load this as late as possible to avoid premature keras backend initialization
        from codex.cytometry.models import unet_v2 as unet_model
        path = osp.join(codex_data.get_cache_dir(), 'cytometry', 'unet_v2_weights.h5')
        return codex_data.download_file_from_google_drive(unet_model.WEIGHTS_FILE_ID, path, name='UNet Weights')

    def get_segmentation_mask(self, img_bin_nuci, img_memb=None, dilation_factor=0):
        if dilation_factor > 0:
            img_bin_nuci = cv2.dilate(
                img_bin_nuci.astype(np.uint8),
                morphology.disk(dilation_factor)
            ).astype(np.bool)
        if img_memb is None:
            return img_bin_nuci

        # Construct mask as threshold on membrane image OR binary nucleus mask
        img_bin_memb = img_memb > filters.threshold_otsu(img_memb)
        img_bin_memb = img_bin_memb | img_bin_nuci
        return morphology.remove_small_holes(img_bin_memb, 64)

    def segment(self, img_nuc, img_memb=None, nucleus_dilation=4, min_size=12,
                batch_size=DEFAULT_BATCH_SIZE, return_masks=False):
        if not self.initialized:
            self.initialize()

        # Convert images to segment or otherwise analyze to 8-bit
        img_nuc = _to_uint8(img_nuc, 'nucleus')
        if img_memb is not None:
            img_memb = _to_uint8(img_memb, 'membrane')

        # Add batch dimension if not present
        if img_nuc.ndim == 2:
            img_nuc = np.expand_dims(img_nuc, 0)
        if img_nuc.ndim != 3:
            raise ValueError('Must provide image as ZHW or HW (image shape given = {})'.format(img_nuc.shape))

        # Make predictions for each image after converting to 0-1; Result has shape
        # NHWC where C=3 and C1 = bg, C2 = interior, C3 = border
        img_pred = self.model.predict(np.expand_dims(img_nuc / 255., -1), batch_size=batch_size)
        assert img_pred.shape[0] == img_nuc.shape[0], \
            'Expecting {} predictions (shape = {})'.format(img_nuc.shape[0], img_pred.shape)
        assert img_pred.shape[-1] == 3, \
            'Expecting 3 outputs in predictions (shape = {})'.format(img_pred.shape)

        img_seg_list, img_bin_list = [], []
        nz = img_nuc.shape[0]
        for i in range(nz):

            # Use nuclei interior mask as watershed markers
            img_bin_nucm = np.argmax(img_pred[i], axis=-1) == 1

            # Remove markers (which determine number of cells) below the given size
            if min_size > 0:
                img_bin_nucm = morphology.remove_small_objects(img_bin_nucm, min_size=min_size)

            # Define the entire nucleus interior as a slight dilation of the markers noting that this
            # actually works better than using the union of predicted interiors and predicted boundaries
            # (which are often too thick)
            img_bin_nuci = cv2.dilate(img_bin_nucm.astype(np.uint8), morphology.disk(1)).astype(np.bool)

            # Label the markers and create the basin to segment over
            img_bin_nucm_label = morphology.label(img_bin_nucm)
            img_basin = -1 * ndimage.distance_transform_edt(img_bin_nucm)

            # Determine the overall mask to segment across by dilating nuclei by some fixed amount
            # or if possible, using the given cell membrane image
            img_bin_mask = self.get_segmentation_mask(
                img_bin_nuci, img_memb=img_memb[i] if img_memb is not None else None,
                dilation_factor=nucleus_dilation)

            # Run watershed using markers and expanded nuclei / cell mask
            img_cell_seg = segmentation.watershed(img_basin, img_bin_nucm_label, mask=img_bin_mask)

            # Generate nucleus segmentation based on cell segmentation and nucleus mask
            # and relabel nuclei objections using corresponding cell labels
            img_nuc_seg = (img_cell_seg > 0) & img_bin_nuci
            img_nuc_seg = img_nuc_seg * img_cell_seg

            # Add labeled images to results
            assert img_cell_seg.dtype == img_nuc_seg.dtype, \
                'Cell segmentation dtype {} != nucleus segmentation dtype {}'\
                .format(img_cell_seg.dtype, img_nuc_seg.dtype)
            img_seg_list.append(np.stack([img_cell_seg, img_nuc_seg], axis=0))

            # Add mask images to results, if requested
            if return_masks:
                img_bin_list.append(np.stack([img_bin_nuci, img_bin_nucm, img_bin_mask], axis=0))

        assert img_nuc.shape[0] == len(img_seg_list)
        if return_masks:
            assert img_nuc.shape[0] == len(img_bin_list)

        # Stack final segmentation image as (z, c, h, w)
        img_seg = np.stack(img_seg_list, axis=0)
        img_bin = np.stack(img_bin_list, axis=0) if return_masks else None
        assert img_seg.ndim == 4, 'Expecting 4D segmentation image but shape is {}'.format(img_seg.shape)

        # Return (in this order) labeled volumes, prediction volumes, mask volumes
        return img_seg, img_pred, img_bin

    def quantify(self, tile, img_seg, channel_names=None, channel_name_prefix=DEFAULT_CHANNEL_PREFIX):
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

                cell_area, nuc_area = cell_prop.area, nuc_prop.area
                row = [
                    cell_prop.label, cell_prop.centroid[1], cell_prop.centroid[0], iz,
                    cell_area, codex_math.area_to_diameter(cell_area), cell_prop.solidity,
                    nuc_area, codex_math.area_to_diameter(nuc_area), nuc_prop.solidity
                ]
                row += list(intensities)
                res.append(row)

        # Note: "size" is used here instead of "area" for compatibility between 2D and 3D
        columns = [
            'id', 'x', 'y', 'z',
            'cell_size', 'cell_diameter', 'cell_solidity',
            'nucleus_size', 'nucleus_diameter', 'nucleus_solidity'
        ]
        return pd.DataFrame(res, columns=columns + channel_names)


# def _get_flat_ball(size):
#     struct = morphology.ball(size)
#
#     # Ball structs should always be of odd size and double given radius pluse one
#     assert struct.shape[0] == size * 2 + 1
#     assert struct.shape[0] % 2 == 1
#
#     # Get middle index (i.e. position 2 (0-index 1) for struct of size 3)
#     mid = ((struct.shape[0] + 1) // 2) - 1
#
#     # Flatten the ball so there is no connectivity in the z-direction
#     struct[(mid + 1):] = 0
#     struct[:(mid)] = 0
#
#     return struct

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