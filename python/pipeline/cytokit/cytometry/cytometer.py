import cv2
import numpy as np
import pandas as pd
import os.path as osp
from skimage import segmentation
from skimage import morphology
from skimage import measure
from skimage import filters
from skimage import exposure
from skimage import transform
from skimage import img_as_float
from skimage.future import graph as label_graph
from centrosome import propagate
from scipy import ndimage
from cytokit import math as cytokit_math
from cytokit import data as cytokit_data

DEFAULT_BATCH_SIZE = 1
CELL_CHANNEL = 0
NUCLEUS_CHANNEL = 1
DEFAULT_CELL_INTENSITY_PREFIX = 'ci:'
DEFAULT_NUCL_INTENSITY_PREFIX = 'ni:'
DEFAULT_CELL_GRAPH_PREFIX = 'cg:'
DEFAULT_CELL_SPOT_PREFIX = 'cs:'

DEFAULT_PREFIXES = [
    DEFAULT_CELL_INTENSITY_PREFIX,
    DEFAULT_NUCL_INTENSITY_PREFIX,
    DEFAULT_CELL_SPOT_PREFIX,
    DEFAULT_CELL_GRAPH_PREFIX
]

DEFAULT_SPOT_CIRCULARITY = [.5, 1]
DEFAULT_SPOT_AREA = [2, 64]

COMP_CELL = 'cell'
COMP_NUCLEUS = 'nucleus'

INTENSITY_COMPONENTS = {
    COMP_CELL: DEFAULT_CELL_INTENSITY_PREFIX,
    COMP_NUCLEUS: DEFAULT_NUCL_INTENSITY_PREFIX
}


class KerasCytometer2D(object):

    def __init__(self, input_shape, target_shape=None, weights_path=None):
        """Cytometer Initialization

        Args:
            input_shape: Shape of input images as HWC tuple
            target_shape: Shape of resized images to use for prediction as HW tuple; if None (default),
                then the images will not be resized
            weights_path: Path to model weights; if None (default), a default path will be used
        """
        self.input_shape = input_shape
        self.target_shape = target_shape
        self.weights_path = weights_path
        self.initialized = False
        self.model = None

        if len(input_shape) != 3:
            raise ValueError('Input shape must be HWC 3 tuple (given {})'.format(input_shape))
        if target_shape is not None and len(target_shape) != 2:
            raise ValueError('Target shape must be HW 2 tuple (given {})'.format(target_shape))

        # Set resize indicator to true only if target HW dimensions differ from original
        self.resize = target_shape is not None and target_shape != input_shape[:2]

    def initialize(self):
        # Choose input shape for model based on whether or not resizing is being used
        if self.resize:
            # Set as HWC where HW comes from target shape
            input_shape = tuple(self.target_shape) + (self.input_shape[-1],)
        else:
            input_shape = self.input_shape
        self.model = self._get_model(input_shape)
        self.model.load_weights(self.weights_path or self._get_weights_path())
        self.initialized = True
        return self

    def _resize(self, img, shape):
        """Resize NHWC image to target shape

        Args:
            img: Image array with shape NHWC
            shape: Shape to resize to (HW tuple)
        Return:
            Image array with shape NHWC where H and W are equal to sizes in `shape`
        """
        if img.ndim != 4:
            raise ValueError('Expecting 4D NHWC image to resize (given shape = {})'.format(img.shape))
        if len(shape) != 2:
            raise ValueError('Expecting 2 tuple for target shape (given {})'.format(shape))

        # Resize expects images as HW first and then trailing dimensions will be ignored if
        # explicitly set to resize them to the same values
        input_shape = img.shape
        output_shape = tuple(shape) + (input_shape[-1], input_shape[0])
        img = np.moveaxis(img, 0, -1)  # NHWC -> HWCN
        img = transform.resize(img, output_shape=output_shape, mode='constant', anti_aliasing=True, preserve_range=True)
        img = np.moveaxis(img, -1, 0)  # HWCN -> NHWC

        # Ensure result agrees with original in N and C dimensions
        assert img.shape[0] == input_shape[0] and img.shape[-1] == input_shape[-1], \
            'Resized image does not have expected batch and channel dim values (input shape = {}, result shape = {}' \
            .format(input_shape, img.shape)
        return img

    def predict(self, img, batch_size):
        """Run prediction for an image

        Args:
            img: Image array with shape NHWC
            batch_size: Number of images to predict at one time
        Return:
            Predictions from model with shape NHWC where C can differ from input while all other dimensions are
                the same (difference depends on prediction targets of model)
        """
        if img.ndim != 4:
            raise ValueError('Expecting 4D NHWC image for prediction but got image with shape "{}"'.format(img.shape))
        if img.shape[1:] != self.input_shape:
            raise ValueError(
                'Given image with shape {} does not match expected image shape {} in non-batch dimensions'
                .format(img.shape[1:], self.input_shape)
            )
        if batch_size < 1:
            raise ValueError('Batch size must be integer >= 1 (given {})'.format(batch_size))

        shape = img.shape

        # Resize input, if necessary
        if self.resize:
            img = self._resize(img, self.target_shape)

        # Run predictions on NHWC0 image to give NHWC1 result where C0 possibly != C1
        img = self.model.predict(img, batch_size=batch_size)

        # Make sure results are NHWC
        if img.ndim != 4:
            raise AssertionError('Expecting 4D prediction image results but got image with shape {}'.format(img.shape))

        # Convert HW dimensions of predictions back to original, if necessary
        if self.resize:
            img = self._resize(img, self.input_shape[:2])

        # Ensure results agree with input in NHW dimensions
        if img.shape[:-1] != shape[:-1]:
            raise AssertionError(
                'Prediction and input images do not have same NHW dimensions (input shape = {}, result shape = {})'
                .format(shape, img.shape)
            )

        return img

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


class ObjectProperties(object):

    def __init__(self, cell, nucleus):
        self.props = {COMP_CELL: cell, COMP_NUCLEUS: nucleus}
        if cell.label != nucleus.label:
            raise ValueError(
                'Expecting equal labels for cell and nucleus (nucleus label = {}, cell label = {})'
                .format(nucleus.label, cell.label)
            )

    def __getitem__(self, key):
        return self.props[key]

    @property
    def cell(self):
        return self.props[COMP_CELL]

    @property
    def nucleus(self):
        return self.props[COMP_NUCLEUS]


class FeatureCalculator(object):

    def get_feature_names(self):
        raise NotImplementedError()

    def get_feature_values(self, signals, labels, graph, props, z):
        raise NotImplementedError()


class BasicCellFeatures(FeatureCalculator):

    def get_feature_names(self):
        # Note: "size" is used here instead of area/volume for compatibility between 2D and 3D
        return [
            'id', 'x', 'y', 'z',
            'cell_size', 'cell_diameter', 'cell_perimeter', 'cell_circularity', 'cell_solidity',
            'nucleus_size', 'nucleus_diameter', 'nucleus_perimeter', 'nucleus_circularity', 'nucleus_solidity'
        ]

    def get_feature_values(self, signals, labels, graph, props, z):
        # Extract these once as their calculations are not cached
        cell_area, nuc_area = props.cell.area, props.nucleus.area
        cell_perimeter, nuc_perimeter = props.cell.perimeter, props.nucleus.perimeter
        return [
            props.cell.label, props.cell.centroid[1], props.cell.centroid[0], z,

            cell_area, cytokit_math.area_to_diameter(cell_area),
            cell_perimeter, cytokit_math.circularity(cell_area, cell_perimeter), props.cell.solidity,

            nuc_area, cytokit_math.area_to_diameter(nuc_area),
            nuc_perimeter, cytokit_math.circularity(nuc_area, nuc_perimeter), props.nucleus.solidity
        ]


def _quantify_intensities(image, prop):
    # Get a (n_pixels, n_channels) array of intensity values associated with
    # this region and then average across n_pixels dimension
    intensities = image[prop.coords[:, 0], prop.coords[:, 1]].mean(axis=0)
    assert intensities.ndim == 1, 'Expecting 1D resulting intensities but got shape {}'.format(intensities.shape)
    return list(intensities)


class IntensityFeatures(FeatureCalculator):

    def __init__(self, n_channels, channel_names, component):
        self.n_channels = n_channels
        self.channel_names = channel_names
        self.component = component

        if len(channel_names) != n_channels:
            raise ValueError(
                'Channel name list expected to have {} names (names given = {})',
                n_channels, channel_names
            )
        if component not in INTENSITY_COMPONENTS:
            raise ValueError(
                'Cellular component to quantify intensities for must be one of {} not "{}"'
                .format(list(INTENSITY_COMPONENTS.keys()), component)
            )

    def get_feature_names(self):
        prefix = INTENSITY_COMPONENTS[self.component]
        return [prefix + c for c in self.channel_names]

    def get_feature_values(self, signals, labels, graph, props, z):
        # Signals should have shape ZHWC
        assert signals.ndim == 4, 'Expecting 4D signals image but got shape {}'.format(signals.shape)

        # Extract target skimage region prop
        prop = props[self.component]

        values = _quantify_intensities(signals[z], prop)
        if len(values) != self.n_channels:
            raise AssertionError(
                'Expecting {} {} intensity measurements but got result {}'
                .format(self.n_channels, self.component, values)
            )
        return values


class SpotFeatures(FeatureCalculator):

    METRICS = ['area_in_cell', 'area_in_nucleus', 'circularity', 'coverage']

    def __init__(self, channel_indexes, channel_names, threshold=None, sigma=None):
        self.channel_indexes = channel_indexes
        self.channel_names = channel_names
        self.threshold = threshold
        self.sigma = sigma

        if len(channel_indexes) != len(channel_names):
            raise ValueError(
                'Channel name and index lists must have same length (names given = {}, indexes given = {})',
                channel_names, channel_indexes
            )

    def get_feature_names(self):
        features = []
        for c in self.channel_names:
            for metric in self.METRICS:
                features.append(DEFAULT_CELL_SPOT_PREFIX + c + ':' + metric)
        return features

    def get_feature_values(self, signals, labels, graph, props, z):
        # Signals should have shape ZHWC
        assert signals.ndim == 4, 'Expecting 4D signals image but got shape {}'.format(signals.shape)

        # Determine spot count (with cell component) for each channel requested
        res = []
        default_value = [None] * len(self.METRICS)
        for ci in self.channel_indexes:

            # Extract image for channel over cell component
            image = signals[z, ..., ci]

            # Extract bounding box in original image and multiply by cell.image to give
            # original image with region containing object alone
            assert image.ndim == 2, 'Expecting 2D image at this point, not image with shape {}'.format(image.shape)
            min_row, min_col, max_row, max_col = props.cell.bbox
            image = image[min_row:max_row, min_col:max_col] * props.cell.image

            # If image contains only one intensity value (or none), add zero count and continue
            if len(np.unique(image)) <= 1:
                res.extend(default_value)
                continue

            # Apply thresholding and labeling before getting spot properties, using supplied threshold
            # if possible and otsu otherwise
            if self.sigma is not None:
                image = filters.gaussian(image, sigma=self.sigma, preserve_range=True)
            threshold = filters.threshold_otsu(image) if self.threshold is None else self.threshold
            thresh_image = image > threshold
            ps = measure.regionprops(measure.label(thresh_image))

            # If there are no spot objects, short-circuit to next loop
            if not ps:
                res.extend(default_value)
                continue

            # Create a binary nucleus image within the cell image bounding box
            nuc_image = np.zeros_like(props.cell.image, dtype=bool)
            nuc_image[props.nucleus.coords[:, 0] - min_row, props.nucleus.coords[:, 1] - min_col] = True

            # Loop through spot objects and add metrics for each
            metrics = []
            for p in ps:
                area, perimeter = p.area, p.perimeter
                nuc_area = nuc_image[p.coords[:, 0], p.coords[:, 1]].sum()
                circularity = cytokit_math.circularity(area, perimeter)
                coverage = np.clip(area / props.cell.area, 0, 1)
                metrics.append((area, nuc_area, circularity, coverage))

            for i in range(len(self.METRICS)):
                res.append(','.join([str(m[i]) for m in metrics]))

            # Debugging single cell images
            # from skimage import io as skio
            # label = props.cell.label
            # print('Cell {}: {}'.format(label, metrics))
            # skio.imsave('/lab/data/cellimages/{:05d}_extract.png'.format(label), image.astype(signals.dtype))
            # skio.imsave('/lab/data/cellimages/{:05d}_cell_binary.png'.format(label), props.cell.image.astype(np.uint8) * 255)
            # skio.imsave('/lab/data/cellimages/{:05d}_nucleus_binary.png'.format(label), nuc_image.astype(np.uint8) * 255)
            # skio.imsave('/lab/data/cellimages/{:05d}_threshold_ct{}.png'.format(label, len(ps)), thresh_image.astype(np.uint8) * 255)

        return res


def _pct_list(fractions):
    return ','.join(['{:.2f}'.format(100 * v) for v in fractions])


class GraphFeatures(FeatureCalculator):

    def get_feature_names(self):
        return [
            DEFAULT_CELL_GRAPH_PREFIX + c
            for c in ['n_neighbors', 'neighbor_ids', 'adj_neighbor_pct', 'adj_bg_pct']
        ]

    def get_feature_values(self, signals, labels, graph, props, z):
        # graph.adj behaves like a dict keyed by node id where each node id is an integer label in the
        # labeled image and the value associated is another dictionary keyed by neighbor node ids (with
        # values equal to the data associated with the edge).  Examples:
        # rag.adj[1] --> AtlasView({2: {'weight': 1.0, 'count': 24}})
        # rag.adj[1][2] --> {'weight': 1.0, 'count': 24}
        # Also note that if a background class is present all nodes will be neighbors of it, but if there is no
        # background (when watershed returns no 0 labeled images if no mask given) then there will be no "0"
        # node id (so be careful with assuming its there)

        # Get the edges/neighbors data from the graph for this cell
        nbrs = graph.adj[props.cell.label]

        # Get list of non-bg neighbor ids
        nids = [nid for nid in nbrs.keys() if nid != 0]

        # Get raw weight (which is number of bordering pixels on both sides of boundary)
        # associated with each non-bg neighbor
        nbwts = np.array([nbrs[nid]['count'] for nid in nids])

        # Get raw weight of background, if present
        bgwt = nbrs[0]['count'] if 0 in nbrs else 0

        wtsum = bgwt + nbwts.sum()
        assert wtsum > 0, \
            'Cell {} has no neighbors and associated boundary pixel counts (this should not be possible)'\
            .format(props.cell.label)
        return [
            len(nids),
            ','.join([str(nid) for nid in nids]),
            _pct_list(nbwts / wtsum),
            _pct_list([bgwt / wtsum])
        ]


class Cytometer2D(KerasCytometer2D):

    def _get_model(self, input_shape):
        # Load this as late as possible to avoid premature keras backend initialization
        from cytokit.cytometry.models import unet_v2 as unet_model
        return unet_model.get_model(3, input_shape)

    def _get_weights_path(self):
        return cytokit_data.initialize_cytometry_2d_model()

    def get_segmentation_mask(self, img_bin_nuc, img_memb=None,
                              min_dist=None, max_dist=None, hole_size=None,
                              method='li', sigma=None, gamma=None):
        # Determine mask image for minimum distance from nuclei
        img_bin_min = img_bin_nuc
        if min_dist:
            img_bin_min = cv2.dilate(
                img_bin_nuc.astype(np.uint8),
                morphology.disk(min_dist)
            ).astype(np.bool)

        # Return immediately if no cell membrane/boundary image is available
        if img_memb is None:
            return img_bin_min

        # Construct mask as threshold on membrane image OR binary nucleus mask
        if sigma:
            # Image from gaussian filter is float with original image range
            img_memb = filters.gaussian(img_memb, sigma=sigma, preserve_range=True)
        if gamma is not None:
            # Gamma adjustment preserves data type and range
            img_memb = exposure.adjust_gamma(img_memb, gamma=gamma)

        threshold_fn = getattr(filters, 'threshold_' + method)
        img_bin_memb = img_memb > threshold_fn(img_memb)
        img_bin_memb = img_bin_memb | img_bin_min

        # Fill small holes in mask based on given area threshold
        if hole_size:
            img_bin_memb = morphology.remove_small_holes(img_bin_memb, hole_size)

        # Eliminate mask pixels more than max_dist pixels from nuclei
        if max_dist:
            img_dist = ndimage.distance_transform_edt(~img_bin_nuc)
            img_bin_memb[img_dist > max_dist] = False

        assert img_bin_memb.dtype == np.bool, \
            'Segmentation mask should be boolean not {}'.format(img_bin_memb.dtype)
        return img_bin_memb

    def segment(self, img_nuc, img_memb=None,
                marker_dilation=1, marker_min_size=16,
                memb_min_dist=5, memb_max_dist=10, memb_hole_size=16,
                memb_sigma=1, memb_gamma=None, memb_tresh_method='li', memb_propagation_regularization=.05,
                batch_size=DEFAULT_BATCH_SIZE, return_masks=False):
        if not self.initialized:
            self.initialize()

        if (memb_min_dist or 0) >= (memb_max_dist or np.inf):
            raise ValueError(
                'Membrane min distance argument (memb_min_dist = {}) used to set minimum cell boundary '
                'must be <= maximum cell boundary distance from nucleus (memb_max_dist = {})'
                .format(memb_min_dist, memb_max_dist)
            )

        # Convert images to segment or otherwise analyze to 8-bit
        img_nuc = _to_uint8(img_nuc, 'nucleus')
        if img_memb is not None:
            img_memb = _to_uint8(img_memb, 'membrane')

        # Add z dimension (equivalent to batch dim in this case) if not present
        if img_nuc.ndim == 2:
            img_nuc = np.expand_dims(img_nuc, 0)
        if img_nuc.ndim != 3:
            raise ValueError('Must provide image as ZHW or HW (image shape given = {})'.format(img_nuc.shape))

        # Make predictions on image converted to 0-1 and with trailing channel dimension to give NHWC;
        # Result has shape NHWC where C=3 and C1 = bg, C2 = interior, C3 = border
        img_pred = self.predict(np.expand_dims(img_nuc / 255., -1), batch_size)
        assert img_pred.shape[-1] == 3, \
            'Expecting 3 outputs in predictions (shape = {})'.format(img_pred.shape)

        img_seg_list, img_bin_list = [], []
        nz = img_nuc.shape[0]
        for i in range(nz):

            # Use nuclei interior mask as watershed markers
            img_bin_nucm = np.argmax(img_pred[i], axis=-1) == 1

            # Remove markers (which determine number of cells) below the given size
            if marker_min_size > 0:
                img_bin_nucm = morphology.remove_small_objects(img_bin_nucm, min_size=marker_min_size)

            # Define the entire nucleus as a slight dilation of the markers noting that this
            # works better than using the union of predicted interiors and predicted boundaries
            # (which are too thick)
            img_bin_nuci = img_bin_nucm
            if marker_dilation > 0:
                img_bin_nuci = cv2.dilate(
                    img_bin_nucm.astype(np.uint8), morphology.disk(marker_dilation)).astype(np.bool)

            # Determine the overall mask to segment across by dilating nuclei by some fixed amount
            # or if possible, using the given cell membrane image
            img_bin_mask = self.get_segmentation_mask(
                img_bin_nuci, img_memb=img_memb[i] if img_memb is not None else None,
                min_dist=memb_min_dist, max_dist=memb_max_dist, hole_size=memb_hole_size,
                method=memb_tresh_method, sigma=memb_sigma, gamma=memb_gamma)

            # Label the nuclei markers (which determines number of cells to output)
            # *Note: It is important to keep this separate from nuclei interior as single or double pixel
            # gaps between nuclei are common when densely packed
            img_bin_nucm_label = morphology.label(img_bin_nucm)

            # Create labeled cell image
            if img_memb is None or memb_propagation_regularization is None:
                # Run watershed using markers and expanded nuclei / cell mask
                img_basin = -1 * ndimage.distance_transform_edt(img_bin_nucm)
                img_cell_seg = segmentation.watershed(img_basin, img_bin_nucm_label, mask=img_bin_mask)
            else:
                # Before running propagation segmentation, make sure that the input image is 0-1 float
                # as the regularization threshold is calibrated to work only with data in that range
                img_cell_seg, _ = propagate.propagate(
                    img_as_float(img_memb[i]), img_bin_nucm_label,
                    img_bin_mask, memb_propagation_regularization
                )

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

        assert nz == len(img_seg_list)
        if return_masks:
            assert nz == len(img_bin_list)

        # Stack final segmentation image as (z, c, h, w)
        img_seg = np.stack(img_seg_list, axis=0)
        img_bin = np.stack(img_bin_list, axis=0) if return_masks else None
        assert img_seg.ndim == 4, 'Expecting 4D segmentation image but shape is {}'.format(img_seg.shape)

        # Return (in this order) labeled volumes, prediction volumes, mask volumes
        return img_seg, img_pred, img_bin

    def quantify(self, tile, img_seg, channel_names=None,
                 include_cell_intensity=True,
                 include_nucleus_intensity=False,
                 include_cell_graph=False,
                 spot_count_channels=None,
                 spot_count_params=None):
        ncyc, nz, _, nh, nw = tile.shape

        # Move cycles and channels to last axes (in that order)
        tile = np.moveaxis(tile, 0, -1)
        tile = np.moveaxis(tile, 1, -1)

        # Collapse tile to ZHWC (instead of cycles and channels being separate)
        tile = np.reshape(tile, (nz, nh, nw, -1))
        nch = tile.shape[-1]

        # Generate default channel names list if necessary
        if channel_names is None:
            channel_names = ['{:03d}'.format(i) for i in range(nch)]

        if nch != len(channel_names):
            raise ValueError(
                'Tile has {} channels but given channel name list has {} (they should be equal); '
                'channel names given = {}, tile shape = {}'
                .format(nch, len(channel_names), channel_names, tile.shape)
            )

        # Configure features to be calculated based on provided flags
        feature_calculators = [BasicCellFeatures()]
        if include_cell_intensity:
            feature_calculators.append(IntensityFeatures(nch, channel_names, COMP_CELL))
        if include_nucleus_intensity:
            feature_calculators.append(IntensityFeatures(nch, channel_names, COMP_NUCLEUS))
        if include_cell_graph:
            feature_calculators.append(GraphFeatures())
        if spot_count_channels is not None:
            indexes = [channel_names.index(c) for c in spot_count_channels]
            params = spot_count_params or {}
            feature_calculators.append(SpotFeatures(indexes, spot_count_channels, **params))

        # Compute list of resulting feature names (values will be added in this order)
        feature_names = [v for fc in feature_calculators for v in fc.get_feature_names()]

        feature_values = []
        for z in range(nz):
            # Calculate properties of masked+labeled cell components
            cell_props = measure.regionprops(img_seg[z][CELL_CHANNEL], cache=False)
            nucleus_props = measure.regionprops(img_seg[z][NUCLEUS_CHANNEL], cache=False)
            if len(cell_props) != len(nucleus_props):
                raise ValueError(
                    'Expecting cell and nucleus properties to have same length (nucleus props = {}, cell props = {})'
                    .format(len(nucleus_props), len(cell_props))
                )

            # Compute RAG for cells if necessary
            graph = None
            if include_cell_graph:
                labels = img_seg[z][CELL_CHANNEL]

                # rag_boundary fails on all zero label matrices so default to empty graph if that is the case
                # see: https://github.com/scikit-image/scikit-image/blob/master/skimage/future/graph/rag.py#L386
                if np.count_nonzero(labels) > 0:
                    graph = label_graph.rag_boundary(labels, np.ones(labels.shape))
                else:
                    graph = label_graph.RAG()

            # Loop through each detected cell and compute features
            for i in range(len(cell_props)):
                props = ObjectProperties(cell=cell_props[i], nucleus=nucleus_props[i])

                # Run each feature calculator and add results in order
                feature_values.append([
                    v for fc in feature_calculators
                    for v in fc.get_feature_values(tile, img_seg, graph, props, z)
                ])

        return pd.DataFrame(feature_values, columns=feature_names)
