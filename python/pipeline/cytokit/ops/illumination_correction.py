import os.path as osp
import numpy as np
import pandas as pd
from collections import OrderedDict
from cytokit.ops import op as cytokit_op
from cytokit import io as cytokit_io
from cytokit.cytometry.cytometer import DEFAULT_CELL_INTENSITY_PREFIX
from cytokit.function import data as function_data
from sklearn.linear_model import HuberRegressor, Ridge, Lasso, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from skimage import transform
from cytokit import SEED
import logging
logger = logging.getLogger(__name__)

DEFAULT_FILTER_PARAMS = dict(percentile_range=[.01, .99], features=['cell_size'], max_cells=250000)
DEFAULT_FEATURE_PARAMS = dict(
    tile=dict(type='polynomial', degree=2),
    region=dict(type='polynomial', degree=2),
    index=dict(type='polynomial', degree=2)
)
DEFAULT_MODEL_PARAMS = dict(type='ls')
DEFAULT_PREDICTION_PARAMS = dict(step_size=10)
MODELS = {
    'ls': {'factory': lambda args: LinearRegression(**{**dict(normalize=True), **args})},
    'huber': {'factory': lambda args: HuberRegressor(**args)},
    'ridge': {'factory': lambda args: Ridge(**{**dict(normalize=True), **args})},
    'gbr': {'factory': lambda args: GradientBoostingRegressor(random_state=SEED, **args)},
    'knn': {'factory': lambda args: KNeighborsRegressor(**args)},
    'mlp': {'factory': lambda args: Pipeline([('scale', StandardScaler()), ('est', MLPRegressor(**args))])}
}


def _get_params(key, params, defaults):
    # Copy default values
    res = dict(defaults)
    # If params present, overwrite default values
    if key in params:
        res.update(params[key])
    return res


def _get_coordinate_features(region_shape, tile_shape, step=1):
    ri, ci = np.arange(0, region_shape[0], step), np.arange(0, region_shape[1], step)
    nr, nc = len(ri), len(ci)

    # Produce 2D array with 2 cols, ry and rx
    X_region = np.transpose([np.repeat(ri, nc), np.tile(ci, nr)])

    # Create tile index features by dividing region y/x coordinates by tile shape
    X_index = np.floor_divide(X_region, tile_shape)

    # Mod region coordinates by tile height and width to give coordinates within tile
    X_tile = np.mod(X_region, tile_shape)

    # Return all 4 features as a data frame as well as number of (potentially sampled
    # coordinate points along each dimension)
    # NOTE: All field names here must match fields in cytometry data frames
    X = pd.DataFrame(np.hstack((X_region, X_index, X_tile)), columns=['ry', 'rx', 'tile_y', 'tile_x', 'y', 'x'])

    return X, (nr, nc)


class IlluminationCorrection(cytokit_op.CytokitOp):

    def __init__(
            self, config,
            filter_params=DEFAULT_FILTER_PARAMS,
            feature_params=DEFAULT_FEATURE_PARAMS,
            model_params=DEFAULT_MODEL_PARAMS,
            prediction_params=DEFAULT_PREDICTION_PARAMS,
            intensity_prefix=DEFAULT_CELL_INTENSITY_PREFIX):
        super().__init__(config)

        params = config.illumination_correction_params

        # Get mapping of source to target channel names, and sort by source channel
        self.channel_mapping = pd.Series(params['channel_mapping']).sort_index()

        # Extra parameters relevant to each step
        self.filter_params = _get_params('filter_params', params, filter_params)
        self.feature_params = _get_params('feature_params', params, feature_params)
        self.model_params = _get_params('model_params', params, model_params)
        self.prediction_params = _get_params('prediction_params', params, prediction_params)

        # Extract prefix of intensity signals within cytometry data to be used for correction
        # (these correspond to cell or nucleus intensities, though either prefix is configurable)
        self.intensity_prefix = params.get('intensity_prefix', intensity_prefix)

        # Extra commonly used shape information from experiment configuration
        self.region_shape = (
            self.config.region_height * self.config.tile_height,
            self.config.region_width * self.config.tile_width
        )
        self.tile_shape = self.config.tile_height, self.config.tile_width

        ############################
        # Configuration Validation #
        ############################

        if len(self.filter_params['percentile_range']) != 2:
            raise ValueError(
                '`percentile_range` in filter_params must be 2 item list (given = {})'
                .format(self.filter_params['percentile_range'])
            )
        for v in self.filter_params['percentile_range']:
            if not 0 <= v <= 1:
                raise ValueError(
                    '`percentile_range` value in filter_params values must be in [0, 1] (given = {})'
                    .format(v)
                )

        if self.prediction_params['step_size'] < 1:
            raise ValueError(
                '`step_size` in prediction_params must be > 1 (given = {})'
                .format(self.prediction_params['step_size'])
            )
        if self.model_params['type'] not in MODELS:
            raise ValueError(
                '`type` in model_params type must be one of {} (given = "{}")'
                .format(list(MODELS.keys()), self.model_params['type'])
            )

        self.data = None
        self.data_saved = False

    def _get_filter_masks(self, df, features):
        """Get masks for each filtered feature

        Args:
            df: Cytometry dataframe
            features: List of features to perform percentile filtering on
        Returns:
            DataFrame with column for each filter feature and values equal to boolean mask
                (where true means that the record is WITHIN the desired range)
        """
        # Compute low/high thresholds as dataframe like:
        #     feat_1 feat_2
        # p_lo    .1     10
        # p_hi   5.3    100
        ranges = df[features].quantile(q=self.filter_params['percentile_range'])

        # Stack masks horizontally as dataframe
        return pd.concat([
            df[c].between(ranges[c].iloc[0], ranges[c].iloc[1])
            for c in ranges
        ], axis=1)

    def _get_filter_features(self, df):
        feats = []
        for feat in self.filter_params['features']:
            # Test membership for feature as provided
            if feat in df:
                feats.append(feat)
                continue
            # Test membership when prefixed by channel markers
            ch_feat = self.intensity_prefix + feat
            if ch_feat in df:
                feats.append(ch_feat)
                continue
            raise ValueError('Feature "{}" not present in cytometry data'.format(feat))
        return feats

    def _prepare_prediction_features(self, df):

        # Feature group preparation function
        def prep(X, group):
            params = self.feature_params[group]
            if params is None:
                return None
            type = params.get('type')
            if not type or type == 'none':
                return None
            if type != 'polynomial':
                raise ValueError(
                    'Feature preparation type "{}" is not valid (must be None or "polynomial")'.format(type))
            return PolynomialFeatures(degree=params['degree'], include_bias=False).fit_transform(X)

        # Get features for each group (region and tile) or None, if either is disabled
        features = [x for x in [
            prep(df[['ry', 'rx']], 'region'),
            prep(df[['tile_y', 'tile_x']], 'index'),
            prep(df[['y', 'x']], 'tile')
        ] if x is not None]
        if len(features) == 0:
            raise ValueError('At least one of region or tile features must be enabled')

        # Concatenate features, if necessary
        features = features[0] if len(features) == 1 else np.hstack(tuple(features))
        return features

    def _estimate_model(self, X, y):
        # Fit regression model used to represent illumination surface
        args = self.model_params.get('args', {})
        est = MODELS[self.model_params['type']]['factory'](args)
        return est.fit(X, y)

    def get_illumination_models(self, region_index, df):
        n = len(df)
        ests = OrderedDict()
        for channel in self.channel_mapping.index:

            # Set list of features to filter on
            features = self._get_filter_features(df)

            # Restrict cell data to only records matching the given filters
            dfm = df[self._get_filter_masks(df, features).all(axis=1).values].copy()

            # If necessary, downsample modeling data to improve performance
            if len(dfm) > self.filter_params['max_cells']:
                dfm = dfm.sample(n=self.filter_params['max_cells'], random_state=SEED)

            if len(dfm) == 0:
                raise ValueError(
                    'Cytometry data empty after application of feature filters for channel {}'.format(channel))

            # Extract spatial cell features and prediction target
            X = self._prepare_prediction_features(dfm)
            y = dfm[self.intensity_prefix + channel]

            logger.debug(
                'Building illumination model for region %s, channel "%s" using %s cells (%s originally) [feature '
                'array shape = %s, response shape = %s]',
                region_index, channel, len(dfm), n, X.shape, y.shape
            )

            if np.isclose(y.mean(), 0):
                raise ValueError(
                    'Average {} channel intensity for region {} (across {} cells) is ~0, '
                    'making illumination correction impossible'
                    .format(region_index, channel, len(df))
                )
            y = y / y.mean()

            # Fit regression model used to represent illumination surface
            ests[channel] = self._estimate_model(X, y)
        return ests

    def _estimate_image(self, est):
        # Get prepared coordinate features for entire region and run predictions
        step_size = self.prediction_params['step_size']
        df, image_shape = _get_coordinate_features(self.region_shape, self.tile_shape, step_size)
        y = est.predict(self._prepare_prediction_features(df))

        # Some models may intentionally produce nans for pixels too far away from foreground
        # objects so set those values to 1 as this will lead to no change in correction
        y = np.where(np.isnan(y), 1., y)

        # Reshape from 1D to 2D image shape (which will be shape of entire region or a downsampled version of it)
        y = y.reshape(image_shape)

        # Upsample if necessary
        if step_size > 1:
            y = transform.resize(
                y, self.region_shape, mode='constant', order=1,
                anti_aliasing=False, preserve_range=False
            )

        assert y.shape == self.region_shape, \
            'Expecting result with shape {} but was {}'.format(self.region_shape, y.shape)
        return y.astype(np.float32)

    def get_illumination_images(self, ests):
        """Get an illumination image by predicting the intensity at each pixel across a region

        Args:
            ests: Illumination models
        Returns:
            Dictionary mapping source channels to a 3D float32 array with shape equal to
            (region height * tile height, region width * tile width)
        """
        imgs = OrderedDict()
        for channel, est in ests.items():
            imgs[channel] = self._estimate_image(est)

        if len(imgs) > 0:
            img = list(imgs.values())[0]
            logger.debug('Resulting illumination image array shape = %s (dtype = %s)', img.shape, img.dtype)

        return imgs

    def prepare_region_data(self, output_dir):
        if self.data is not None:
            return
        logger.info('Preparing illumination correction models')

        # Use whatever cytometry data was generated, whether it was for best
        # z planes, all planes, or a specific one
        df = function_data.get_cytometry_data(output_dir, self.config, mode='all')
        n = len(df)
        if df is None or n == 0:
            raise ValueError('Cytometry data cannot be empty in order to use it for illumination correction')

        self.data = {}
        for region_index, region_data in df.groupby('region_index'):
            ests = self.get_illumination_models(region_index, region_data)
            imgs = self.get_illumination_images(ests)
            self.data[region_index] = (imgs, ests)

    def save_region_data(self, output_dir):
        if self.data is None:
            raise ValueError('Region data cannot be saved until `prepare_region_data` is called')
        if self.data_saved:
            return None
        path = None
        for region_index, (imgs, ests) in self.data.items():
            # Stack 2D images on first axis to give 3D array
            img = np.stack(list(imgs.values()), 0)
            assert img.ndim == 3, 'Expecting 3D array, got shape {}'.format(img.shape)
            path = osp.join(output_dir, cytokit_io.get_illumination_function_path(region_index))
            cytokit_io.save_image(path, img)

        self.data_saved = True
        return osp.dirname(path or '')

    def _run(self, tile, tile_indices):
        # Get illumination image for this region
        imgs = self.data[tile_indices.region_index][0]

        # Get tile type information prior to multiplication
        dtype = tile.dtype
        dinfo = np.iinfo(dtype)

        # Determine starting offsets for tile slicing
        r, c = tile_indices.tile_y * self.config.tile_height, tile_indices.tile_x * self.config.tile_width

        # Loop through channel mapping and apply each adjustment
        tile = tile.astype(np.float32)
        for source_channel, target_channel in self.channel_mapping.to_dict().items():

            logger.debug('Applying correction from source channel "%s" to target "%s"', source_channel, target_channel)

            # Extract matching patch in illumination image
            img = imgs[source_channel][r:(r + self.config.tile_height), c:(c + self.config.tile_width)]

            # If application is across all channels, do single matrix division
            if target_channel == 'all':
                # Divide correction across HW dimensions of tile (must first add leading dims to match 5D tile)
                tile = tile / img[np.newaxis, np.newaxis, np.newaxis, :, :]

            # Otherwise, find target channel's coordinates within
            # tile and apply adjustment only to that slice
            else:
                cyc, ch = self.config.get_channel_coordinates(target_channel)
                tile[cyc, :, ch, :, :] = tile[cyc, :, ch, :, :] / img[np.newaxis, :, :]

        # Clip to range of original data type and convert
        tile = tile.clip(dinfo.min, dinfo.max).astype(dtype)

        return tile

    def save(self, tile_indices, output_dir, tile):
        # Overwrite the original preprocessed tile with corrected version
        path = cytokit_io.get_processor_img_path(
            tile_indices.region_index, tile_indices.tile_x, tile_indices.tile_y)
        cytokit_io.save_tile(osp.join(output_dir, path), tile, config=self.config)
        return path

