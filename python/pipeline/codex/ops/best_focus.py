from codex.ops.op import CodexOp, get_tf_config
from codex.miq import prediction
from codex import data as codex_data
import tensorflow as tf
import numpy as np
import logging

logger = logging.getLogger(__name__)


def get_best_z_index(classifications):
    """Get optimal z index based on quality classifications

    Ties are broken using the index nearest to the center of the sequence
    of all possible z indexes
    """
    nz = len(classifications)
    best_score = np.min(classifications)
    top_z = np.argwhere(np.array(classifications) == best_score).ravel()
    return top_z[np.argmin(np.abs(top_z - (nz // 2)))]


class CodexFocalPlaneSelector(CodexOp):
    """Best focal plan selection operation

    Args:
        config: Codex configuration
        patch_size: size of patches within image to estimate quality for; defaults to 84, same as default
            in originating classifier project
        n_classes: number of different quality strata to predict logits for; defaults to 11, same as default
            in originating classifier project

    Note:
        See https://github.com/google/microscopeimagequality for more details on the classifier used by this operation
    """

    def __init__(self, config, patch_size=84, n_classes=11):
        super(CodexFocalPlaneSelector, self).__init__(config)
        self.mqiest = None
        self.graph = None

        params = config.best_focus_params
        self.patch_size = params.get('patch_size', patch_size)
        self.n_classes = params.get('n_classes', n_classes)
        self.focus_cycle, self.focus_channel = config.get_channel_coordinates(params['channel'])

    def initialize(self):
        model_path = codex_data.initialize_best_focus_model()
        self.graph = tf.Graph()
        self.mqiest = prediction.ImageQualityClassifier(
            model_path, self.patch_size, self.n_classes,
            graph=self.graph, session_config=get_tf_config(self)
        )
        return self

    def shutdown(self):
        self.mqiest._sess.close()
        return self

    def _run(self, tile, **kwargs):
        # Subset to 3D stack based on reference cycle and channel
        # * tile should have shape (cycles, z, channel, height, width)
        img = tile[self.focus_cycle, :, self.focus_channel, :, :]
        nz = img.shape[0]

        classifications = []
        probabilities = []
        for iz in range(nz):
            pred = self.mqiest.predict(img[iz])
            classifications.append(pred.predictions)
            probabilities.append(pred.probabilities)
        best_z = get_best_z_index(classifications)
        self.record({'classifications': classifications, 'best_z': best_z})
        logger.debug('Best focal plane: z = {} (classifications: {})'.format(best_z, classifications))

        # Subset tile to best focal plane
        best_focus_tile = tile[:, [best_z], :, :, :]

        # Return best focus tile and other context
        return best_focus_tile, best_z, classifications, probabilities

    def save(self, tile_indices, output_dir, data):
        region_index, tile_index, tx, ty = tile_indices
        best_focus_tile, best_z, classifications, probabilities = data
        path = codex_io.get_best_focus_img_path(region_index, tx, ty, best_z)
        codex_io.save_tile(osp.join(output_dir, path), best_focus_tile)
        return [path]
