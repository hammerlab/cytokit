import os.path as osp
from cytokit.ops.op import CytokitOp, get_tf_config
from cytokit.miq import prediction
from cytokit import data as cytokit_data
from cytokit import io as cytokit_io
import tensorflow as tf
import numpy as np
import logging

logger = logging.getLogger(__name__)


class CytokitFocalPlaneSelector(CytokitOp):
    """Best focal plan selection operation

    Args:
        config: Cytokit configuration
        patch_size: size of patches within image to estimate quality for; defaults to 84, same as default
            in originating classifier project
        n_classes: number of different quality strata to predict logits for; defaults to 11, same as default
            in originating classifier project
        save_tile: Indicates whether or not best-focus tiles (with single z-plane) should be saved; default false

    Note:
        See https://github.com/google/microscopeimagequality for more details on the classifier used by this operation
    """

    def __init__(self, config, patch_size=84, n_classes=11, save_tile=False):
        super().__init__(config)
        self.mqiest = None
        self.graph = None

        params = config.best_focus_params
        self.patch_size = params.get('patch_size', patch_size)
        self.n_classes = params.get('n_classes', n_classes)
        self.focus_cycle, self.focus_channel = config.get_channel_coordinates(params['channel'])
        self.save_tile = params.get('save_tile', save_tile)

    def initialize(self):
        model_path = cytokit_data.initialize_best_focus_model()
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

        scores = []
        classes = []
        for iz in range(nz):
            pred = self.mqiest.predict(img[iz])
            # Append n_classes length array of class probabilities ordered from 0 to n_classes
            # where 0 is the best possible quality and n_classes the worst
            scores.append(pred.probabilities)
            classes.append(np.argmax(pred.probabilities))

        # Calculate scores as probability weighted sum of (reversed) class indexes, giving one score per z-plane
        scores = np.dot(np.array(scores), np.arange(self.n_classes)[::-1])
        assert len(scores) == nz, \
            'Expecting {} scores but only {} were found (scores = {})'.format(nz, len(scores), scores)

        # Reverse class designations
        classes = self.n_classes - np.array(classes) - 1

        # Determine best z plane as index with highest score
        best_z = np.argmax(scores)

        # Record and log classification information
        self.record({'scores': scores, 'classes': classes, 'best_z': best_z})
        logger.debug('Best focal plane: z = {} (score: {})'.format(best_z, scores.max()))

        # Subset tile to best focal plane
        best_focus_tile = tile[:, [best_z], :, :, :]

        # Return best focus tile and other context
        return best_focus_tile, best_z, scores

    def save(self, tile_indices, output_dir, data):
        region_index, tile_index, tx, ty = tile_indices
        best_focus_tile, best_z, scores = data
        path = cytokit_io.get_best_focus_img_path(region_index, tx, ty, best_z)
        if self.save_tile:
            cytokit_io.save_tile(osp.join(output_dir, path), best_focus_tile, config=self.config)
        return [path]
