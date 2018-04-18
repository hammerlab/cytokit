

from codex.ops.op import CodexOp
from codex.miq import prediction
from codex import data as codex_data
import numpy as np
import logging

logger = logging.getLogger(__name__)


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
        self.patch_size = patch_size
        self.n_classes = n_classes

    def initialize(self):
        model_path = codex_data.initialize_best_focus_model()
        self.mqiest = prediction.ImageQualityClassifier(model_path, self.patch_size, self.n_classes)
        return self

    def shutdown(self):
        self.mqiest._sess.close()
        return self

    def run(self, tile):
        # ncyc, nx, ny, nz, nch = self.config.tile_dims()

        # Tile should have shape (cycles, z, channel, height, width)
        focus_cycle, focus_channel = self.config.best_focus_reference()

        # Subset to 3D stack based on reference cycle and channel
        img = tile[focus_cycle, :, focus_channel, :, :]

        classifications = []
        probabilities = []
        for iz in range(img.shape[0]):
            pred = self.mqiest.predict(img[iz])
            # print('\tshape = ', img[iz].shape, 'res = ', pred.probabilities)
            classifications.append(pred.predictions)
            probabilities.append(pred.probabilities)
        best_z = np.argmin(classifications)
        logger.info('Best focal plane: z = {} (scores: {})'.format(best_z, classifications))

        # Return tile subset to best z plane as well as other context
        return tile[:, best_z, :, :, :], best_z, classifications, probabilities






