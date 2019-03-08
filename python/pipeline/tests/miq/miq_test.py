import unittest
import numpy as np
from skimage import filters
from cytokit import data as ck_data
from cytokit.ops import op
from cytokit.miq import prediction
from cytokit.ops.best_focus import DEFAULT_PATCH_SIZE, DEFAULT_N_CLASSES


class TestMiq(unittest.TestCase):

    def test_image_quality_classification(self):
        model_path = ck_data.initialize_best_focus_model()
        # Make sure to use default op TF config since GPU memory errors during testing will
        # occur otherwise (i.e. when gpu_options.allow_growth is not set to True)
        miqest = prediction.ImageQualityClassifier(
            model_path, DEFAULT_PATCH_SIZE, DEFAULT_N_CLASSES, session_config=op.get_tf_config(None))

        img1 = np.zeros((DEFAULT_PATCH_SIZE*2, DEFAULT_PATCH_SIZE*2))
        img1[6:10, 6:10] = 1
        img2 = filters.gaussian(img1, 5)

        # Get class prediction for images (0 = best quality)
        max1 = np.argmax(miqest.predict(img1).probabilities)
        max2 = np.argmax(miqest.predict(img2).probabilities)

        # Assert that probabilities (for ~11 classes) for original image have
        # max at first element (i.e. the highest quality class)
        self.assertEqual(
            max1, 0,
            msg='Expecting highest quality class prediction for static image (got class {})'
            .format(max1)
        )

        # Ensure that max class for blurred image (where higher class = lower quality) is greater than original
        self.assertTrue(
            max2 > max1,
            msg='Expecting quality class of original image ({}) to be less than quality class of blurred image ({})'
            .format(max1, max2)
        )
