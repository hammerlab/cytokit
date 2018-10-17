import unittest
from skimage import data as sk_data
import numpy as np
import scipy as sp
from cytokit.ops import drift_compensation
import tensorflow as tf
from flowdec import fft_utils_np, fft_utils_tf, test_utils
from numpy.testing import assert_array_equal, assert_almost_equal


class TestDriftCompensation(unittest.TestCase):

    def _test_2d_translation_calculator(self, op, img, shift):
        # Offset image by known shift
        offset_img = sp.ndimage.shift(img, shift)

        # Recover shift noting that it is in the opposite direction
        res = -op.run(img, offset_img)['translation']

        # Ensure that recovered shift is within .5 pixels of actual
        diff = shift - res
        msg = 'Inferred shift = {}, Actual shift = {}'.format(res, shift)
        self.assertTrue(np.all(diff <= .5), msg=msg)

    def test_2d_translation_calculator(self):
        img = sk_data.camera()
        op = drift_compensation.TranslationCalculator(2).initialize()

        for shift in [
            (0, 0), (1, 0), (0, 1), (-1, 0), (0, -1),
            (50, 5), (5, 50), (-50, 5), (-5, 50), (-5, -50),
            (25.8, .4), (25.4, .4), (25.5, 1.5),
            (-25.8, .4), (-25.4, -.4), (-.6, -.4)
        ]:
            self._test_2d_translation_calculator(op, img, shift)

    def test_2d_translation_applier(self):
        img = np.reshape(np.arange(1, 41), (5, 8))
        op = drift_compensation.TranslationApplier(2).initialize()

        for shift in [(0, 0), (1, 1), (-1, -1),(10, -20), (25, 15)]:
            # Run shift on image
            res = op.run(img, shift)['result']

            # Make sure that result equals same thing from scipy version
            actual = sp.ndimage.shift(img, shift, output=res.dtype)
            assert_array_equal(res, actual)
