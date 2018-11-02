import unittest
from skimage import data as sk_data
import numpy as np
import scipy as sp
from cytokit.ops import drift_compensation
from cytokit import simulation
from numpy.testing import assert_array_equal


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

        for shift in [(0, 0), (1, 1), (-1, -1), (10, -20), (25, 15)]:
            # Run shift on image
            res = op.run(img, shift)['result']

            # Make sure that result equals same thing from scipy version
            actual = sp.ndimage.shift(img, shift, output=res.dtype)
            assert_array_equal(res, actual)

    def test_5d_tile_translation(self):
        # Load 3D float image (with min/max = 0/65535) and convert to uint16 as well as
        # specify shift in XY direction for first channel in first and second cycle
        tile, config, info = simulation.load_simulated_bars_experiment(blur=False, shift=(0, -5, 15))
        tile = tile.astype(np.uint16)

        # Run compensation
        op = drift_compensation.CytokitDriftCompensator(config).initialize()
        res = op.run(tile)

        # Verify that original channels differ across cycles while compensated channels do not
        self.assertFalse(np.array_equal(tile[0, :, 0].max(axis=0), tile[1, :, 0].max(axis=0)))
        self.assertFalse(np.array_equal(tile[0, :, 1].max(axis=0), tile[1, :, 1].max(axis=0)))
        self.assertTrue(np.array_equal(res[0, :, 0].max(axis=0), res[1, :, 0].max(axis=0)))
        self.assertTrue(np.array_equal(res[0, :, 1].max(axis=0), res[1, :, 1].max(axis=0)))
