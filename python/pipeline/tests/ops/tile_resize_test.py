import unittest
import numpy as np
from skimage.measure import compare_ssim
from cytokit.ops import tile_resize
from cytokit import simulation as cytokit_simulation


class TestTileResize(unittest.TestCase):

    def test_tile_resizer(self):
        from flowdec import data as fd_data
        image = fd_data.bars_25pct().data.astype(np.uint16)
        tile = image[np.newaxis, :, np.newaxis, ...]
        self.assertTrue(tile.ndim == 5, 'Expecting 5D shape but got {}'.format(tile.shape))
        config = cytokit_simulation.get_example_config(example_name='ex1')

        # Equate resolution to do fair comparison between anisotropic skimage resize
        # and tensorflow isotropic resize
        config._conf['acquisition']['axial_resolution'] = 1.
        config._conf['acquisition']['lateral_resolution'] = 1.

        # Set resizing factors and implementation to ensure the operation will run
        config._conf['processor']['tile_resize']['factors'] = [.75, .75, .75]
        config._conf['processor']['tile_resize']['implementation'] = 'skimage'
        resizer = tile_resize.CytokitTileResize(config).initialize()
        sk_res = resizer.run(tile)[0, :, 0]
        self.assertTrue(sk_res.ndim == 3, 'Expecting 3D shape but got {}'.format(sk_res.shape))

        # Run again using tensorflow implementation
        config._conf['processor']['tile_resize']['implementation'] = 'tensorflow'
        resizer = tile_resize.CytokitTileResize(config).initialize()
        tf_res = resizer.run(tile)[0, :, 0]
        self.assertTrue(tf_res.ndim == 3, 'Expecting 3D shape but got {}'.format(tf_res.shape))

        # Ensure that differences are both minor and present to such a degree that
        # the same implementation was not used for comparison
        ssim = compare_ssim(sk_res, tf_res)
        self.assertTrue(.99 <= ssim < 1, 'Expecting SSIM in [.99, 1) but got {}'.format(ssim))
