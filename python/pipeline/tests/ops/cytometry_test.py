import unittest
import os.path as osp
import numpy as np
import pandas as pd
import cytokit
from skimage import segmentation
from cytokit import config as ck_config
from cytokit import io as ck_io
from cytokit.function import core as ck_core
from cytokit.ops import cytometry
from cytokit.ops import tile_generator
from cytokit.cytometry import cytometer
from numpy.testing import assert_array_equal


class TestCytometry(unittest.TestCase):

    def test_cytometry_channel_spec(self):

        # Test pre-defined channel names
        for k, v in {
            'cell_mask': ('cell_mask', cytometry.CHANNEL_COORDINATES['cell_mask']),
            'nucleus_boundary': ('nucleus_boundary', cytometry.CHANNEL_COORDINATES['nucleus_boundary']),
            'mychannel(0,3)': ('mychannel', (0, 3)),
            'mych._-;:(2 , 9)': ('mych._-;:', (2, 9))
        }.items():
            self.assertEqual(cytometry.get_channel_coordinates(k), v)

        # Test invvalid dynamic channel coordinates
        with self.assertRaises(ValueError):
            cytometry.get_channel_coordinates('ch(1-0)')
        with self.assertRaises(ValueError):
            cytometry.get_channel_coordinates('ch(1,0,1)')
        with self.assertRaises(ValueError):
            cytometry.get_channel_coordinates('ch*(1,0)')

    def test_quantification(self):
        """Validate quantification of objects in the "Random Shapes" test dataset"""

        exp_dir = osp.join(cytokit.test_data_dir, 'experiment', 'random-shapes')
        config_path = osp.join(exp_dir, 'config', 'experiment.yaml')
        config = ck_config.load(config_path)
        config.register_environment()

        # Pull shape of grid (i.e. region)
        region_shape = config.region_height, config.region_width

        #########################
        # Load tiles and original
        #########################

        # Load each tile for the experiment
        tiles = [
            tile_generator.CytokitTileGenerator(config, osp.join(exp_dir, 'raw'), region_index=0, tile_index=i).run()
            for i in range(config.n_tiles_per_region)
        ]
        self.assertEqual(tiles[0].ndim, 5, 'Expecting 5D tiles, got shape {}'.format(tiles[0].shape))

        # Load original image used to create individual tile images (i.e. at region scale) and compare
        # to a montage generated from the tiles just loaded
        img_mtv = ck_io.read_image(osp.join(exp_dir, 'validation', 'original_shapes_image.tif'))
        # Create montage from first channel (which contains object ids for reference)
        img_mtg = ck_core.montage(tiles, config)[0, 0, 0]
        assert_array_equal(img_mtg, img_mtv)

        # Classify objects as either free or on border
        # * First create a stacked image containing cleared tiles
        img_clr = np.stack([segmentation.clear_border(t[0, 0, 0]) for t in tiles])
        # Split ids into 2 groups based on the cleared stack image
        ids = np.unique(img_mtg[img_mtg > 0])
        ids_free = np.setdiff1d(np.unique(img_clr), [0])
        ids_brdr = np.setdiff1d(ids, ids_free)
        # Check that the background id is not included in any of the above
        self.assertTrue(np.all(ids_free > 0) and np.all(ids_brdr > 0) and np.all(ids > 0))

        ####################
        # Run quantification
        ####################

        def create_segments(im):
            # Create segmentation images as (z, ch, h, w)
            imb = segmentation.find_boundaries(im, mode='inner')
            segments = np.stack([im, im, imb, imb])[np.newaxis]
            assert segments.ndim == 4
            return segments

        # Quantify each tile image and concatenate results
        df = pd.concat([
            cytometer.Base2D.quantify(
                tiles[i], create_segments(tiles[i][0, 0, 0]),
                channel_names=config.channel_names,
                cell_intensity=['mean', 'median', 'sum', 'var'], nucleus_intensity=False,
                cell_graph=True, border_features=True, morphology_features=True
            ).assign(tile_x=c, tile_y=r, tile_index=i)
            for i, (r, c) in enumerate(np.ndindex(region_shape))
        ])

        #########################
        # Validate quantification
        #########################

        # Ensure that all objects in original image are also present in cytometry data
        self.assertTrue(
            len(np.intersect1d(ids, df['id'].unique())) == len(ids),
            'Object ids expected do not match those found\nIds found: {}\nIds expected: {}'
            .format(sorted(df['id'].unique()), sorted(ids))
        )

        # Check that objects classified as on border or not are correct
        assert_array_equal(sorted(df[df['cb:on_border'] > 0]['id'].unique()), sorted(ids_brdr))
        assert_array_equal(sorted(df[df['cb:on_border'] < 1]['id'].unique()), sorted(ids_free))

        # Loop through objects identified and validate each one
        for i, r in df.iterrows():
            # Fetch tile image from tile list and make sure that size of cell returned matches that in image
            area = (tiles[r['tile_index']][0, 0, 0] == r['id']).sum()
            self.assertEquals(r['cm:size'], area)

            # For each channel validate that:
            # - mean and median equal the id times channel index (1-based)
            # - sum equals area times id times channel index
            # - variance is 0
            for j, c in enumerate(config.channel_names):
                for f in ['mean', 'median']:
                    self.assertEquals(r['id'] * (j + 1), r['ci:{}:{}'.format(c, f)])
                self.assertEquals(r['id'] * (j + 1) * area, r['ci:{}:sum'.format(c)])
                self.assertEquals(0, r['ci:{}:var'.format(c)])

