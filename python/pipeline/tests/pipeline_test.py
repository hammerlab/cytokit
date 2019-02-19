import unittest
import os.path as osp
import cytokit
import tempfile
from cytokit.cli import processor, analysis, operator
from cytokit.function import data as ck_fn
from cytokit import config as ck_config
from cytokit import io as ck_io
from skimage import io as sk_io
import logging


class TestConfig(unittest.TestCase):

    def test_pipeline_01(self):
        out_dir = tempfile.mkdtemp(prefix='cytokit_test_pipeline_01_')
        logging.info('Initialized output dir {} for pipeline test 01'.format(out_dir))

        raw_dir = osp.join(cytokit.test_data_dir, 'experiment', 'cellular-marker-small', 'raw')
        val_dir = osp.join(cytokit.test_data_dir, 'experiment', 'cellular-marker-small', 'validation')
        config_dir = osp.join(cytokit.test_data_dir, 'experiment', 'cellular-marker-small', 'config')
        config = ck_config.load(config_dir)

        # Run processor and extractions/aggregations
        processor.Processor(data_dir=raw_dir, config_path=config_dir).run_all(output_dir=out_dir)
        operator.Operator(data_dir=out_dir, config_path=config_dir).run_all()
        analysis.Analysis(data_dir=out_dir, config_path=config_dir).run_all()

        # ##################### #
        # Processor Data Checks #
        # ##################### #
        df = ck_fn.get_processor_data(out_dir)['drift_compensator']
        # Expect one drift comp record since there are two cycles and one is the reference
        self.assertEqual(len(df), 1)
        # Expecting 12 row and -3 col translation introduced in synthetic data
        self.assertEqual(df.iloc[0]['translation'], [12, -3])

        df = ck_fn.get_processor_data(out_dir)['focal_plane_selector']
        # Expect one focal selection record (there is only 1 tile in experiment and these
        # records are per-tile)
        self.assertEqual(len(df), 1)
        # Expecting second of 3 z planes to have the best focus (data was generated this way)
        self.assertEqual(df.iloc[0]['best_z'], 1)

        # ##################### #
        # Cytometry Stats Check #
        # ##################### #
        df = ck_fn.get_cytometry_data(out_dir, config, mode='best_z_plane')

        # Verify that the overall cell count and size found are in the expected ranges
        self.assertTrue(20 <= len(df) <= 25, 'Expecting between 20 and 25 cells, found {} instead'.format(len(df)))
        nuc_diam, cell_diam = df['nm:diameter'].mean(), df['cm:diameter'].mean()
        self.assertTrue(4 < nuc_diam < 6,
                        'Expecting mean nucleus diameter in [4, 6] um, found {} instead'.format(nuc_diam))
        self.assertTrue(8 < cell_diam < 10,
                        'Expecting mean cell diameter in [8, 10] um, found {} instead'.format(cell_diam))

        # The drift aligned dapi channels should be nearly identical across cycles, but in this case there are border
        # cells that end up with dapi=0 for cval=0 in drift compensation translation function so make the check
        # on a threshold (the ratio is < .5 with no drift compensation)
        dapi_ratio = df['ni:DAPI2:mean'].mean() / df['ni:DAPI1:mean'].mean()
        self.assertTrue(.8 < dapi_ratio <= 1,
                        'Expecting cycle 2 DAPI averages to be similar to cycle 1 DAPI after drift compensation, '
                        'found ratio {} (not in (.8, 1])'.format(dapi_ratio))

        # Check that all records are for single z plane (with known best focus)
        self.assertEqual(df['z'].nunique(), 1)
        self.assertEqual(int(df['z'].unique()[0]), 1)

        # Verify that single cell image generation works
        df = ck_fn.get_single_cell_image_data(out_dir, df, 'best_z_segm', image_size=(64, 64))
        self.assertEqual(df['image'].iloc[0].shape, (64, 64, 3))
        self.assertTrue(df['image'].notnull().all())

        # ################## #
        # Segmentation Check #
        # ################## #
        # Load extract with object masks
        img, meta = ck_io.read_tile(
            osp.join(out_dir, ck_io.get_extract_image_path(ireg=0, tx=0, ty=0, name='best_z_segm')),
            return_metadata=True
        )
        # Ensure that the 8 channels set for extraction showed up in the resulting hyperstack
        self.assertEqual(len(meta['labels']), 8)

        # Verify that IoU for both nuclei and cell masks vs ground-truth is > 80%
        img_seg_cell = img[0, 0, meta['labels'].index('cyto_cell_mask')]
        img_seg_nucl = img[0, 0, meta['labels'].index('cyto_nucleus_mask')]
        img_val_cell = sk_io.imread(osp.join(val_dir, 'cells.tif'))
        img_val_nucl = sk_io.imread(osp.join(val_dir, 'nuclei.tif'))

        def iou(im1, im2):
            return ((im1 > 0) & (im2 > 0)).sum() / ((im1 > 0) | (im2 > 0)).sum()
        self.assertGreater(iou(img_seg_cell, img_val_cell), .8)
        self.assertGreater(iou(img_seg_nucl, img_val_nucl), .8)

        # ############# #
        # Montage Check #
        # ############# #
        # Load montage and check that it has the same dimensions as the extract image above,
        # since there is only one tile in this case
        img_mntg = ck_io.read_tile(osp.join(out_dir, ck_io.get_montage_image_path(ireg=0, name='best_z_segm')))
        self.assertEqual(img.shape, img_mntg.shape)
        self.assertEqual(img.dtype, img_mntg.dtype)
