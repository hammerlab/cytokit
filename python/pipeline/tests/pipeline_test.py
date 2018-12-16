import unittest
import os.path as osp
import cytokit
import tempfile
from cytokit.cli import processor, analysis, operator
from cytokit.function import data as ck_fn
from cytokit import config as ck_config
from cytokit import io as ck_io


class TestConfig(unittest.TestCase):

    def test_pipeline_01(self):
        out_dir = tempfile.mkdtemp(prefix='cytokit_test_pipeline_01_')
        print('Initialized output dir {} for pipeline test 01'.format(out_dir))

        raw_dir = osp.join(cytokit.test_data_dir, 'experiment', 'cellular-marker-small', 'raw')
        config_dir = osp.join(cytokit.test_data_dir, 'experiment', 'cellular-marker-small', 'config')

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
        config = ck_config.load(config_dir)
        df = ck_fn.get_cytometry_data(out_dir, config, mode='best_z_plane')

        # Verify that the overall cell count and size found are in the expected ranges
        self.assertTrue(20 <= len(df) <= 25, 'Expecting between 20 and 25 cells, found {} instead'.format(len(df)))
        nuc_diam, cell_diam = df['nucleus_diameter'].mean(), df['cell_diameter'].mean()
        self.assertTrue(4 < nuc_diam < 6,
                        'Expecting mean nucleus diameter in [4, 6] um, found {} instead'.format(nuc_diam))
        self.assertTrue(8 < cell_diam < 10,
                        'Expecting mean cell diameter in [8, 10] um, found {} instead'.format(cell_diam))

        # The drift align dapi channels should be nearly identical across cycles, but in this case there are border
        # cells that end up with dapi=0 for cval=0 in drift compensation translation function so make the check
        # on a threshold (the ratio is < .5 with no drift compensation)
        dapi_ratio = df['ni:DAPI2'].mean() / df['ni:DAPI1'].mean()
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

        # ############# #
        # Montage Check #
        # ############# #

        # TODO:
        # - load and check segmented cells using CELL1 channel IoU comparison to cyto_cell_boundary
        # - check meta data in tile and montage image
        # - validate FCS reading
