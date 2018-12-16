import unittest
import os.path as osp
import cytokit
import tempfile
from cytokit.cli import processor
from cytokit.function import data as ck_fn
import logging
logger = logging.getLogger(__name__)


class TestConfig(unittest.TestCase):

    def test_experiement_01(self):
        out_dir = tempfile.mkdtemp(prefix='cytokit_pipeline_test')
        logger.info('Experiment temporary output directory: %s', out_dir)
        proc = processor.Processor(
            data_dir=osp.join(cytokit.test_data_dir, 'experiment', 'cellular-marker-small', 'raw'),
            config_path=osp.join(cytokit.test_data_dir, 'experiment', 'cellular-marker-small', 'config')
        )
        proc.run_all(output_dir=out_dir)

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

        # TODO:
        # - load and check segmented cells (add cyto_cell_object to conf)
        # - load and check quantification of channel 4
        # - load cytometry data frame and check stats
