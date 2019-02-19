import unittest
from cytokit import data as ck_data


class TestData(unittest.TestCase):

    def test_data_functions(self):
        # Test model initialization functions (which download weights or other model-specific files)
        ck_data.get_cache_dir()
        ck_data.initialize_best_focus_model()
        ck_data.initialize_cytometry_2d_model()
