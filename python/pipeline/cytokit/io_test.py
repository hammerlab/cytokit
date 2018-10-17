import unittest
import cytokit
from cytokit import io as cytokit_io


class TestIo(unittest.TestCase):

    def test_raw_img_paths(self):
        cytokit.set_file_format_version(cytokit.FF_V01)

        # Test raw image path generation with default settings
        path = cytokit_io.get_raw_img_path(ireg=0, itile=0, icyc=0, ich=0, iz=0)
        self.assertEqual('Cyc1_reg1/1_00001_Z001_CH1.tif', path)

        # Test raw image path generation with explicit overrides on index mappings
        cytokit.set_raw_index_symlinks(dict(region={1: 2}, cycle={1: 5}, z={1: 3}, channel={1: 9}))
        path = cytokit_io.get_raw_img_path(ireg=0, itile=0, icyc=0, ich=0, iz=0)
        self.assertEqual('Cyc5_reg2/2_00001_Z003_CH9.tif', path)
        cytokit.set_raw_index_symlinks({})
