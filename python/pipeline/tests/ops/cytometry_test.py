import unittest
from cytokit.ops import cytometry


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
