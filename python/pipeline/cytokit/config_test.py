import unittest
import cytokit
import os.path as osp


class TestConfig(unittest.TestCase):

    def _get_example_conf(self, file_type='yaml'):
        cytokit.set_config_default_filename('experiment.' + file_type)
        conf_dir = osp.join(cytokit.conf_dir, 'v0.1', 'examples', 'ex1')
        return cytokit.config.load(conf_dir)

    def test_load_conf(self):
        conf = self._get_example_conf(file_type='yaml')
        self.assertTrue(len(conf.channel_names) > 0)

    def test_get_region_point_coordinates(self):
        conf = self._get_example_conf()

        # get_region_point_coordinates(self, tile_coord, tile_point):
        rw, rh = conf.region_width, conf.region_height
        p = conf.get_region_point_coordinates((0, 0), (0, 0))
        self.assertEqual(p, (0, 0))
        p = conf.get_region_point_coordinates((rw-1, rh-1), (0, 0))
        self.assertEqual(p, (conf.tile_width * (rw-1), conf.tile_height * (rh-1)))

    def test_get_tile_point_coordinates(self):
        conf = self._get_example_conf()

        # get_tile_point_coordinates(self, region_coord)
        max_y = conf.region_height * conf.tile_height
        max_x = conf.region_width * conf.tile_width
        tile_coord, point_coord = conf.get_tile_point_coordinates((max_x-.01, max_y-.01))
        self.assertEqual(tile_coord, (conf.region_width-1, conf.region_height-1))
        self.assertAlmostEqual(point_coord[0], conf.tile_width - .01, 3)
        self.assertAlmostEqual(point_coord[1], conf.tile_height - .01, 3)



