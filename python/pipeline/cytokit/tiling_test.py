import unittest
from cytokit import tiling

class TestTiling(unittest.TestCase):

    def _to_coordinates(self, index, tiler, w, h):
        x, y = tiler.coordinates_from_index(index, w, h)
        actual = tiler.index_from_coordinates(x, y, w, h)
        self.assertEqual(actual, index)

    def test_to_coordinates(self):
        cases = [
            # index, width, height
            (0, 5, 4),
            (4, 5, 4),
            (5, 5, 4),
            (9, 5, 4),
            (10, 5, 4),
            (19, 5, 4),
            (0, 3, 3),
            (2, 3, 3),
            (3, 3, 3)
        ]
        for t in tiling.TILINGS.values():
            for c in cases:
                self._to_coordinates(c[0], t, c[1], c[2])

    def _to_index(self, x, y, tiler, w, h):
        index = tiler.index_from_coordinates(x, y, w, h)
        actual = tiler.coordinates_from_index(index, w, h)
        self.assertEqual(actual, (x, y))

    def test_to_index(self):
        cases = [
            # x, y, width, height
            (0, 0, 5, 4),
            (4, 0, 5, 4),
            (4, 1, 5, 4),
            (0, 1, 5, 4),
            (0, 2, 5, 4),
            (4, 4, 5, 4),
            (0, 0, 3, 3),
            (2, 0, 3, 3),
            (2, 1, 3, 3)
        ]
        for t in tiling.TILINGS.values():
            for c in cases:
               self._to_index(c[0], c[1], t, c[2], c[3])