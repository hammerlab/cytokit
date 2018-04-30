import unittest
from codex.ops import best_focus


class TestBestFocus(unittest.TestCase):

    def test_best_index_fn(self):
        # Test easy case
        self.assertEqual(best_focus.get_best_z_index([8, 7, 1, 5, 11]), 2)

        # Test ties with odd number of elements
        self.assertEqual(best_focus.get_best_z_index([1, 1, 1, 1, 1]), 2)
        self.assertEqual(best_focus.get_best_z_index([1, 1, 1, 1, 0]), 4)
        self.assertEqual(best_focus.get_best_z_index([1, 1, 1, 0, 0]), 3)
        self.assertEqual(best_focus.get_best_z_index([1, 1, 0, 0, 0]), 2)
        self.assertEqual(best_focus.get_best_z_index([1, 0, 0, 0, 0]), 2)
        self.assertEqual(best_focus.get_best_z_index([0, 0, 0, 0, 0]), 2)

        # Test ties with even number of elements
        self.assertEqual(best_focus.get_best_z_index([1, 1, 1, 1]), 2)
        self.assertEqual(best_focus.get_best_z_index([1, 1, 1, 0]), 3)
        self.assertEqual(best_focus.get_best_z_index([1, 1, 0, 0]), 2)
        self.assertEqual(best_focus.get_best_z_index([1, 0, 0, 0]), 2)
        self.assertEqual(best_focus.get_best_z_index([0, 0, 0, 0]), 2)

