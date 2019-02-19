import unittest
import math
from cytokit import math as ck_math


class TestMath(unittest.TestCase):

    def test_circularity(self):
        a1, p1 = math.pi, math.pi * 2  # Perfect circule
        a2, p2 = math.pi, math.pi * 2 + 1  # Non-perfect circle
        self.assertEqual(ck_math.circularity(a1, p1), 1)
        self.assertTrue(ck_math.circularity(a2, p2) < 1)
