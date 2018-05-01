"""Notebook visualization utilities"""
import mizani
import numpy as np


class Log1pTrans(mizani.transforms.trans):
        """Natural log plus one transformation for plotnine"""

        @staticmethod
        def transform(x):
            return np.log1p(x)

        @staticmethod
        def inverse(x):
            return (2 ** x) - 1