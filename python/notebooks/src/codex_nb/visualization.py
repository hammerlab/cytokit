"""Notebook visualization utilities"""
from mizani import transforms
import matplotlib.pyplot as plt
import numpy as np


class Log1pTrans(transforms.trans):
        """Natural log plus one transformation for plotnine"""

        @staticmethod
        def transform(x):
            return np.log1p(x)

        @staticmethod
        def inverse(x):
            return (2 ** x) - 1


def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)