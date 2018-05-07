from codex import io as codex_io
import pandas as pd
import os
import re

def get_best_focal_planes(config, data_dir, ireg=0):
    """Get data frame containing information about best focal planes for each tile
    
    Note: This is a temporary implementation that should ultimately extract this
        information from dedicated files rather than inferring it from file names
    """
    best_focus_dir = os.path.join(data_dir, 'bestFocus')
    prefix = 'reg{:03d}'.format(ireg + 1)
    m = re.compile(prefix + '_X(\d+)_Y(\d+)_Z(\d+).tif')
    res = []
    for bf in os.listdir(best_focus_dir):
        if not bf.startswith(prefix + '_X'):
            continue
        groups = m.match(bf).groups()
        if len(groups) != 3:
            raise ValueError('X/Y/Z coords could not be extracted from file "{}"'.format(bf))
        tx, ty, tz = groups
        res.append((int(tx), int(ty), int(tz)))
        
    if len(res) != config.region_width * config.region_height:
        raise ValueError('Expecting best z planes for {} tiles but only found {}'\
            .format(config.region_width * config.region_height, len(res)))
    return pd.DataFrame(res, columns=['x', 'y', 'z'])