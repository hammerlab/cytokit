"""Utility functions for analyzing keyence dataset metadata"""
import re
import os
import os.path as osp
import pandas as pd
import warnings


def get_zpitch_cmd(bcf_path):
    """Command to return single integer value representing z step size from bcf file"""
    cmd = 'unzip -c ' + bcf_path
    cmd += " | grep Pitch | grep -o '<Pitch Type=\\\"System.Int32\\\">[0-9]\\{1,4\\}</Pitch>' | grep -o '>[0-9]\{1,4\}' | sed 's/>//g'"
    return cmd


def analyze_keyence_dataset(path):
    bcf_files = []
    df = []
    for f in os.listdir(path):
        if f.endswith('.bcf'):
            bcf_files.append(osp.join(path, f))
        if not f.endswith('.tif'):
            continue

        parts = re.findall('_([0-9]{5})_Z([0-9]{3})_CH([0-9]{1}).tif', f)
        if not parts:
            continue
        df.append(parts[0] + (f,))
    df = pd.DataFrame(df, columns=['tile', 'z', 'ch', 'file'])

    df['z_pitch'] = None
    if len(bcf_files) != 1:
        warnings.warn('Found {} bcf files in path {} so bcf processing will be skipped'.format(len(bcf_files), path))
    else:
        cmd = get_zpitch_cmd(bcf_files[0])
        try:
            df['z_pitch'] = int(os.popen(cmd).read())
        except:
            warnings.warn('Failed to extract z pitch from bcf file {} (command = {})'.format(bcf_files[0], cmd))
    return df
