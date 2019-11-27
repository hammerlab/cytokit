import cytokit
import sys
import subprocess
import os.path as osp

CP_CLI = osp.join(cytokit.ext_dir, 'cellprofiler', 'cpcli.py')
CP_PY = '/opt/conda/envs/cellprofiler/bin/python'


def to_args(kwargs):
    return ['--{}={}'.format(k.replace('_', '-'), str(v)) for k, v in kwargs.items()]


def run_quantification(**kwargs):
    args = to_args(kwargs)
    cmd = CP_PY + ' ' + CP_CLI + ' ' + ' '.join(args)
    rc = subprocess.run(cmd, stderr=sys.stderr, stdout=sys.stdout, shell=True)
    if rc.returncode != 0:
        raise ValueError('CellProfiler cli command returned code {}; Command:\n{}'.format(rc.returncode, cmd))
