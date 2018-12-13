## Cytokit Cell Size Analysis


#### Cytoflow Setup

*Modified from https://cytoflow.readthedocs.io/en/latest/INSTALL.html*

```
conda create -n cytoflow python=3.5.2
conda activate cytoflow
cd ~/repos/misc/python/cytoflow/
wget https://github.com/bpteague/cytoflow/releases/download/0.9.3/_Logicle.cpython-35m-darwin.so
mv _Logicle.cpython-35m-darwin.so cytoflow/utility/logicle_ext/
export NO_LOGICLE=True
pip install -e .
pip install jupyter
```
