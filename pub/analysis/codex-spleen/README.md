## CODEX Data Analysis

This directory contains configurations necessary for analyzing the [public datasets](http://welikesharingdata.blob.core.windows.net/forshare/index.html) from [Deep profiling of mouse splenic architecture with CODEX multiplexed imaging](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6086938/) (Goltsev et al. 2018).

The pipeline configured here will process the BALBc-1 sample and make the quantified cell data available through Cytokit Explorer and Cell Profiler Analyst (CPA).

### CPA Instructions

A SQLite database will be generated at ```$OUTPUT_DIR/cytometry/cellprofiler/results``` and if the host (i.e. the machine running the Docker container) is not X11-enabled then there are several possible ways to use CPA given that the image paths (and others) written to the SQLite DB will all be based on the filesystem within the docker container.  Some options are:

1. Copy the data at ```$OUTPUT_DIR/cytometry/cellprofiler/results``` on the host to any machine with a windows manager (Mac, Windows, Ubuntu Desktop, etc.) at a path equivalent to the path used in the container (i.e. create ```/lab/data``` root directory and place everything under it).  For example: ```mac> rsync -rP myhost:/data/disk1/20180101_codex_spleen/20180101_codex_mouse_spleen_balbc_slide1/output/v00/cytometry/cellprofiler/* /lab/data/20180101_codex_spleen/20180101_codex_mouse_spleen_balbc_slide1/output/v00/cytometry/cellprofiler/```
2. Create a network drive (e.g. w/ samba on Mac) shared with the host running docker and symlink a ```/lab/data``` directory to it.  For example on a Mac, if the samba mount is at ```/Volumes/disk1``` then the data could linked with ```mac> mkdir /lab; cd /lab; ln -s /Volumes/disk1/ /lab/data```

After copying or linking the data, CPA can be pointed at the ```CPA_Exp_Cell.properties``` properties file and run as usual.