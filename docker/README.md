# Docker Instructions


Running the production container:

```bash
# Building a development container
nvidia-docker pull hammerlab/cytokit:latest

nvidia-docker run --rm -ti -p 8888:8888 -p 6006:6006 -p 8787:8787 -p 8050:8050 \
-v $CYTOKIT_DATA_DIR:/lab/data \
eczech/cytokit
```

Running the development container:

```bash

# Building a development container
nvidia-docker build -t cytokit-dev -f Dockerfile.dev .

# Run development container
export CYTOKIT_DATA_DIR=/data/disk1/
export CYTOKIT_REPO_DIR=/home/eczech/repos/cytokit

# Run and open ports for jupyter, tensorboard, dask, and dash
nvidia-docker run --rm -ti -p 8888:8888 -p 6006:6006 -p 8787:8787 -p 8050:8050 \
-v $CYTOKIT_DATA_DIR:/lab/data \
-v $CYTOKIT_REPO_DIR:/lab/repos/cytokit \
cytokit-dev

```