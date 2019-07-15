# Docker Instructions


Pull and run the production container:

```bash
# Building a development container
nvidia-docker pull eczech/cytokit:latest

# Run and open ports for jupyter, tensorboard, dask, and dash
nvidia-docker run --rm -ti -p 8888:8888 -p 6006:6006 -p 8787:8787 -p 8050:8050 \
-v $CYTOKIT_DATA_DIR:/lab/data \
eczech/cytokit
```


Build and run the development container:

```bash

# Building a development container
nvidia-docker build -t cytokit-dev -f Dockerfile.dev .

# Run development container
export CYTOKIT_DATA_DIR=/data/disk1/
export CYTOKIT_REPO_DIR=/home/eczech/repos/cytokit

nvidia-docker run --rm -ti -p 8888:8888 -p 6006:6006 -p 8787:8787 -p 8050:8050 \
-v $CYTOKIT_DATA_DIR:/lab/data \
-v $CYTOKIT_REPO_DIR:/lab/repos/cytokit \
cytokit-dev

```

DockerHub Deployment:

```bash
# Build the image and find its ID (alternatively, run the build with the release tag and skip next step)
docker images 

# Tag with release number
docker tag 1f265be86a54 eczech/cytokit:0.1.1

# Login and push
echo $PASSWORD | docker login --username eczech --password-stdin
docker push eczech/cytokit
```