# Docker Instructions

```

# Building a development container
nvidia-docker build -t codex-dev -f Dockerfile.dev .

# Run development container
export CODEX_DATA_DIR=/data/disk1/
export CODEX_REPO_DIR=/home/eczech/repos/codex
export CVUTILS_REPO_DIR=/home/eczech/repos/cvutils
nvidia-docker run --rm -ti -p 8888:8888 -p 6006:6006 \
-v $CODEX_DATA_DIR:/lab/data \
-v $CODEX_REPO_DIR:/lab/repos/codex \
-v $CVUTILS_REPO_DIR:/lab/repos/cvutils 
codex-dev

```