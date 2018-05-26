# Docker Instructions

```

# Building a development container
nvidia-docker build --no-cache -t codex-dev -f docker/Dockerfile.devel .

# Run development container
export CODEX_DATA_DIR=/data/disk1/
export CODEX_REPO_DIR=/home/eczech/repos/codex
nvidia-docker run -rm -ti -p 8888:8888 \
-v $CODEX_DATA_DIR:/lab/data \
-v $CODEX_REPO_DIR:/lab/repos/codex \
codex-dev

```