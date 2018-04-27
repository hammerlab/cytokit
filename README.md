[![Build Status](https://travis-ci.org/hammerlab/codex.svg?branch=master)](https://travis-ci.org/hammerlab/codex)
[![Coverage Status](https://coveralls.io/repos/github/hammerlab/codex/badge.svg?branch=master)](https://coveralls.io/github/hammerlab/codex?branch=master)

## CODEX

Processing worfklows for CODEX cytometric imaging data

### Deconvolution

See [Standalone Deconvolution Instructions](python/standalone/deconvolution) for information on how to run image deconvolution inline with the rest of the CODEX processing pipeline.


### Installation

Currently, the simplest installation method is through docker.  First, follow the
 [installation instructions for nvidia-docker](https://github.com/NVIDIA/nvidia-docker#quickstart) 
 and once that is operational the codex can be installed and run as follows:
 
```bash
cd $REPOS
git clone https://github.com/hammerlab/codex.git
cd codex/docker

# Build the container
nvidia-docker build -t codex

# Run the container 
nvidia-docker run -td -p 8888:8888 --name codex codex

# Open a shell within container
nvidia-docker exec -it codex /bin/bash
``` 

Alternatively, the steps in [Dockerfile.gpu](docker/Dockerfile.gpu) could be followed individually to 
install the necessary dependencies without using Docker (better docs/scripts for this are a WIP)

### Command Line

To run the codex processing steps in a container, a ```codex-processor``` command is present 
that can be run as follows:

```bash

# Open a shell within container
nvidia-docker exec -it codex /bin/bash

root@containerid> codex-processor localhost -- --help

Args:
    data_dir: Path to directoring containing raw acquisition data files
    output_dir: Directory to save results in; will be created if it does not exist
    region_indexes: 1-based sequence of region indexes to process; can be specified as:
        - None: Region indexes will be inferred from experiment configuration
        - str or int: A single value will be interpreted as a single index
        - tuple: A two-item tuple will be interpreted as a right-open range (e.g. '(1,4)' --> [1, 2, 3])
        - list: A list of integers will be used as is
    tile_indexes: 1-based sequence of tile indexes to process; has same semantics as `region_indexes`
    config_dir: Directory containing experiment configuration files; defaults to `data_dir` if not given
    n_workers: Number of tiles to process in parallel; should generally match number of gpus and if
        the `gpus` argument is given, then the length of that list will be used as a default (otherwise
        default is 1)
    gpus: 0-based list of gpu indexes to use for processing; has same semantics as other integer
        list arguments like `region_indexes` and `tile_indexes` (i.e. can be a scalar, list, or 2-tuple)
    memory_limit: Maximum amount of memory to allow per-worker; defaults to 32G
    tile_prefetch_capacity: Number of input tiles to buffer into memory for processing; default is 2
        which is nearly always good as this means one tile will undergo processing while a second
        is buffered into memory asynchronously
    run_best_focus: Flag indicating that best focal plan selection operations should be executed
    n_iter_decon: Number of deconvolution iterations
    codex_py_log_level: Logging level for CODEX and dependent modules (except TensorFlow); can be
        specified as string or integer compatible with python logging levels (e.g. 'info', 'debug',
        'warn', 'error', 'fatal' or corresponding integers)
    tf_py_log_level: TensorFlow python logging level; same semantics as `codex_py_log_level`
    tf_cpp_log_level: TensorFlow C++ logging level; same semantics as `codex_py_log_level`

Usage:       codex-processor localhost DATA_DIR OUTPUT_DIR [REGION_INDEXES] [TILE_INDEXES] [CONFIG_DIR] [N_WORKERS] [GPUS] [MEMORY_LIMIT] [TILE_PREFETCH_CAPACITY] [RUN_BEST_FOCUS] [N_ITER_DECON] [CODEX_PY_LOG_LEVEL] [TF_PY_LOG_LEVEL] [TF_CPP_LOG_LEVEL]
             codex-processor localhost --data-dir DATA_DIR --output-dir OUTPUT_DIR [--region-indexes REGION_INDEXES] [--tile-indexes TILE_INDEXES] [--config-dir CONFIG_DIR] [--n-workers N_WORKERS] [--gpus GPUS] [--memory-limit MEMORY_LIMIT] [--tile-prefetch-capacity TILE_PREFETCH_CAPACITY] [--run-best-focus RUN_BEST_FOCUS] [--n-iter-decon N_ITER_DECON] [--codex-py-log-level CODEX_PY_LOG_LEVEL] [--tf-py-log-level TF_PY_LOG_LEVEL] [--tf-cpp-log-level TF_CPP_LOG_LEVEL]
```

To test the command line application, a small simulated experiment dataset is installed within the 
container that can be run using a command like this:

```bash
root@containerid> codex-processor localhost \
    --data-dir=/data/codex/simulations/sim-exp-01 \
    --output-dir=/data/codex/simulations/sim-exp-01-out
    
2018-04-24 18:05:40,139:INFO:76:codex.exec.pipeline: Starting pipeline for 1 tasks
2018-04-24 18:05:40,183:INFO:88:codex.exec.pipeline: Loaded tile 1 for region 1 [shape = (2, 5, 2, 297, 366)]
...
```


### TODO

- Documentation/scripts for non-Docker installation
- Investigate cloud orchestration tools:
    - https://github.com/llevar/butler
    - https://github.com/ray-project/ray
    - https://cloudstack.apache.org/about.html
    - https://github.com/StanfordBioinformatics/loom
    - https://toil.readthedocs.io/en/releases-3.5.x/index.html
- Best Focus:
    - Create validation framework for this
    - Default to most "central" z-plane in the event of ties