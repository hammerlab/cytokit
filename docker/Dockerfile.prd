FROM tensorflow/tensorflow:1.8.0-gpu-py3
ARG CYTOKIT_REPO_URL="https://github.com/hammerlab/cytokit.git"
ARG SIM_DIR=/lab/sim
ARG REPO_DIR=/lab/repos
ARG DATA_DIR=/lab/data
ARG CYTOKIT_REPO_DIR=$REPO_DIR/cytokit

RUN mkdir -p $LAB_DIR $REPO_DIR $DATA_DIR $SIM_DIR

RUN apt-get update && apt-get install -y --no-install-recommends git vim wget
RUN pip install --upgrade pip

# OpenCV package dependencies and tk matplotlib backend
RUN apt-get install -y libsm6 libxext6 libfontconfig1 libxrender1 python3-tk

RUN pip --no-cache-dir install \
    PyYAML==3.13 \
    numpy==1.14.2 \
    scipy==1.0.1 \
    pandas==0.22.0 \
    scikit-image==0.14.0 \
    scikit-learn==0.20.1 \
    opencv-python==3.4.3.18 \
    requests==2.20.1 \
    matplotlib==2.2.2 \
    dask[distributed]==1.0.0 \
    bokeh==1.0.1 \
    keras==2.2.4 \
    centrosome==1.1.5 \
    mahotas==1.4.5 \
    plotnine==0.4.0 \
    jupyterlab \
    python-dotenv \
    papermill \
    fcswrite \
    tifffile \
    fire \
    seaborn

# Install Dash and per their instructions, freezing specific versions
# See: https://dash.plot.ly/getting-started
RUN pip --no-cache-dir install dash==0.21.1  \
    dash-renderer==0.13.0 \
    dash-html-components==0.11.0 \
    dash-core-components==0.23.0 \
    plotly

# Install Flowdec for deconvolution
RUN pip --no-cache-dir install flowdec

# Download simulation data for testing
RUN cd $SIM_DIR && \
    wget -nv https://storage.googleapis.com/musc-codex/datasets/simulations/sim-exp-01.zip && \
    unzip -q sim-exp-01.zip

# Clone cytokit repo
RUN cd $REPO_DIR && git clone $CYTOKIT_REPO_URL

# Add any source directories for development to python search path
RUN mkdir -p $(python -m site --user-site) && \
    echo "$CYTOKIT_REPO_DIR/python/pipeline" > $(python -m site --user-site)/local.pth && \
    echo "$CYTOKIT_REPO_DIR/python/notebooks/src" >> $(python -m site --user-site)/local.pth && \
    echo "$CYTOKIT_REPO_DIR/python/applications" >> $(python -m site --user-site)/local.pth

WORKDIR "/lab"

ENV CYTOKIT_SIM_DIR $SIM_DIR
ENV CYTOKIT_DATA_DIR $DATA_DIR
ENV CYTOKIT_REPO_DIR $CYTOKIT_REPO_DIR
ENV SHELL /bin/bash

# Set these or papermill will error out with:
# "Click will abort further execution because Python 3 was configured to use ASCII as encoding for the environment"
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Eliminate these warnings globally: FutureWarning: Conversion of the second argument of issubdtype from
# `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`
# See here for discussion: https://github.com/h5py/h5py/issues/961
ENV PYTHONWARNINGS "ignore::FutureWarning:h5py"

# Create cli links at runtime instead of container buildtime due to source scripts being
# in repos mounted at runtime
CMD chmod a+x $CYTOKIT_REPO_DIR/python/pipeline/cytokit/cli/main.py && \
    ln -s $CYTOKIT_REPO_DIR/python/pipeline/cytokit/cli/main.py /usr/local/bin/cytokit && \
    jupyter lab --allow-root
