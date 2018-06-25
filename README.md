[![Build Status](https://travis-ci.org/hammerlab/codex.svg?branch=master)](https://travis-ci.org/hammerlab/codex)
[![Coverage Status](https://coveralls.io/repos/github/hammerlab/codex/badge.svg?branch=master)](https://coveralls.io/github/hammerlab/codex?branch=master)

## Cytokit

Cytokit is a processing and analysis toolkit for analyzing high-dimensional microscopy images.  The 
majority of the operations provided within Cytokit are intended to run as trained
deep learning models or other computational graphs on top of TensorFlow so as to exploit
GPU acceleration for processing terabyte-sized experiments.  

While intended for large datasets, 
single images or otherwise low-dimensional samples are also supported where any of the following 
common operations could be conducted to capitalize on the freely-available, GPU-based implementations
provided here:

- **Image Registration** - Repeated imaging of cells under different conditions or in different 
time periods can result in sample drift, which can be inverted using a TensorFlow 
implementation of [Phase Correlation](https://en.wikipedia.org/wiki/Phase_correlation) for 
image registration
- **Deconvolution** - Flourescent image blur can be compensated for using a GPU-based implementation
of the [Richardson-Lucy Deconvolution](https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution) algorithm (provided via [Flowdec](https://github.com/hammerlab/flowdec)).
- **Focal Quality Assessment** - Identifying peaks in focal quality within image volumes is 
provided using an image classifier designed by [GoogleAi](https://ai.google/) 
(see [Using Deep Learning to Facilitate Scientific Image Analysis](https://ai.googleblog.com/2018/03/using-deep-learning-to-facilitate.html)) 
- **Cell Segmentation** - Cytokit includes an application of a Keras-based
U-Net model for nuclei and cell segmentation, as well as attribution of other signals
to those segmented volumes (e.g. quantifying CD4 signals in t-cells identified
by nuclear stains).
- **Cytometric Analysis** - FCS and CSV exports can also be produced by Cytokit to facilitate
analysis in other tools like [FlowJo](https://www.flowjo.com/), [Cytobank](https://www.cytobank.org/), [FCS Express](https://www.denovosoftware.com/site/Flow-RUO-Overview.shtml), etc.    

Cytokit was built initially for processing Keyence images, specifically resulting from the [CODEX](https://www.akoyabio.com/technology/) protocol,
but would support any imaging process that produces tiled images with a specific naming convention.

#### CODEX Backport

As a small piece of standalone functionality, instructions can be found here for how to
run deconvolution for free on CODEX samples: [Standalone Deconvolution Instructions](python/standalone/deconvolution)


### Installation

TBD

### Examples 

TBD
