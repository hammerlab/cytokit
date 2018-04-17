# Akoya Scripts

This directory contains a script (for now just one) that is useful for running deconvolution on image stacks after processing and before segmentation in the CODEX pipeline.

### Setup

In order to be able to use this script, you'll need to do the following:

1. Install [Anaconda](https://www.anaconda.com/download/)
2. Follow the Tensorflow instructions for GPU configuration:
    - [Linux](https://www.tensorflow.org/install/install_linux): See "NVIDIA requirements to run TensorFlow with GPU support"
    - [Windows](https://www.tensorflow.org/install/install_windows): See "Requirements to run TensorFlow with GPU support"
    - Mac - Not supported for GPU acceleration by Tensorflow
    - **Note**: The only things absolutely necessary on either platform are CUDA Tookit and cuDNN (I've never had to install specific drivers or anything beyond that)

3. Create a fresh environment

```
conda create -n codex python=3.6
source activate codex
```

4. Clone and install the [Flowdec](https://github.com/hammerlab/flowdec) project

```
git clone https://github.com/hammerlab/flowdec
cd flowdec/python
pip install .
```

5. Test that Tensorflow on GPU is enabled

```
# Launch python shell with environment active
(codex)> python
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess.run(hello))
``` 

If you receive no errors and you see a line in the output like "Adding visible gpu devices: 0" then everything is good to go.


6. Clone this project (codex-analysis) and add necessary files to PYTHONPATH or .pth file for Anaconda like below

```
git clone https://github.com/hammerlab/codex-analysis
echo "$REPOS/codex-analysis/deconvolution/flowdec/python" >> ~/anaconda3/envs/codex/lib/python3.6/site-packages/local.pth
```

### Execution

The Akoya CLI should be run after running CODEX processor with deconvolution disabled and will copy any necessary metadata files as well as write deconvolution results to a new directory.  Assuming the CODEX processor input dir was ```data/codex-acquisition-raw``` and the output dir was ```data/codex-processor-output```, then the following example could be used to perform deconvolution:

```
(codex)> cd $REPOS/codex-analysis/deconvolution/flowdec/python/akoya
(codex)> python akoya_deconvolution.py \
    --raw-dir=data/codex-acquisition-raw \
    --input-dir=data/codex-processor-output \
    --output-dir=data/codex-deconvolution-output
```

See usage below for other settings/parameters but they should rarely be necessary other than perhaps ```--n-iter``` or ```--dry-run```.

### Usage


The [akoya_deconvolution.py](akoya_deconvolution.py) script has the following usage:

```
(codex)> python akoya_deconvolution.py --help

usage: akoya_deconvolution.py [-h] --raw-dir RAW --input-dir INPUT
                              --output-dir OUTPUT [--psf-dir PSFDIR]
                              [--psf-pattern PSFPATTERN] [--pad-dims PADDIMS]
                              [--pad-mode PADMODE]
                              [--scale-factor SCALE_FACTOR]
                              [--scale-mode {stack,slice}]
                              [--observer-dir OBSERVER_DIR]
                              [--observer-coords OBSERVER_COORDS]
                              [--n-iter N_ITER] [--dry-run]

optional arguments:
  -h, --help            show this help message and exit
  --raw-dir RAW         Path to original data directory containing
                        acquisitions
  --input-dir INPUT     Path to directory containing images stacks from CODEX
                        processor
  --output-dir OUTPUT   Path to directory to contain results
  --psf-dir PSFDIR      Optional path to directory containing psf stacks; if
                        not given PSFs will be generated based on experiment
                        configuration (which is almost always what you want)
  --psf-pattern PSFPATTERN
                        Optional PSF file naming pattern; e.g. 'psf-
                        ch{channel_id}.tif' where channel_id is 1-based index
                        (must be given if --psf-dir is also provided)
  --pad-dims PADDIMS    Amount by which to pad a single z-stack as a 'x,y,z'
                        string; e.g. '0,0,6' for no x or y padding and at
                        least 6 units of padding in z-direction (6 units in
                        z-direction would correspond to 3 slices on top and 3
                        on bottom)
  --pad-mode PADMODE    Either 'log2' or 'none' indicating whether or not to
                        stretch dimension lengths out to those optimal for FFT
  --scale-factor SCALE_FACTOR
                        Each z-stack will be multiplied by this number after
                        matching its mean intensity with that of the original
                        image. One reason to do this is to minimize
                        saturation, as seems to be the original intention in
                        the Akoya codebase (which uses the value 1/2, which is
                        the default value for the parameter).
  --scale-mode {stack,slice}
                        One of 'stack' or 'slice' indicating whether or not
                        scaling should be applied to whole z-stacks or to
                        individual slices
  --observer-dir OBSERVER_DIR
                        Directory in which to save per-iteration images
                        (useful for determining proper iteration counts
  --observer-coords OBSERVER_COORDS
                        Coordinates of single 2D images to save per-iteration
                        views on, a feature helpful for choosing the number of
                        iterations to use; should be specified in
                        '<tile>,<cycle>,<channel>,<z>' format where each is a
                        one-based index
  --n-iter N_ITER       Number of Richardson-Lucy iterations to execute
                        (defaults to 25)
  --dry-run             Flag indicating to only show inputs and proposed
                        outputs
```