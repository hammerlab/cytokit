import numpy as np
import tempfile
import pathlib as pl
from skimage import util
from skimage import draw
from skimage import filters
from skimage import measure
from skimage import exposure
from skimage import transform
from skimage import feature
from skimage import morphology
from skimage import segmentation
from skimage import img_as_float
from skimage.feature.blob import _prune_blobs
from centrosome import propagate
from cytokit import math as ck_math
from cytokit.cytometry import cytometer
from skimage import io as sk_io
from cytokit.ops.cytometry import CHANNEL_COORDINATES
from scipy import ndimage as ndi
import logging
logger = logging.getLogger(__name__)


class CPExec(object):
    
    def run(*args):
        pass

class CPCytometer(cytometer.Cytometer2D):

    def initialize(self):
        super().initialize()
        self.tmpdir = pl.Path(tempfile.mkdtemp())
        logger.info('Initialized CP cytometer with tempdir "%s"', self.tmpdir)
        pass

    def quantify(self, tile, segments, z_plane=None, tile_indices=None, **kwargs):
        # tile -> (18, 15, 3, 1008, 1344) [cycle, z, channel, y, x]
        # segments -> (15, 4, 1008, 1344) [z, channel, y, x]
        
        # Select (cycle, channel, y, x) 
        tile = tile[:, z_plane]
        segments = segments[z_plane]
        
        tmpdir = self.tmpdir / ('tile_y{:02d}_x{:02d}'.format(tile_indices.tile_y, tile_indices.tile_x))
        tmpdir.mkdir(exist_ok=True)
        
        # Save object images
        for channel in ['nucleus_mask', 'cell_mask']:
            image = segments[CHANNEL_COORDINATES[channel][1]]
            path = tmpdir / ('object_' + channel + '.tif')
            sk_io.imsave(str(path), image)
            
        # Save expression images
        for channel in self.config.channel_names:
            image = tile[self.config.get_channel_coordinates(channel)]
            path = tmpdir / ('expression_' + channel + '.tif')
            sk_io.imsave(str(path), image)
        
        raise ValueError('in', tile.shape, segments.shape, z_plane, tmpdir)
#         assert tile.ndim == 5
#         # Run max-z projection across all channels and insert new axis where z dimension was
#         tile = tile.max(axis=1)[:, np.newaxis]
#         assert tile.ndim == 5, 'Expecting result after max-z projection to be 5D but got shape {}'.format(tile.shape)
#         assert tile.shape[0] == tile.shape[1] == 1
#         return cytometer.CytometerBase.quantify(tile, segments, **kwargs)

    def augment(self, df):
        df = cytometer.CytometerBase.augment(df, self.config.microscope_params)
        # Attempt to sum live + dead intensities if both channels are present
        for agg_fun in ['mean', 'sum']:
            cols = df.filter(regex='ci:(LIVE|DEAD):{}'.format(agg_fun)).columns.tolist()
            if len(cols) == 2:
                df['ci:LIVE+DEAD:{}'.format(agg_fun)] = df[cols[0]] + df[cols[0]]
        return df
    
    def shutdown(self):
        logger.info('In CP cytometer shutdown')
        # cleanup 
