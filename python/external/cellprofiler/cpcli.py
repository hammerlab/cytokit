"""Core functionality for building and executing CellProfiler pipelines

**Important:** This is intended for use in a CellProfiler-compatible python 2.7 environment
"""
# This order is necessary
import tifffile
import numpy as np
import os.path as osp
import glob
import yaml
import json
import re
import sys
import os
import optparse
import traceback
import logging
import warnings
logger = logging.getLogger(__name__)

cp = None

# Bury numpy warnings
# np.seterr(all="ignore")


OBJECT_NAME_CELL = 'Cell'
OBJECT_NAME_NUCLEUS = 'Nucleus'
DEFAULT_CYTOMETRY_CHANNELS = {0: OBJECT_NAME_CELL, 1: OBJECT_NAME_NUCLEUS}
DEFAULT_OBJECT_NAMES = list(DEFAULT_CYTOMETRY_CHANNELS.values())


def _load_config(path):
    with open(path, 'r') as f:
        return yaml.load(f)


class QuantificationPipeline(object):

    def __init__(self, output_dir, channel_names,
                 object_names=DEFAULT_OBJECT_NAMES,
                 export_spreadsheet=True, export_database=True):
        self.output_dir = output_dir
        self.channel_names = channel_names
        self.object_names = object_names
        self.export_spreadsheet = export_spreadsheet
        self.export_database = export_database
        self.module_ct = 0

    def get_images(self):
        m = cp.modules.images.Images()
        m.filter_choice.value = cp.modules.images.FILTER_CHOICE_IMAGES
        return m

    def get_metadata(self):
        m = cp.modules.metadata.Metadata()
        return m

    def get_namesandtypes(self):
        # Resources:
        # - https://github.com/CellProfiler/CellProfiler/blob/master/tests/modules/test_namesandtypes.py#L960
        m = cp.modules.namesandtypes.NamesAndTypes()
        m.assignment_method.value = cp.modules.namesandtypes.ASSIGN_RULES
        # "Image set matching" -- determines how images from different channels are aligned
        # ORDER means that the file paths must sort alphabetically in a way that will align them
        m.matching_choice.value = cp.modules.namesandtypes.MATCH_BY_ORDER
        i = 0
        for v in self.object_names:
            if i != 0:
                m.add_assignment()
            m.assignments[i].image_name.value = v
            m.assignments[i].object_name.value = v
            m.assignments[i].rule_filter.value = u'file does contain "OBJ_' + v + '"'
            m.assignments[i].load_as_choice.value = cp.modules.namesandtypes.LOAD_AS_OBJECTS
            i += 1
        for v in self.channel_names:
            if i != 0:
                m.add_assignment()
            m.assignments[i].image_name.value = v
            m.assignments[i].object_name.value = v
            m.assignments[i].rule_filter.value = 'file does contain "EXP_' + v + '"'
            m.assignments[i].load_as_choice.value = cp.modules.namesandtypes.LOAD_AS_GRAYSCALE_IMAGE
            i += 1
        return m

    def get_measureobjectsizeshape(self):
        m = cp.modules.measureobjectsizeshape.MeasureObjectSizeShape()
        m.object_groups[0].name.value = self.object_names[0]
        m.add_object()
        m.object_groups[1].name.value = self.object_names[1]
        return m

    def get_exporttospreadsheet(self):
        m = cp.modules.exporttospreadsheet.ExportToSpreadsheet()
        # Corresponds to "Elsewhere..." option in UI
        m.directory.dir_choice = cp.preferences.ABSOLUTE_FOLDER_NAME
        m.directory.custom_path = self.output_dir
        m.prefix.set_is_yes(True)
        m.prefix.value = 'Table_'
        m.delimiter.value = cp.modules.exporttospreadsheet.DELIMITER_TAB
        m.add_metadata.value = True
        m.wants_overwrite_without_warning.value = True
        return m

    def get_exporttodatabase(self):
        m = cp.modules.exporttodatabase.ExportToDatabase()
        m.experiment_name.value = 'Exp'
        m.table_prefix.value = 'exp_'
        m.db_name.value = 'CPA'
        m.save_cpa_properties.value = True
        m.location_object.value = OBJECT_NAME_CELL
        m.directory.value = cp.preferences.ABSOLUTE_FOLDER_NAME
        m.directory.custom_path = self.output_dir
        m.db_type.value = cp.modules.exporttodatabase.DB_SQLITE
        m.allow_overwrite.value = cp.modules.exporttodatabase.OVERWRITE_ALL
        return m

    def add(self, m):
        self.module_ct += 1
        m.module_num = self.module_ct
        pipeline.add_module(m)

    def create(self):
        self.module_ct = 0
        pipeline = cp.pipeline.Pipeline()
        self.add(self.get_images())
        self.add(self.get_metadata())
        self.add(self.get_namesandtypes())
        self.add(self.get_measureobjectsizeshape())
        if self.export_spreadsheet:
            self.add(self.get_exporttospreadsheet())
        if self.export_database:
            self.add(self.get_exporttodatabase())
        return pipeline


def load_image_filters(output_dir):
    """Build per-tile dimension filters for extraction (currently based on focal plane selection)"""
    path = osp.join(output_dir, 'processor', 'data.json')
    with open(path, 'r') as f:
        pd = json.load(f)
    if 'focal_plane_selector' not in pd:
        raise ValueError(
            'Focal plane selection data not found in experiment output; ensure that experiment config contains '
            '`run_best_focus: True` as this is necessary to provide 2D image inputs to CellProfiler')
    return {
        # Return 5D tile image filter for each tile
        (r['region_index'], r['tile_x'], r['tile_y']): lambda img: img[:, (r['best_z'],)]
        for r in pd['focal_plane_selector']
    }


def get_coordinates(filename):
    coords = re.findall('R(\d+)_X(\d+)_Y(\d+).tif', filename)
    if not coords:
        raise ValueError('Failed to extract coordinates from filename "' + filename + '"')
    # Return as 0-based (region_index, tile_x, tile_y)
    return tuple([int(c) - 1 for c in coords[0]])


def extract(filters, image_dir, channels=None):
    """Generate individual CP-compatible images from multidimensional tif files"""
    #files = glob.glob(osp.join(image_dir, '*.tif'))
    files = glob.glob(osp.join(image_dir, 'R*01_X*01_Y*01.tif'))
    if not files:
        raise ValueError('No images found in directory "{}"'.format(image_dir))
    for f in files:

        # Read image as (cycles, z, channels, y, x) and extract region/tile coordinates
        img = tifffile.imread(f)
        if img.ndim != 5:
            raise ValueError('Expecting 5D tile image, got shape {}'.format(img.shape))
        coords = get_coordinates(osp.basename(f))

        # Extract 2D images from 5D tile
        img = filters[coords](img)
        if img.shape[1] != 1:
            raise AssertionError('Filters must return 2D image (got image with shape {})'.format(img.shape))

        # Reshape (cycles, z, channels, y, x) -> (channels, y, x)
        img = np.reshape(np.squeeze(img, axis=1), (-1,) + img.shape[-2:])

        # Set default channel names if none provided
        chs = channels
        if chs is None:
            chs = ['CH{}'.format(i) for i in range(img.shape[0])]

        if img.shape[0] != len(chs):
            raise ValueError(
                'Expression image at "{}" has shape {} with {} channels but {} are expected'
                .format(f, img.shape, img.shape[0], len(chs))
            )
        # Generate channel name, index, region/tile coords, and image
        yield [
            (f, channel, i, coords, img[i])
            for i, channel in enumerate(chs)
        ]


def _filename(coords, typ, name):
    return 'R{:03d}_X{:03d}_Y{:03d}_{}_{}.tif'.format(coords[0] + 1, coords[1] + 1, coords[2] + 1, typ, name)


def run_extraction(output_dir, extraction_dir, expression_channels, cytometry_channels=DEFAULT_CYTOMETRY_CHANNELS):
    if not osp.exists(extraction_dir):
        os.makedirs(extraction_dir)

    filters = load_image_filters(output_dir)
    processor_image_dir = osp.join(output_dir, 'processor', 'tile')
    for channel_images in extract(filters, processor_image_dir, expression_channels):
        for file, channel, i, coords, img in channel_images:
            filename = _filename(coords, 'EXP', channel)
            path = osp.join(extraction_dir, filename)
            logger.debug('Extracting expression image to: %s', path)
            tifffile.imsave(path, img)

    cytometry_image_dir = osp.join(output_dir, 'cytometry', 'tile')
    for channel_images in extract(filters, cytometry_image_dir):
        for file, channel, i, coords, img in channel_images:
            if i not in cytometry_channels:
                continue
            filename = _filename(coords, 'OBJ', cytometry_channels[i])
            path = osp.join(extraction_dir, filename)
            logger.debug('Extracting object image to: %s', path)
            tifffile.imsave(path, img)


def run_quantification(config_file, output_dir):
    config = _load_config(config_file)
    channels = config['acquisition']['channel_names']
    # num_regions = len(config['acquisition']['region_names'])
    # region_height = config['acquisition']['region_height']
    # region_width = config['acquisition']['region_width']
    # num_cycles = config['acquisition']['num_cycles']

    cp_dir = osp.join(output_dir, 'cytometry', 'cellprofiler')
    cp_input_dir = osp.join(cp_dir, 'images')
    cp_output_dir = osp.join(cp_dir, 'results')

    # Extract individual 2D images for each region+tile+channel combination (including cell/nuclei objects)
    run_extraction(output_dir, cp_input_dir, channels)

    # Define the pipeline modules
    pipeline = QuantificationPipeline(
        output_dir=cp_output_dir,
        channel_names=channels
    )
    # Instantiate as CP pipeline instance
    pipeline = pipeline.create()

    # Set input files for processing
    input_paths = glob.glob(osp.join(cp_input_dir, '*.tif'))
    if len(input_paths) == 0:
        raise AssertionError('Found no files to process in CP input directory {}'.format(cp_input_dir))
    pipeline.add_pathnames_to_file_list(input_paths)

    # Export for reference/debugging
    path = osp.join(cp_dir, 'pipeline.cppipe')
    logger.info('Saving pipeline to path "%s"', path)
    pipeline.savetxt(path)

    # Run the pipeline
    logger.info('Running CP pipeline')
    measurements_path = osp.join(cp_output_dir, 'measurements.h5')
    pipeline.run(measurements_filename=measurements_path)

    logger.info('CP pipeline run complete; results at: %s', cp_output_dir)


def parse():
    parser = optparse.OptionParser()

    parser.add_option(
        "-a",
        "--analysis",
        dest="analysis",
        type="choice",
        choices=['quantification'],
        help="Name of analysis to run (Currently only 'quantification' implemented)",
        default='quantification',
    )
    parser.add_option(
        "-c",
        "--config-file",
        dest="config_file",
        help="Path to cytokit experiment configuration file (yaml)",
        default=None,
    )
    parser.add_option(
        "-o",
        "--output-dir",
        dest="output_dir",
        help="Path to cytokit experiment output directory "
             "(e.g. /lab/data/20180101_codex_spleen/20180101_codex_mouse_spleen_balbc_slide1/output/v01)",
        default=None,
    )
    parser.add_option(
        "-l",
        "--log-level",
        dest="log_level",
        help="Python logging level",
        default="INFO",
    )
    return parser.parse_args()


def main():
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in true_divide')

    global cp
    import cellprofiler
    import cellprofiler.preferences
    import cellprofiler.utilities.cpjvm
    cellprofiler.preferences.set_awt_headless(True)
    cellprofiler.preferences.set_headless()
    cellprofiler.utilities.cpjvm.cp_start_vm()
    # Modules must be loaded after headless settings to avoid wx import errors
    import cellprofiler.modules
    cp = cellprofiler

    options = parse()[0]

    logging.basicConfig(level=options.log_level)

    if options.analysis.lower() not in ['quantification']:
        raise ValueError('Analysis type "{}" not valid'.format(options.analysis))

    try:
        run_quantification(options.config_file, options.output_dir)
        return 0
    except:
        traceback.print_exc(file=sys.stderr)
        return 1
    finally:
        cellprofiler.utilities.cpjvm.cp_stop_vm()


if __name__ == "__main__":
    sys.exit(main())


