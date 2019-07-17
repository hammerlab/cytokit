"""Core functionality for building and executing CellProfiler pipelines

**Important:** This is intended for use in a CellProfiler-compatible python 2.7 environment
"""
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

# The cellprofiler module maintains state related to javabridge so
# in order to refer to it globally while managing that state locally,
# this variable will be used as the module reference
cp = None

DEFAULT_CONFIG_FILE = 'experiment.yaml'
OBJECT_NAME_CELL = 'Cell'
OBJECT_NAME_NUCLEUS = 'Nucleus'
DEFAULT_CYTOMETRY_CHANNELS = {0: OBJECT_NAME_CELL, 1: OBJECT_NAME_NUCLEUS}
DEFAULT_OBJECT_NAMES = list(DEFAULT_CYTOMETRY_CHANNELS.values())


def _load_config(path):
    if not osp.isfile(path):
        path = osp.join(path, DEFAULT_CONFIG_FILE)
    logger.info('Loading experiment configuration from file "%s"', path)
    with open(path, 'r') as f:
        return yaml.load(f)


class QuantificationPipeline(object):

    def __init__(self, output_dir, channel_names,
                 object_names=DEFAULT_OBJECT_NAMES,
                 export_csv=True, export_db=True, export_db_objects_separately=False):
        """Quantification pipeline model

        This will construction a CP pipeline instance based on a provided number of channels (and a few other options)

        Args:
            output_dir: Directory in which results should be stored
            channel_names: List of channel names expected to appear in image files
            object_names: Names of cell and nuclei objects (defaults to "Cell" and "Nucleus")
            export_csv: Export measurements as spreadsheet
            export_db: Export measurements to SQLite DB (for CellProfiler Analyst)
            export_db_objects_separately: Determines whether or not a single object table is created or if
                multiple tables are created for each object type (i.e. Cells and Nuclei); False is generally
                better for CPA analysis as more data is available but with large multiplexed experiments,
                this means that only half as many channels can be supported before hitting CP "max cols exceeded"
                errors -- essentially this should be left false until that error is hit in which case it can
                be set to true with the knowledge that CPA will only have the context of one object at a time
        """
        self.output_dir = output_dir
        self.channel_names = channel_names
        self.object_names = object_names
        self.export_csv = export_csv
        self.export_db = export_db
        self.export_db_objects_separately = export_db_objects_separately
        self.module_ct = 0
        self.pipeline = None

    def reset(self):
        self.module_ct = 0
        self.pipeline = cp.pipeline.Pipeline()

    def get_images(self):
        m = cp.modules.images.Images()
        m.filter_choice.value = cp.modules.images.FILTER_CHOICE_IMAGES
        return m

    def get_metadata(self):
        m = cp.modules.metadata.Metadata()
        m.wants_metadata.value = True
        regex = u'^R(?P<Region>\\d+?)_X(?P<TileX>\\d+?)_Y(?P<TileY>\\d+?)_(?P<ImgType>\\w+?)_.*\\.tif'
        m.extraction_methods[0].file_regexp.value = regex
        # Set extraction method to "Extract from file/folder names"
        m.extraction_methods[0].extraction_method.value = cp.modules.metadata.X_MANUAL_EXTRACTION
        # Set source for choice above to "File name"
        m.extraction_methods[0].source.value = cp.modules.metadata.XM_FILE_NAME
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

    def get_measureobjectintensity(self):
        m = cp.modules.measureobjectintensity.MeasureObjectIntensity()
        for i, v in enumerate(self.object_names):
            if i != 0:
                m.add_object()
            m.objects[i].name.value = v
        for i, v in enumerate(self.channel_names):
            if i != 0:
                m.add_image()
            m.images[i].name.value = v
        return m

    def get_measureobjectneighbors(self):
        m = cp.modules.measureobjectneighbors.MeasureObjectNeighbors()
        m.object_name.value = OBJECT_NAME_CELL
        m.neighbors_name.value = OBJECT_NAME_CELL
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

        # Export to SQLite
        m.db_type.value = cp.modules.exporttodatabase.DB_SQLITE
        m.experiment_name.value = 'Exp'
        m.table_prefix.value = 'Exp_'
        m.sqlite_file.value = 'CPA.db'
        m.location_object.value = OBJECT_NAME_CELL

        # Set output options (custom directory in this case)
        m.directory.value = cp.preferences.ABSOLUTE_FOLDER_NAME
        m.directory.custom_path = self.output_dir
        m.save_cpa_properties.value = True

        # Set "Overwrite without warning" to "Data and schema"
        m.allow_overwrite.value = cp.modules.exporttodatabase.OVERWRITE_ALL

        # Use "One table per object type" or "Single object table" and disable image level aggregations
        # since multiplexed experiments have an awkwardly large number of channels, which often leads
        # to "Too many columns" errors during the execution of this module
        if self.export_db_objects_separately:
            m.separate_object_tables.value = cp.modules.exporttodatabase.OT_PER_OBJECT
        else:
            m.separate_object_tables.value = cp.modules.exporttodatabase.OT_COMBINE
        m.wants_agg_mean.value = False
        m.wants_agg_median.value = False
        return m

    def add(self, m):
        self.module_ct += 1
        m.module_num = self.module_ct
        self.pipeline.add_module(m)

    def create(self):
        self.reset()
        self.add(self.get_images())
        self.add(self.get_metadata())
        self.add(self.get_namesandtypes())
        self.add(self.get_measureobjectsizeshape())
        self.add(self.get_measureobjectintensity())
        self.add(self.get_measureobjectneighbors())
        if self.export_csv:
            self.add(self.get_exporttospreadsheet())
        if self.export_db:
            self.add(self.get_exporttodatabase())
        return self.pipeline


def _get_filter(r):
    def fn(img):
        return img[:, (r['best_z'],)]
    return fn


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
        (r['region_index'], r['tile_x'], r['tile_y']): _get_filter(r)
        for r in pd['focal_plane_selector']
    }


def get_coordinates(filename):
    # Assume coordinates present in file naming convention
    coords = re.findall('R(\d+)_X(\d+)_Y(\d+).tif', filename)
    if not coords:
        raise ValueError('Failed to extract coordinates from filename "' + filename + '"')
    # Return as 0-based (region_index, tile_x, tile_y)
    return tuple([int(c) - 1 for c in coords[0]])


def read_tile(file):
    """Read 5D tif that may have been saved with squeezed dimensions"""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore', category=UserWarning,
            message='unpack: string size must be a multiple of element size'
        )
        warnings.filterwarnings(
            'ignore', category=RuntimeWarning,
            message='py_decodelzw encountered unexpected end of stream'
        )
        with tifffile.TiffFile(file) as tif:
            tags = dict(tif.imagej_metadata)
            if 'axes' not in tags:
                warnings.warn('ImageJ tags do not contain "axes" property (file = {}, tags = {})'.format(file, tags))
            else:
                if tags['axes'] != 'TZCYX':
                    warnings.warn(
                        'Image has tags indicating that it was not saved in TZCYX format.  '
                        'The file should have been saved with this property explicitly set and further '
                        'processing of it may be unsafe (file = {})'.format(file)
                    )
            slices = [
                slice(None) if 'frames' in tags else None,
                slice(None) if 'slices' in tags else None,
                slice(None) if 'channels' in tags else None,
                slice(None),
                slice(None)
            ]
            res = tif.asarray()[tuple(slices)]

            if res.ndim != 5:
                raise ValueError(
                    'Expected 5 dimensions in image at "{}" but found {} (shape = {})'
                    .format(file, res.ndim, res.shape)
                )
            return res


def extract(filters, image_dir, channels=None):
    """Generate individual CP-compatible images from multidimensional tif files"""
    files = glob.glob(osp.join(image_dir, '*.tif'))
    if not files:
        raise ValueError('No images found in directory "{}"'.format(image_dir))
    for f in files:
        # Read image as (cycles, z, channels, y, x) and extract region/tile coordinates
        img = read_tile(f)
        if img.ndim != 5:
            raise ValueError('Expecting 5D tile image, got shape {} (path = {})'.format(img.shape, f))
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
    filters = load_image_filters(output_dir)

    logger.info('Extracting expression channel images')
    processor_image_dir = osp.join(output_dir, 'processor', 'tile')
    for channel_images in extract(filters, processor_image_dir, expression_channels):
        for file, channel, i, coords, img in channel_images:
            filename = _filename(coords, 'EXP', channel)
            path = osp.join(extraction_dir, filename)
            logger.debug('Extracting expression image to: %s', path)
            tifffile.imsave(path, img)

    logger.info('Extracting object images')
    cytometry_image_dir = osp.join(output_dir, 'cytometry', 'tile')
    for channel_images in extract(filters, cytometry_image_dir):
        for file, channel, i, coords, img in channel_images:
            if i not in cytometry_channels:
                continue
            filename = _filename(coords, 'OBJ', cytometry_channels[i])
            path = osp.join(extraction_dir, filename)
            logger.debug('Extracting object image to: %s', path)
            tifffile.imsave(path, img)


def create_dirs(dirs):
    for d in dirs:
        if not osp.exists(d):
            os.makedirs(d)


def run_quantification(config_path, output_dir,
                       export_db=True, export_db_objects_separately=False,
                       export_csv=True, do_extraction=True):
    config = _load_config(config_path)
    channels = config['acquisition']['channel_names']
    # num_regions = len(config['acquisition']['region_names'])
    # region_height = config['acquisition']['region_height']
    # region_width = config['acquisition']['region_width']
    # num_cycles = config['acquisition']['num_cycles']

    cp_dir = osp.join(output_dir, 'cytometry', 'cellprofiler')
    cp_input_dir = osp.join(cp_dir, 'images')
    cp_output_dir = osp.join(cp_dir, 'results')
    create_dirs([cp_dir, cp_input_dir, cp_output_dir])

    # Extract individual 2D images for each region+tile+channel combination (including cell/nuclei objects)
    # *This may have already been done so it can be skipped if unnecessary
    if do_extraction:
        run_extraction(output_dir, cp_input_dir, channels)

    # Define the pipeline modules
    pipeline = QuantificationPipeline(
        output_dir=cp_output_dir,
        channel_names=channels,
        export_csv=export_csv,
        export_db=export_db,
        export_db_objects_separately=export_db_objects_separately
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
    measurements_path = osp.join(cp_output_dir, 'Measurements.h5')
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
        "--config-path",
        dest="config_path",
        help="Path to cytokit experiment configuration file "
             "(if a directory, 'experiment.yaml' will be suffixed to the path)",
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
        "-d",
        "--export-db",
        dest="export_db",
        choices=['true', 'false'],
        help="Export measurements as SQLite DB",
        default='true',
    )
    parser.add_option(
        "-s",
        "--export-db-objects-separately",
        dest="export_db_objects_separately",
        choices=['true', 'false'],
        help="Create a separate DB table for each object type (Cell and Nucleus) or one single table with both "
             "(false is better here unless max col errors occur, in which case true is necessary)",
        default='false',
    )
    parser.add_option(
        "-t",
        "--export-csv",
        dest="export_csv",
        choices=['true', 'false'],
        help="Export measurements as csv files",
        default='true',
    )
    parser.add_option(
        "-e",
        "--do-extraction",
        dest="do_extraction",
        choices=['true', 'false'],
        help="Extract individual images for CP pipeline "
             "(this is necessary the first time, but not on repeat executions)",
        default='true',
    )
    parser.add_option(
        "-l",
        "--log-level",
        dest="log_level",
        help="Python logging level",
        default="INFO",
    )
    return parser.parse_args()


def start_cp():
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


def stop_cp():
    cp.utilities.cpjvm.cp_stop_vm()


def main():
    # Bury deprecation warnings (primarily related to numpy)
    warnings.filterwarnings('ignore', category=FutureWarning)
    # Hide division by zero warnings as they are expected for some statistics
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in true_divide')

    options = parse()[0]

    # Convert log level from string int to int (or use as string)
    try:
        log_level = int(options.log_level)
    except:
        log_level = options.log_level
    logging.basicConfig(level=log_level)

    if options.analysis.lower() not in ['quantification']:
        raise ValueError('Analysis type "{}" not valid'.format(options.analysis))

    start_cp()
    try:
        run_quantification(
            options.config_path, options.output_dir,
            export_csv=options.export_csv == 'true',
            export_db=options.export_db == 'true',
            do_extraction=options.do_extraction == 'true',
            export_db_objects_separately=options.export_db_objects_separately == 'true'
        )
        return 0
    finally:
        stop_cp()


if __name__ == "__main__":
    sys.exit(main())
