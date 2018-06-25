from codex.ops.op import CodexOp
import os
import os.path as osp
import codex
from codex import io as codex_io
import logging
logger = logging.getLogger(__name__)


class AnalysisOp(CodexOp):

    def get_analysis_op_config(self):
        params = self.config.analysis_params
        opname = self.get_op_name()
        return params.get(opname, {})


class NotebookAnalysisOp(AnalysisOp):

    def _get_nb_path(self, nb_name):
        return osp.join(codex.nb_dir, 'analysis', nb_name)

    def _run_nb(self, nb_name, nb_output_path, nb_params):
        import papermill as pm
        op_name = self.get_op_name()
        logger.info('Running {} operation'.format(op_name))
        nb_input_path = self._get_nb_path(nb_name)
        pm.execute_notebook(nb_input_path, nb_output_path, parameters=nb_params)
        logger.info('{} operation complete; view results with `jupyter notebook {}`'.format(op_name, nb_output_path))


def get_best_focus_data(output_dir):
    """Get precomputed best focus plane information

    Note that this will return a data frame with references to 0-based region/tile indexes
    """
    from codex.ops import best_focus
    from codex import cli

    # Extract best focal plane selections from precomputed processor data
    best_focus_op = CodexOp.get_op_for_class(best_focus.CodexFocalPlaneSelector)
    processor_data_filepath = osp.join(output_dir, codex_io.get_processor_data_path())
    focus_data = cli.read_processor_data(processor_data_filepath)
    if best_focus_op not in focus_data:
        raise ValueError(
            'No focal plane statistics found in statistics file "{}".  '
            'Are you sure the processor.py app was run with `run_best_focus`=True?'
            .format(processor_data_filepath)
        )
    return focus_data[best_focus_op][['region', 'tile_index', 'tile_x', 'tile_y', 'best_z']].dropna().drop_duplicates()


class CytometryStatisticsAggregation(AnalysisOp):

    MODES = ['best_z_plane', 'most_cells', 'all']

    def _run(self, output_dir, **kwargs):
        import pandas as pd
        from codex.cytometry import data as cytometry_data

        # Collect arguments for this operation
        op_config = self.get_analysis_op_config()
        mode = op_config.get('mode', 'all')
        export_csv = op_config.get('export_csv', True)
        export_fcs = op_config.get('export_fcs', True)
        variant = op_config.get('variant', None)

        if mode not in CytometryStatisticsAggregation.MODES:
            raise ValueError(
                'Cytometry stats aggregation mode must be one of {} not "{}"'
                .format(CytometryStatisticsAggregation.MODES, mode)
            )

        # Aggregate all cytometry csv data (across tiles)
        cyto_data = cytometry_data.aggregate(self.config, output_dir)

        # If configured, select only data associated with "best" z planes
        if mode == 'best_z_plane':
            # Extract best focal plane selections from precomputed processor data
            focus_data = get_best_focus_data(output_dir)

            # Merge to cytometry data on region / tile index (this will add a single column, "best_z")
            merge_data = pd.merge(
                cyto_data, focus_data[['region_index', 'tile_index', 'best_z']],
                on=['region_index', 'tile_index'],
                how='left'
            )
            if merge_data['best_z'].isnull().any():
                # Create list of regions / tiles with null z planes
                ex = merge_data[merge_data['best_z'].isnull()][['region_index', 'tile_x', 'tile_y']]
                raise ValueError(
                    'Failed to find best z plane settings for at least one tile;\n'
                    'The following region/tile combinations have no z-planes: {}'
                    .format(ex.values)
                )
            # Filter result to where z plane equals best z
            res = merge_data[merge_data['best_z'] == merge_data['z']]

        # If configured, select only data for z planes associated with the most
        # cells (only makes sense w/ 2D segmentation)
        elif mode == 'most_cells':
            # Count number of cells per tile / z
            cts = cyto_data.groupby(['region_index', 'tile_index', 'z']).size().rename('count').reset_index()

            # Determine z plane with highest cell count
            cts = cts.groupby(['region_index', 'tile_index'])\
                .apply(lambda g: g.sort_values('count').iloc[-1]['z']).rename('z').reset_index()

            # Restrict data to only the z planes with the most cells
            res = pd.merge(
                cyto_data, cts,
                on=['region_index', 'tile_index', 'z'],
                how='inner'
            )
            assert len(res) == len(cts), \
                'Before/after merge sizes not equal (before = {}, after = {})'.format(len(cts), len(res))

        # Otherwise, do nothing
        else:
            res = cyto_data

        # Get file extension, possibly with user-defined "variant" name to be included in all
        # resulting file names
        def ext(file_ext):
            return file_ext if variant is None else '{}.{}'.format(variant, file_ext)

        # Export result as csv
        csv_path, fcs_path = None, None
        if export_csv:
            csv_path = osp.join(output_dir, codex_io.get_cytometry_agg_file_path(ext('csv')))
            res.to_csv(csv_path, index=False)
            logger.info('Saved cytometry aggregation results to csv at "{}"'.format(csv_path))
        if export_fcs:
            import re
            import fcswrite
            nonalnum = '[^0-9a-zA-Z]+'

            # For FCS exports, save only integer and floating point values and replace any non-alphanumeric
            # column name characters with underscores
            res_fcs = res.select_dtypes(['int', 'float']).rename(columns=lambda c: re.sub(nonalnum, '_', c))
            fcs_path = osp.join(output_dir, codex_io.get_cytometry_agg_file_path(ext('fcs')))
            fcswrite.write_fcs(filename=fcs_path, chn_names=res_fcs.columns.tolist(), data=res_fcs.values)
            logger.info('Saved cytometry aggregation results to fcs at "{}"'.format(fcs_path))
        return csv_path, fcs_path


class ProcessorDataSummary(NotebookAnalysisOp):

    def _run(self, output_dir, **kwargs):
        processor_data_path = osp.join(output_dir, codex_io.get_processor_data_path())
        nb_name = 'processor_data_analysis.ipynb'
        nb_output_path = osp.join(output_dir, osp.dirname(processor_data_path), 'processor_data_analysis.ipynb')
        nb_params = {'processor_data_path': processor_data_path}
        self._run_nb(nb_name, nb_output_path, nb_params)


class BestFocusMontageGenerator(AnalysisOp):

    def _run(self, output_dir, **kwargs):
        from codex.exec import montage

        # Set list of regions to process as 0-based indexes
        region_indexes = self.config.region_indexes

        # Extract best focal plane selections from precomputed processor data
        focus_data = get_best_focus_data(output_dir).set_index('region')

        # Loop through regions and generate a montage for each, skipping any (with a warning) that
        # do not have focal plane selection information
        for ireg in self.config.region_indexes:
            if ireg not in focus_data.index:
                logger.warning(
                    'Skipping region {} as it does not contain best focus information '
                    'resulting from the processing step'
                    .format(ireg + 1)
                )
                continue

            region_data = focus_data.loc[[ireg]].set_index(['tile_x', 'tile_y'], append=True)['best_z']
            logging.info('Generating montage for region %d of %d', ireg + 1, len(region_indexes))
            tiles = []
            for itile in range(self.config.n_tiles_per_region):
                tx, ty = self.config.get_tile_coordinates(itile)
                best_z = region_data.loc[(ireg, tx, ty)]
                path = codex_io.get_best_focus_img_path(ireg, tx, ty, best_z)
                tile = codex_io.read_image(osp.join(output_dir, path))
                tiles.append(tile)
            reg_img_montage = montage.montage(tiles, self.config)
            path = osp.join(output_dir, codex_io.get_best_focus_montage_path(ireg))
            logging.info('Saving montage to file "%s"', path)
            codex_io.save_image(path, reg_img_montage)
        logging.info('Montage generation complete')


OP_CLASSES = [
    CytometryStatisticsAggregation,
    ProcessorDataSummary,
    BestFocusMontageGenerator
]

# Map snake case (lcase + underscores) names to class object
OP_CLASSES_MAP = {CodexOp.get_op_for_class(c): c for c in OP_CLASSES}

