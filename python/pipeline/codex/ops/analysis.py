from codex.ops.op import CodexOp
import logging
logger = logging.getLogger(__name__)


class AnalysisOp(CodexOp):

    def get_analysis_op_config(self):
        params = self.config.analysis_params
        opname = self.get_op_name()
        return params.get(opname, {})


class CytometryStatisticsAggregation(AnalysisOp):

    def _run(self, output_dir, **kwargs):
        import fcswrite
        op_config = self.get_analysis_op_config()

        for ti in tile_indices:
            pass


class ProcessorDataSummary(AnalysisOp):

    def _get_nb_path(self, nb_name):
        return osp.join(codex.nb_dir, 'analysis', nb_name)

    def _run(self, output_dir, **kwargs):
        logging.info('Running processor data analysis')
        processor_data_path = osp.join(output_dir, processor_data_filepath)
        nb_input_path = self._get_nb_path('processor_data_analysis.ipynb')
        nb_output_path = osp.join(output_dir, 'processor_data_analysis.ipynb')
        pm.execute_notebook(nb_input_path, nb_output_path, parameters={'processor_data_path': processor_data_path})
        logging.info('Processor data analysis complete; view with `jupyter notebook {}`'.format(nb_output_path))


class BestFocusMontageGenerator(AnalysisOp):

    def _run(self, output_dir, **kwargs):
        from codex.ops import op, best_focus
        from codex import config as codex_config
        from codex.ops import op, best_focus
        from codex.exec import montage
        from codex import io as codex_io

        config = codex_config.load(config_dir)
        if region_indexes is None:
            region_indexes = config.region_indexes

        best_focus_op = op.CodexOp.get_op_for_class(best_focus.CodexFocalPlaneSelector)
        processor_data_filepath = osp.join(output_dir, processor_data_filepath)
        focus_data = cli.read_processor_data(processor_data_filepath)
        if best_focus_op not in focus_data:
            raise ValueError(
                'No focal plane statistics found in statistics file "{}".  '
                'Are you sure the processor.py app was run with `run_best_focus`=True?'
                .format(processor_data_filepath)
            )
        focus_data = focus_data[best_focus_op].set_index(['region', 'tile_x', 'tile_y'])['best_z']

        for ireg in region_indexes:
            logging.info('Generating montage for region %d of %d', ireg + 1, len(region_indexes))
            tiles = []
            for itile in range(config.n_tiles_per_region):
                tx, ty = config.get_tile_coordinates(itile)
                best_z = focus_data.loc[(ireg, tx, ty)]
                path = codex_io.get_best_focus_img_path(ireg, tx, ty, best_z)
                tile = codex_io.read_image(osp.join(output_dir, path))
                tiles.append(tile)
            reg_img_montage = montage.montage(tiles, config)
            path = osp.join(output_dir, 'bestFocus', 'reg{:03d}_montage.tif'.format(ireg+1))
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

