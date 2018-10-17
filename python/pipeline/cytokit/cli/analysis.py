#!/usr/bin/python
"""Analysis CLI application"""
import fire
import os.path as osp
import logging
from cytokit import cli
from cytokit.function import core
from cytokit import io as cytokit_io


class Analysis(cli.DataCLI):

    def _get_function_configs(self):
        return self.config.analysis_params

    def processor_data_summary(self):
        logging.info('Running processor data summary operation')
        processor_data_path = osp.join(self.data_dir, cytokit_io.get_processor_data_path())
        nb_name = 'processor_data_analysis.ipynb'
        nb_output_path = osp.join(self.data_dir, osp.dirname(processor_data_path), 'processor_data_analysis.ipynb')
        nb_params = {'processor_data_path': processor_data_path}
        core.run_nb(nb_name, nb_output_path, nb_params)
        logging.info('Processor data summary complete; view results with `jupyter notebook %s`', nb_output_path)

    def aggregate_cytometry_statistics(self, mode='all', export_csv=True, export_fcs=True):
        logging.info('Running cytometry statistics aggregation')
        core.aggregate_cytometry_statistics(
            self.data_dir, self.config, mode=mode, export_csv=export_csv, export_fcs=export_fcs)


if __name__ == '__main__':
    fire.Fire(Analysis)
