#!/usr/bin/python
"""CODEX analysis CLI application"""
import fire
import codex
from codex import cli
import os.path as osp
import papermill as pm
import logging
logging.basicConfig(level=logging.INFO, format=cli.LOG_FORMAT)

class CodexAnalyzer(object):

    def _get_nb_path(self, nb_name):
        return osp.join(codex.nb_dir, 'data_analysis', nb_name)

    def processor_data(self, output_dir, filename=cli.DEFAULT_PROCESSOR_DATA_FILENAME):
        """Analyze data collected by the processing pipeline
        
        This operation will execute a parameterized notebook and print the notebook location
        (as well as how to view it) on completion.

        Args:
            output_dir: Directory containing processor output (should be same as output_dir specified
                to that application)
            filename: In the event that the processor data (json) file was given a non-default name,
                it can be specified here; otherwise, the default value should not be changed
        """
        logging.info('Running processor data analysis')
        processor_data_path = osp.join(output_dir, filename)
        nb_input_path = self._get_nb_path('processor_data_analysis.ipynb')
        nb_output_path = osp.join(output_dir, 'processor_data_analysis.ipynb')
        pm.execute_notebook(nb_input_path, nb_output_path, parameters={'processor_data_path': processor_data_path})
        logging.info('Processor data analysis complete; view with `jupyter notebook {}`'.format(nb_output_path))


if __name__ == '__main__':
    fire.Fire(CodexAnalyzer)