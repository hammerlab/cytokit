#!/usr/bin/python
"""Analysis CLI application"""
import fire
import codex
from codex import cli
from codex import config as codex_config
from codex.ops import analysis as codex_analysis
from codex.ops import op as codex_op
import os.path as osp
import papermill as pm
import logging
logging.basicConfig(level=logging.INFO, format=cli.LOG_FORMAT)


def _run_ops(data_dir, op_classes, config_path=None, config=None):
    if not data_dir or not osp.exists(data_dir):
        raise ValueError('Provided data directory "{}" does not exist'.format(data_dir))

    if not config_path:
        config_path = data_dir
    if not config:
        config = codex_config.load(config_path)

    # "Register the environment" meaning that any variables not explicitly defined by env variables
    # should set based on what is present in the configuration
    config.register_environment()

    for opc in op_classes:
        logging.info('Starting "{}" analysis operation'.format(opc))
        analysis_op = opc(config)
        analysis_op.run(data_dir)
        logging.info('Completed "{}" analysis operation'.format(opc))

    logging.info('Analysis execution complete')


class Analysis(object):

    def run(self, data_dir, config_path=None):
        if config_path is None:
            config_path = data_dir
        config = codex_config.load(config_path)

        analysis_params = config.analysis_params
        if len(analysis_params) == 0:
            raise ValueError(
                'Project configuration at "{}" does not currently have any analysis operations specified; '
                'Either specify them in the configuration or run individual operations using this same script '
                '(see ctk-analysis help for available options)'
            )

        op_classes = []
        for op_name in analysis_params.keys():
            if op_name not in codex_analysis.OP_CLASSES_MAP:
                raise ValueError(
                    'Analysis operation "{}" specified in configuration is not valid.  Must be one of {}'
                    .format(list(codex_analysis.OP_CLASSES_MAP.keys()))
                )
            op_classes.append(codex_analysis.OP_CLASSES_MAP[op_name])

        self._run_ops(data_dir, op_classes, config_path=config_path, config=config)
        logging.info('Analysis execution complete')

    def run_best_focus_montage_generator(self, data_dir, config_path=None):
        _run_ops(data_dir, [codex_analysis.BestFocusMontageGenerator], config_path=config_path)


if __name__ == '__main__':
    fire.Fire(Analysis)
