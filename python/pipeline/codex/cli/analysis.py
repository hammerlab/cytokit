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


def _run_ops(data_dir, config, op_classes):
    for opc in op_classes:
        analysis_op = opc(config)

        op_config = analysis_op.get_analysis_op_config()
        if not op_config.get('enabled', True):
            logging.info('Skipping "{}" analysis operation since it has been explicitly disabled'.format(opc.__name__))
            continue

        logging.info('Starting "{}" analysis operation'.format(opc.__name__))
        analysis_op.run(data_dir)
        logging.info('Completed "{}" analysis operation'.format(opc.__name__))

    logging.info('Analysis execution complete')


def _get_config(data_dir, config_path=None):
    # Load experiment configuration and "register" the environment meaning that any variables not
    # explicitly defined by env variables should set based on what is present in the configuration
    # (it is crucial that this happen first)
    if not config_path:
        config_path = data_dir
    config = codex_config.load(config_path)
    config.register_environment()
    return config


class Analysis(object):

    def run(self, data_dir, config_path=None):
        config = _get_config(data_dir, config_path)

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
                    .format(op_name, list(codex_analysis.OP_CLASSES_MAP.keys()))
                )
            op_classes.append(codex_analysis.OP_CLASSES_MAP[op_name])

        _run_ops(data_dir, config, op_classes)
        logging.info('Analysis execution complete')

    def run_best_focus_montage_generator(self, data_dir, config_path=None):
        _run_ops(data_dir, _get_config(data_dir, config_path), [codex_analysis.BestFocusMontageGenerator])


if __name__ == '__main__':
    fire.Fire(Analysis)
