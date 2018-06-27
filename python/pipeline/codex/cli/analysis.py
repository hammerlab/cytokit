#!/usr/bin/python
"""Analysis CLI application"""
import fire
import codex
import logging
from codex import cli
from codex.ops import analysis as codex_analysis


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


class Analysis(cli.CLI):

    def run(self):
        analysis_params = self.config.analysis_params
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

        _run_ops(self.data_dir, self.config, op_classes)
        logging.info('Analysis execution complete')

    # def run_best_focus_montage_generator(self, data_dir, config_path=None):
    #     _run_ops(data_dir, _get_config(data_dir, config_path), [codex_analysis.BestFocusMontageGenerator])


if __name__ == '__main__':
    fire.Fire(Analysis)
