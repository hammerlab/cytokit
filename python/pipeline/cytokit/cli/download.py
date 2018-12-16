#!/usr/bin/python
"""CLI application for downloading necessary data and model files"""
import fire
import logging
from cytokit import data


class Download(cli.CLI):

    def models(self):
        logging.info('Download best focus classifier files')
        data.initialize_best_focus_model()

        logging.info('Download cytometry model files')
        data.initialize_cytometry_2d_model()

        logging.info('Model downloads complete')


if __name__ == '__main__':
    fire.Fire(Download)
