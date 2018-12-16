#!/usr/bin/python
"""CLI application for downloading necessary data and model files"""
import fire
import logging
from cytokit import data
from cytokit import cli


class Download(cli.CLI):

    def models(self):
        logging.info('Downloading best focus classifier files')
        data.initialize_best_focus_model()

        logging.info('Downloading cytometry model files')
        data.initialize_cytometry_2d_model()

        logging.info('Model downloads complete')


if __name__ == '__main__':
    fire.Fire(Download)
