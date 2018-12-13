#!/usr/bin/python
"""Application launcher CLI"""
import fire
import logging
from cytokit import cli


class Application(cli.CLI):

    def run_explorer(self):
        logging.info('Running explorer app')
        from cytokit_app.explorer import app
        app.run()


if __name__ == '__main__':
    fire.Fire(Application)
