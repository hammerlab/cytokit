#!/usr/bin/env python
import fire
from cytokit.cli import analysis, operator, processor, application, config, download


class Cytokit(object):

    def processor(self):
        return processor.Processor

    def operator(self):
        return operator.Operator

    def analysis(self):
        return analysis.Analysis

    def application(self):
        return application.Application
    
    def config(self):
        return config.Config

    def download(self):
        return download.Download


def main():
    fire.Fire(Cytokit)


if __name__ == '__main__':
    main()
