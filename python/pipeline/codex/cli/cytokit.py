#!/usr/bin/env python
import fire
from codex.cli import analysis, operator, processor, config


class Cytokit(object):

    def processor(self):
        return processor.Processor

    def operator(self):
        return operator.Operator

    def analysis(self):
        return analysis.Analysis

    def config(self):
        return config.Config


if __name__ == '__main__':
    fire.Fire(Cytokit)
