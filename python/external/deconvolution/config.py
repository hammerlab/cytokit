import os
import json
from os import path as osp
import numpy as np
import logging
logger = logging.getLogger('CODEXConfig')


def _load_experiment_config(args):
    f = osp.join(args.raw_dir, 'Experiment.json')
    if not osp.exists(f):
        raise ValueError('Required experiment configuration file "{}" does not exist'.format(f))
    with open(f, 'r') as fd:
        return json.load(fd)


def _load_channel_names(args):
    f = osp.join(args.raw_dir, 'channelNames.txt')
    if not osp.exists(f):
        raise ValueError('Required channel names configuration file "{}" does not exist'.format(f))
    with open(f, 'r') as fd:
        return [l.strip() for l in fd.readlines() if l.strip()]


class CODEXConfig(object):

    def __init__(self, exp_config, channel_names):
        self.exp_config = exp_config
        self.channel_names = channel_names

    def all_channel_names(self):
        return self.channel_names

    def n_cycles(self):
        return self.exp_config['num_cycles']

    def n_z_planes(self):
        return self.exp_config['num_z_planes']

    def n_channels_per_cycle(self):
        return len(self.exp_config['channel_names'])

    def n_actual_channels(self):
        return len(self.channel_names)

    def n_expected_channels(self):
        return self.n_cycles() * self.n_channels_per_cycle()


def load_config(args):
    config = CODEXConfig(
        _load_experiment_config(args),
        _load_channel_names(args)
    )

    logger.debug('Experiment configuration summary:')
    logger.debug('\tNum cycles = {}'.format(config.n_cycles()))
    logger.debug('\tNum z planes = {}'.format(config.n_z_planes()))
    logger.debug('\tChannels expected per cycle = {}'.format(config.n_channels_per_cycle()))
    logger.debug('\tChannel names list length = {}'.format(config.n_actual_channels()))

    if config.n_actual_channels() != config.n_expected_channels():
        raise ValueError(
            'Full list of channel names does not have length equal '
            'to num_cycles * n_channels_per_cycle; '
            'n expected channel names = {}, n actual channel names = {}'
            .format(config.n_expected_channels(), config.n_actual_channels())
        )

    return config