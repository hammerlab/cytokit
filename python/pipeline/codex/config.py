import json
import os
import codex
from os import path as osp


def _load_json_config(data_dir, filename):
    f = osp.join(data_dir, filename)
    if not osp.exists(f):
        raise ValueError('Required configuration file "{}" does not exist'.format(f))
    with open(f, 'r') as fd:
        return json.load(fd)


def _load_experiment_config(data_dir):
    return _load_json_config(data_dir, 'Experiment.json')


def _load_processing_options(data_dir):
    return _load_json_config(data_dir, 'processingOptions.json')


def _load_channel_names(data_dir):
    f = osp.join(data_dir, 'channelNames.txt')
    if not osp.exists(f):
        raise ValueError('Required channel names configuration file "{}" does not exist'.format(f))
    with open(f, 'r') as fd:
        return [l.strip() for l in fd.readlines() if l.strip()]


class CODEXConfigV1(object):

    def __init__(self, exp_config, processing_options, channel_names):
        self.exp_config = exp_config
        self.processing_options = processing_options
        self.channel_names = channel_names

    def get_channel_names(self):
        return self.channel_names

    def n_cycles(self):
        return self.exp_config['num_cycles']

    def n_z_planes(self):
        return self.exp_config['num_z_planes']

    def n_channels_per_cycle(self):
        return len(self.exp_config['channel_names'])

    def _n_actual_channels(self):
        return len(self.channel_names)

    def _n_expected_channels(self):
        return self.n_cycles() * self.n_channels_per_cycle()

    def _validate(self):
        if self._n_actual_channels() != self._n_expected_channels():
            raise ValueError(
                'Full list of channel names does not have length equal '
                'to num_cycles * n_channels_per_cycle; '
                'n expected channel names = {}, n actual channel names = {}'
                .format(self._n_expected_channels(), self._n_actual_channels())
            )
        return self

    @staticmethod
    def load(data_dir):
        """Load all CODEX related configuration files given a primary data directory"""
        return CODEXConfigV1(
            _load_experiment_config(data_dir),
            _load_processing_options(data_dir),
            _load_channel_names(data_dir)
        )._validate()


def load(data_dir):
    version = codex.get_version()
    if version == '1':
        return CODEXConfigV1.load(data_dir)
    else:
        raise ValueError(
            'CODEX Version "{}" not supported (determined by env variable {})'
            .format(version, codex.ENV_CODEX_VERSION)
        )
