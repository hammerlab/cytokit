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


class CodexConfigV1(object):

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

    def tile_width(self):
        return self.exp_config['tile_width']

    def tile_height(self):
        return self.exp_config['tile_height']

    def tile_dims(self):
        """Get tile dims as (cycles, width, height, z, channels)"""
        return self.n_cycles(), self.tile_width(), self.tile_height(), self.n_z_planes(), self.n_channels_per_cycle()

    def overlap_x(self):
        return self.exp_config['tile_overlap_X']

    def overlap_y(self):
        return self.exp_config['tile_overlap_Y']

    def drift_compensation_reference(self):
        """Get reference image configured for drift compensation
        Returns:
            (cycle, channel) - 0-based indexes for cycle and channel
        """
        cycle = self.exp_config['driftCompReferenceCycle'] - 1
        channel = self.exp_config['drift_comp_channel'] - 1
        return cycle, channel

    def best_focus_reference(self):
        """Get reference image configured for best focus plan selection
        Returns:
            (cycle, channel) - 0-based indexes for cycle and channel
        """
        cycle = self.exp_config['bestFocusReferenceCycle'] - 1
        channel = self.exp_config['best_focus_channel'] - 1
        return cycle, channel

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
        return CodexConfigV1(
            _load_experiment_config(data_dir),
            _load_processing_options(data_dir),
            _load_channel_names(data_dir)
        )._validate()


def load(data_dir):
    version = codex.get_version()
    if version == '1':
        return CodexConfigV1.load(data_dir)
    else:
        raise ValueError(
            'CODEX Version "{}" not supported (determined by env variable {})'
            .format(version, codex.ENV_CODEX_VERSION)
        )
