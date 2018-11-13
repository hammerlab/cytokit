#!/usr/bin/python
"""Config CLI application"""
import fire
import os
import os.path as osp
import logging
import copy
import pprint
from cytokit import cli
from cytokit import io as cytokit_io
logger = logging.getLogger(__name__)

PROP_CHAR = '.'


def _write_config(path, data):
    # Write config as either json or yaml, depending on the assigned extension
    if path.endswith('.yaml'):
        import yaml
        with open(path, 'w') as fd:
            yaml.dump(data, fd)
    elif path.endswith('.json'):
        import json
        with open(path, 'w') as fd:
            json.dump(data, fd)
    else:
        raise ValueError('Configuration filepath "{}" does not have a valid extension'.format(path))


class ConfigEditor(object):

    def __init__(self, config, output_dir=None):
        self.config = config
        self.output_dir = output_dir
        self.data = None
        self.reset()

    def _keys(self, property):
        if not property:
            return []
        return property.split(PROP_CHAR)

    def _get(self, keys):
        m = self.data
        # Recursively fetch target dictionary
        if keys:
            for k in keys:
                if k not in m:
                    m[k] = {}
                m = m[k]
        return m

    def reset(self):
        """Reset current configuration to base configuration"""
        self.data = copy.deepcopy(self.config.to_dict())
        return self

    def show(self, property=None):
        """Display the values associated with a property
        Args:
            property: String name of property, delimited by '.' for nested properties; Examples:
                - 'processor.args.run_illumination_correction'
                - 'processor.cytometry.segmentation_params.nucleus_dilation'
                - 'processor.best_focus.channel'
                Note that if a property does not exist, an empty dictionary will be created at that level
                in the configuration to contain it in the result and that if the property is None or empty,
                the entire configuration will be shown
        """
        if property:
            print('Configuration values present under property "{}":'.format(property))
        else:
            print('Current configuration:')
        pprint.pprint(self._get(self._keys(property)), indent=2)
        return self

    def set(self, property, value):
        """Set a property in the current configuration

        Args:
            property: String name of property, delimited by '.' for nested properties; Examples:
                - 'processor.args.run_illumination_correction'
                - 'processor.cytometry.segmentation_params.nucleus_dilation'
                - 'processor.best_focus.channel'
                Note that if a property does not exist, an empty dictionary will be created at that level
                in the configuration to contain it in the result
            value: Value to associate with property (can be string, numeric, lists, maps, etc.)
        """
        keys = self._keys(property)
        if not keys:
            logger.error('Cannot assign value for empty property string (action will be ignored)')
            return self

        # Make assignment for property after recursive fetch
        m = self._get(keys[:-1])
        m[keys[-1]] = value
        return self

    def add(self, property, value):
        """Add a value to an existing list of properties

        :param property: String name of property, delimited by '.' for nested properties; Must result to a location
            in the configuration object that is either null or a list
        :param value: Value to add to string
        """
        keys = self._keys(property)
        if not keys:
            logger.error('Cannot assign value for empty property string (action will be ignored)')
            return self

        # Make addition or list creation for property after recursive fetch
        m = self._get(keys[:-1])
        if keys[-1] not in m:
            m[keys[-1]] = []
        m[keys[-1]].append(value)
        return self

    def save_variant(self, path):
        """Save current state of configuration as a variant within experiment output directory

        Args:
            path: Relative path to configuration variant within experiment output (typically
                "v01/config", "v02/config", etc.)
        """
        return self.save(path)

    def save(self, path, name='experiment.yaml', relative=True):
        """Save current state of configuration

        Args:
            path: Directory in which to save configuration (by default, this should be a relative path from the
                root of the output directory)
            name: Name of file to save configuration in (default 'experiment.yaml')
            relative: Flag indicating whether or not path should be treated as relative to the output
                directory (default is True)
        """
        if relative and self.output_dir is None:
            raise ValueError(
                '`output_dir` argument must be set when saving configurations in '
                'paths relative to experiment output directory'
            )
        # Assume file is relative to output dir if relative flag set
        if relative:
            path = osp.join(self.output_dir, path, name)
        # Otherwise, use the path as is unless the filename was explicitly set to None, in which
        # case create an absolute path to the resulting file
        elif name is not None:
            path = osp.join(path, name)

        if not osp.exists(osp.dirname(path)):
            logger.info('Creating non-existent directory "{}" for configuration'.format(osp.dirname(path)))
            os.makedirs(osp.dirname(path), exist_ok=True)

        _write_config(path, self.data)
        logger.info('Configuration saved to path "{}"'.format(path))
        return self

    def exit(self):
        return None


class Config(cli.CLI):

    def editor(self, base_config_path, output_dir=None):
        """Start configuration editing command chain

        This command can be used to make modifications to an existing "base" configuration, which is helpful
        for generating variants of experiments dynamically.

        Args:
            base_config_path: Directory or absolute path for base configuration
            output_dir: Output directory under which generated configurations will be saved (should generally
                be the same as experiment output directory)
        Returns:
            New ConfigEditor CLI
        """
        return ConfigEditor(cli.get_config(base_config_path), output_dir)

