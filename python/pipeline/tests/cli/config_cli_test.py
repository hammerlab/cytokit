import unittest
import os.path as osp
import cytokit
from cytokit import config as cytokit_config
from cytokit import simulation as cytokit_simulation
from cytokit.cli.config import ConfigEditor


class TestConfigCLI(unittest.TestCase):

    def _get_example_conf(self):
        return cytokit_simulation.get_example_config(example_name='ex1')

    def test_add_value(self):
        # TODO: extend this to test cli as system commands instead
        # Example command:
        # cytokit config editor --base-config-path=/lab/repos/cytokit/python/pipeline/cytokit/configs/ - \
        # add operator '{extract: {name:test, channels: [c1,c2]} }' show
        editor = ConfigEditor(self._get_example_conf())
        editor.add('operator', {'extract': {'name': 'test', 'channels': ['c1', 'c2']}})
        extract_names = [m.get('extract', {}).get('name', '') for m in editor.data['operator']]
        self.assertTrue('test' in extract_names)
