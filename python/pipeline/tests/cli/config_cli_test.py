import unittest
import os.path as osp
import cytokit
from cytokit import config as cytokit_config
from cytokit.cli.config import ConfigEditor


class TestConfigCLI(unittest.TestCase):

    def _get_example_conf(self):
        conf_dir = osp.join(cytokit.conf_dir, 'v0.1', 'examples', 'ex1')
        return cytokit_config.load(conf_dir)

    def test_add_value(self):
        # TODO: extend this to test cli as system commands instead
        # Example command:
        # cytokit config editor --base-config-path=/lab/repos/cytokit/python/pipeline/cytokit/configs/ - \
        # add operator '{extract: {name:test, channels: [c1,c2]} }' show
        editor = ConfigEditor(self._get_example_conf())
        editor.add('operator', {'extract': {'name': 'test', 'channels': ['c1', 'c2']}})
        extract_names = [m.get('extract', {}).get('name', '') for m in editor.data['operator']]
        self.assertTrue('test' in extract_names)
