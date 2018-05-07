import json
import sys
import os
import codex


def record_execution(output_dir, filename):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, 'w') as fd:
        json.dump({'args': sys.argv, 'env': codex.get_env_vars()}, fd)
    return path