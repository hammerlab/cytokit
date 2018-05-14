import json
import sys
import os
import codex


def record_execution(output_dir, filename='processor_execution.json'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, 'w') as fd:
        json.dump({'args': sys.argv, 'env': codex.get_env_vars()}, fd)
    return path


def record_processor_data(data, output_dir, filename='processor_data.json'):
    import pandas as pd
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    # Use pandas for serialization as it has built-in numpy type converters
    pd.Series(data).to_json(path, orient='index')
    return path
