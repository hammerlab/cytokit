import json
import sys
import os
import codex

LOG_FORMAT = '%(asctime)s:%(levelname)s:%(process)d:%(name)s: %(message)s'
DEFAULT_PROCESSOR_EXEC_FILENAME = 'processor_execution.json'
DEFAULT_PROCESSOR_DATA_FILENAME = 'processor_data.json'

def record_execution(output_dir, filename=DEFAULT_PROCESSOR_EXEC_FILENAME):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, 'w') as fd:
        json.dump({'args': sys.argv, 'env': codex.get_env_vars()}, fd)
    return path


def record_processor_data(data, output_dir, filename=DEFAULT_PROCESSOR_DATA_FILENAME):
    import pandas as pd
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    # Use pandas for serialization as it has built-in numpy type converters
    pd.Series(data).to_json(path, orient='index')
    return path
