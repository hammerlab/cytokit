import os

ENV_CODEX_VERSION = 'CODEX_VERSION'


def get_version():
    return os.getenv(ENV_CODEX_VERSION, '1')
