from collections import OrderedDict

DEFAULT_COLORS = OrderedDict([
    ('red', [1, 0, 0]),
    ('green', [0, 1, 0]),
    ('blue', [0, 0, 1]),
    ('cyan', [0, 1, 1]),
    ('magenta', [1, 0, 1]),
    ('gray', [1, 1, 1])
])


def get_all_color_names():
    return list(DEFAULT_COLORS.keys())


def get_defaults(n, values=False):
    l = list(DEFAULT_COLORS.values()) if values else list(DEFAULT_COLORS.keys())
    return [l[i % len(l)] for i in range(n)]


def map(color_name):
    if not color_name:
        return DEFAULT_COLORS['gray']
    return DEFAULT_COLORS.get(color_name.lower(), DEFAULT_COLORS['gray'])