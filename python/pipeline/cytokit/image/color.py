from collections import OrderedDict

COLORS = OrderedDict([
    ('red', [1, 0, 0]),
    ('green', [0, 1, 0]),
    ('blue', [0, 0, 1]),
    ('cyan', [0, 1, 1]),
    ('magenta', [1, 0, 1]),
    ('gray', [1, 1, 1]),
    ('none', [0, 0, 0])
])


def get_color_names():
    return list(COLORS.keys())


def get_colors(n, values=False):
    l = list(COLORS.values()) if values else list(COLORS.keys())
    return [l[i % len(l)] for i in range(n)]


def map(color_name):
    if not color_name:
        return COLORS['gray']
    return COLORS.get(color_name.lower(), COLORS['gray'])
