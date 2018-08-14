# Cytokit Explorer App
#
# References:
# - Plotly modebar options: https://github.com/plotly/plotly.js/blob/master/src/components/modebar/buttons.js
import dash
import json
import fire
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
from codex_app.explorer import data as data
from codex_app.explorer.data import cache as ac
from codex_app.explorer import lib as lib
from codex_app.explorer.config import cfg
from codex_app.explorer import color
from codex.cytometry.cytometer import DEFAULT_CELL_INTENSITY_PREFIX, DEFAULT_NUCL_INTENSITY_PREFIX
from collections import OrderedDict
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


data.initialize()

# processors = {}
app = dash.Dash()
# app.config['suppress_callback_exceptions'] = True
app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/dZVMbK.css'})


TILE_IMAGE_STYLE = dict(height='750px')
MONTAGE_IMAGE_STYLE = None
TITLE_STYLE = {'textAlign': 'center', 'margin-top': '0px', 'margin-bottom': '0px'}
SUB_TITLE_STYLE = {**{'font-style': 'italic'}, **TITLE_STYLE}


def get_montage_image():
    return ac['processor']['montage'].run(data.get_montage_image())


def get_tile_image(apply_display_settings=True):
    img = data.get_tile_image(*ac['selection']['tile']['coords'])
    return ac['processor']['tile'].run(img) if apply_display_settings else img


def get_tile_nchannels():
    return len(data.get_tile_image_channels())


ac['shape']['montage'] = data.get_montage_image().shape
ac['processor']['montage'] = lib.ImageProcessor(ac['shape']['montage'][0])
ac['layouts']['montage'] = lib.get_interactive_image_layout(get_montage_image())

ac['shape']['tile'] = data.get_tile_image().shape
ac['processor']['tile'] = lib.ImageProcessor(ac['shape']['tile'][0])
ac['selection']['tile']['coords'] = (0, 0)
ac['layouts']['tile'] = lib.get_interactive_image_layout(get_tile_image())


def get_graph_axis_selections():
    return dict(
        xvar=data.db.get('app', 'axis_x_var'),
        xscale=data.db.get('app', 'axis_x_scale') or 'linear',
        yvar=data.db.get('app', 'axis_y_var'),
        yscale=data.db.get('app', 'axis_y_scale') or 'linear'
    )


def get_base_data():
    # Pull cell expression data frame and downsample if necessary
    df = data.get_cytometry_stats()

    # Apply custom code if any has been passed in
    if data.db.exists('app', 'operation_code'):
        code = data.db.get('app', 'operation_code')
        n = len(df)
        logger.info('Applying custom code:\n%s', code)
        local_vars = {'df': df}
        exec(code, globals(), local_vars)
        df = local_vars['df']
        logger.info('Size of frame before custom operations = %s, after = %s', n, len(df))

    return df


def apply_log(v):
    return np.log10(v)


def invert_log(v):
    return np.power(10, v)


def get_graph_data():
    selections = get_graph_axis_selections()

    if selections['xvar'] is None:
        ac['counts']['graph_data']['n'] = 0
        return None

    # Create list of fields to extract that are always necessary
    cols = list(cfg.CYTO_FIELDS.keys())

    # Map to contain keys '[x|y]var' to the name of the field to duplicate
    # as that new column in the data frame
    vars = {}
    if selections['xvar']:
        vars['xvar'] = selections['xvar']
        if selections['xvar'] not in cols:
            cols += [selections['xvar']]
    if selections['yvar']:
        vars['yvar'] = selections['yvar']
        if selections['yvar'] not in cols:
            cols += [selections['yvar']]

    df = get_base_data().copy()[cols]
    for k, v in vars.items():
        df[k] = df[v]

    if 'xvar' in df and selections['xscale'] == 'log':
        df['xvar'] = apply_log(df['xvar'])
    if 'yvar' in df and selections['yscale'] == 'log':
        df['yvar'] = apply_log(df['yvar'])

    ac['counts']['graph_data']['n'] = len(df)

    return df


def get_graph_figure():
    d = get_graph_data()

    x = d['xvar'] if d is not None and 'xvar' in d else []
    y = d['yvar'] if d is not None and 'yvar' in d else []

    if len(y) > 0:
        fig_data = [
            dict(
                x=x,
                y=y,
                mode='markers',
                marker={'opacity': cfg.graph_point_opacity},
                type='scattergl'
            )
        ]
    else:
        fig_data = [
            dict(x=x, type='histogram')
        ]

    selections = get_graph_axis_selections()
    fig_layout = {
        'margin': dict(t=25),
        'title': '',
        'titlefont': {'size': 12},
        'xaxis': {'title': (selections['xvar'] or '').upper(), 'autorange': True},
        'yaxis': {'title': (selections['yvar'] or '').upper(), 'autorange': True},
        'dragmode': 'select'
    }

    fig = dict(data=fig_data, layout=fig_layout)
    return fig


def get_graph():
    return dcc.Graph(
        id='graph',
        figure=get_graph_figure(),
        animate=False,
        config={
            'showLink': False, 'displaylogo': False, 'linkText': '',
            'modeBarButtonsToRemove': ['toggleSpikelines']
        }
    )


def _ch_disp_name(ch):
    """Convert channel names like 'proc_Plasmid' to 'Plasmid'"""
    return '_'.join(ch.split('_')[1:])


def get_image_settings_layout(id_format, channel_names, class_name, type='tile'):

    values = data.db.get('app', type + '_channel_ranges')

    channel_dtype = data.get_channel_dtype_map()
    assert len(channel_dtype) == len(channel_names), \
        'Channel data type map "{}" does not match length of channel list "{}"'\
        .format(channel_dtype, channel_names)
    ranges = [[0, np.iinfo(t).max] for t in channel_dtype.values()]

    colors = data.db.get('app', type + '_channel_colors')
    if colors is None or len(colors) != len(channel_names):
        colors = color.get_defaults(len(channel_names))

    def get_value_range(i):
        if values:
            return values[i]
        if channel_names[i].startswith(data.CH_SRC_CYTO):
            return [0, 1]
        return ranges[i]

    return html.Div([
            html.Div([
                html.Div(_ch_disp_name(channel_names[i]), style={'width': '50%', 'display': 'inline-block'}),
                html.Div(dcc.Dropdown(
                    id=id_format.format(str(i) + '-color'),
                    options=[{'label': c.title(), 'value': c} for c in color.get_all_color_names()],
                    value=colors[i],
                    clearable=False,
                    searchable=False
                ), style={'width': '50%', 'display': 'inline-block'}),
                dcc.RangeSlider(
                    id=id_format.format(str(i) + '-range'),
                    min=ranges[i][0],
                    max=ranges[i][1],
                    step=1,
                    # Override default value to 1 for labeled images
                    value=get_value_range(i),
                    allowCross=False
                )
            ])
            for i in range(len(channel_names))
        ],
        className=class_name,
        # Pad slightly on left to avoid slider knob overlap with tile figure
        style={'padding-left': '10px'}
    )


def get_axis_settings_layout(axis):
    selections = get_graph_axis_selections()
    var = selections[axis + 'var']
    scale = selections[axis + 'scale']

    # Create field list starting with preconfigured set of always necessary fields
    field_names = OrderedDict(cfg.CYTO_FIELDS)

    # Add list of possible field values based on "base" dataset, which may
    # include expression levels for different cell components
    regex = DEFAULT_CELL_INTENSITY_PREFIX + '|' + DEFAULT_NUCL_INTENSITY_PREFIX
    for c in get_base_data().filter(regex=regex).columns.tolist():
        field_names[c] = c

    return html.Div([
            html.Div(
                html.P(axis.upper()),
                style={'width': '3%', 'display': 'inline-block', 'position': 'relative', 'bottom': '10px'}
            ),
            html.Div(
                dcc.Dropdown(
                    id='axis_' + axis + '_var',
                    options=[
                        {'label': v, 'value': k}
                        for k, v in field_names.items()
                    ],
                    value=var
                ),
                style={'width': '60%', 'display': 'inline-block'}
            ),
            html.Div(dcc.Dropdown(
                id='axis_' + axis + '_scale',
                options=[
                    {'label': 'Log', 'value': 'log'},
                    {'label': 'Linear', 'value': 'linear'}
                ],
                value=scale,
                searchable=False,
                clearable=False
            ), style={'width': '30%', 'display': 'inline-block'}),

        ]
    )


def get_operation_code_layout():
    code = ''
    if data.db.exists('app', 'operation_code'):
        code = data.db.get('app', 'operation_code')
    return [
        html.Div('Custom Operations:', style={'width': '100%'}),
        dcc.Textarea(
            id='operation-code',
            placeholder='Enter custom pre-processing code',
            value=code,
            style={'width': '90%', 'height': '100%'},
            wrap=False,
            rows=25
        ),
        html.Button('Apply', id='apply-button')
        # html.Div('', id='code-message')
    ]


##############
# App Layout #
##############


app.layout = html.Div([
        html.Div(
            className='row',
            children=html.Div([
                html.P(
                    'Cytokit Explorer 1.0',
                    style={
                        'color': 'white', 'float': 'left', 'padding-left': '15px',
                        'padding-top': '10px', 'font': '400 16px system-ui'
                    }
                ),
                html.Div(html.Button(
                    'Save Settings', id='save-button',
                    style={'float': 'right', 'color': 'white', 'border-width': '0px'}
                )),
                # html.Div(html.Button(
                #     'Export', id='exp-button',
                #     style={'float': 'right', 'color': 'white', 'border-width': '0px'}
                # ))
            ]),
            style={'backgroundColor': 'rgb(31, 119, 180)'}
        ),
        html.Div(
            className='row',
            children=[
                html.Div([
                        get_axis_settings_layout('x'),
                        get_axis_settings_layout('y'),
                        get_graph()
                    ],
                    className='five columns'
                ),
                html.Div([
                        html.Div(id='montage-title'),
                        lib.get_interactive_image('montage', ac['layouts']['montage'], style=MONTAGE_IMAGE_STYLE)
                    ],
                    style={'overflowY': 'scroll', 'overflowX': 'scroll'},
                    className='four columns'
                ),
                html.Div(
                    get_operation_code_layout(),
                    className='three columns'
                )
            ],
            style={'margin-top': '10px'}
        ),
        html.Div(
            className='row',
            children=[
                html.Div(
                    html.H4('Selected Cells', style=TITLE_STYLE),
                    id='single-cells',
                    className='two columns'
                ),
                html.Div([
                        html.Div(id='tile-title'),
                        lib.get_interactive_image('tile', ac['layouts']['tile'], style=TILE_IMAGE_STYLE)
                    ],
                    className='seven columns',
                    style={'overflowY': 'scroll', 'overflowX': 'scroll'}
                ),
                get_image_settings_layout('tile-ch-{}', data.get_tile_image_channels(), 'three columns')
            ]
        ),
        html.Div('', id='message'),
        html.P(id='null1')
    ]
)


#############
# Callbacks #
#############


def handle_display_settings_update(channel_ranges, channel_colors):
    data.db.put('app', 'tile_channel_ranges', channel_ranges)
    data.db.put('app', 'tile_channel_colors', channel_colors)
    ac['processor']['tile'].ranges = channel_ranges
    ac['processor']['tile'].colors = [color.map(c) for c in channel_colors]


def handle_selected_data_update(selected_data):
    data.db.put('app', 'selected_data', selected_data)


def handle_montage_click_data(click_data):
    # montage click data example:
    # {'points': [{'x': 114.73916, 'curveNumber': 0, 'pointNumber': 421, 'y': 306.4889, 'pointIndex': 421}]}
    if click_data:
        sy, sx = cfg.montage_target_shape
        ry, rx = cfg.region_shape
        py, px = click_data['points'][0]['y'], click_data['points'][0]['x']
        ac['selection']['tile']['coords'] = (int(px // (sx / rx)), ry - int(py // (sy / ry)) - 1)


@app.callback(
    Output('graph', 'figure'), [
        Input('axis_x_var', 'value'),
        Input('axis_x_scale', 'value'),
        Input('axis_y_var', 'value'),
        Input('axis_y_scale', 'value'),
        Input('apply-button', 'n_clicks')

    ], [
        State('operation-code', 'value')
    ]
)
def update_graph(xvar, xscale, yvar, yscale, _, code):
    data.db.put('app', 'axis_x_var', xvar)
    data.db.put('app', 'axis_x_scale', xscale)
    data.db.put('app', 'axis_y_var', yvar)
    data.db.put('app', 'axis_y_scale', yscale)
    data.db.put('app', 'operation_code', code)
    fig = get_graph_figure()
    return fig


@app.callback(Output('message', 'children'), [Input('save-button', 'n_clicks')])
def save_state(n_clicks):
    if n_clicks is None:
        return ''
    path = data.db.save()
    return 'Application state saved to "{}"'.format(path)


def _rescale_montage_coords(x, y):
    sy, sx = cfg.montage_target_scale_factors
    return x * sx, y * sy


def _relayout(figure, relayout_data):
    if relayout_data:
        if 'xaxis.range[0]' in relayout_data:
            figure['layout']['xaxis']['range'] = [
                relayout_data['xaxis.range[0]'],
                relayout_data['xaxis.range[1]']
            ]
        if 'yaxis.range[0]' in relayout_data:
            figure['layout']['yaxis']['range'] = [
                relayout_data['yaxis.range[0]'],
                relayout_data['yaxis.range[1]']
            ]
    return figure


def selection_type(selected_data):
    if selected_data is None:
        return None
    if 'range' in selected_data:
        return 'range'
    if 'points' in selected_data:
        return 'lasso'
    return None


def _apply_axis_filter(axis, df, var_range):
    assert axis in ['x', 'y']
    axis_selections = get_graph_axis_selections()

    if axis_selections[axis + 'scale'] == 'log':
        var_range = list(invert_log(np.array(var_range)))

    return df[df[axis_selections[axis + 'var']].between(*var_range)]


def get_graph_data_selection():
    selected_data = data.db.get('app', 'selected_data')
    type = selection_type(selected_data)
    if type is None:
        ac['counts']['graph_data_selection']['n'] = 0
        return None

    df = get_graph_data()
    if df is None:
        ac['counts']['graph_data_selection']['n'] = 0
        return None

    mode = '2D' if data.db.get('app', 'axis_y_var') is not None else '1D'

    if mode == '2D':
        if type == 'lasso':
            # Fetch only cells/rows corresponding to selected data in graph (and index by tile loc)
            df = df.iloc[[p['pointIndex'] for p in selected_data['points']]]
        elif type == 'range':
            df = _apply_axis_filter('x', df, selected_data['range']['x'])
            df = _apply_axis_filter('y', df, selected_data['range']['y'])
        else:
            raise ValueError('Selection type "{}" invalid'.format(type))
    else:
        if type == 'lasso':
            px = np.array([p['x'] for p in selected_data['points']])
            df = _apply_axis_filter('x', df, (px.min(), px.max()))
        elif type == 'range':
            df = _apply_axis_filter('x', df, selected_data['range']['x'])
        else:
            raise ValueError('Selection type "{}" invalid'.format(type))

    ac['counts']['graph_data_selection']['n'] = len(df)

    return df


@app.callback(Output('montage-title', 'children'), [Input('montage', 'figure'), Input('graph', 'figure')])
def update_montage_title(*_):
    ny, nx = cfg.region_shape
    ri = cfg.region_index

    children = [html.H4('Region {} ({} x {} Tiles)'.format(ri + 1, nx, ny), style=TITLE_STYLE)]

    # Add informative count data (depending on whether or not any data selections have been made)
    if ac['counts']['graph_data_selection']['n'] > 0:
        children.append(html.P(
            'Showing {} cells of {} selected ({} total in region)'.format(
                ac['counts']['montage']['n'],
                ac['counts']['graph_data_selection']['n'],
                ac['counts']['graph_data']['n']
            ), style=SUB_TITLE_STYLE
        ))
    # If there is no data selected, at least show how many cells there are total (if loaded yet)
    elif ac['counts']['graph_data']['n'] > 0:
        children.append(html.P(
            'Showing {} cells of {} total'.format(
                ac['counts']['montage']['n'],
                ac['counts']['graph_data']['n']
            ), style=SUB_TITLE_STYLE
        ))
    # Otherwise, show no subtitle

    return children


@app.callback(
    Output('montage', 'figure'),
    [Input('graph', 'selectedData'), Input('montage', 'clickData')],
    [State('montage', 'relayoutData')]
)
def update_montage(selected_data, click_data, relayout_data):

    # Persist updates to selected data
    handle_selected_data_update(selected_data)

    # Persist changes to selected tile
    handle_montage_click_data(click_data)

    # Get currently selected cell data
    df = get_graph_data_selection()

    fig_data = [{'x': [], 'y': [], 'mode': 'markers', 'type': 'scattergl'}]
    layout = ac['layouts']['montage']
    layout['shapes'] = None

    ac['counts']['montage']['n'] = 0
    if df is not None:

        # Downsample if necessary
        if len(df) > cfg.max_montage_cells:
            logger.info('Sampling montage expression data for %s cells down to %s records', len(df),
                        cfg.max_montage_cells)
            df = df.sample(n=cfg.max_montage_cells, random_state=cfg.random_state)

        ac['counts']['montage']['n'] = len(df)

        # Update montage points using region x/y of cells
        x, y = _rescale_montage_coords(df['rx'], df['ry'])
        fig_data = [{
            'x': x,
            'y': cfg.montage_target_shape[0] - y,
            'text': [
                'Cell ID (Tile): {}<br>Cell ID (Region): {}'.format(r['id'], r['rid'])
                for _, r in df[['id', 'rid']].iterrows()
            ],
            'mode': 'markers',
            'marker': {'opacity': .5, 'color': 'white'},
            'type': 'scattergl',
            'hoverinfo': 'text'
        }]

        # Add rectangular dividers between tiles in montage, if configured to do so AND
        # when some data has already been selected (otherwise lines aren't helpful)
        if cfg.montage_grid_enabled and data.db.get('app', 'selected_data') is not None:
            # Get shape (rows, cols) of individual montage grid cell by dividing target shape
            # (e.g. 512x512) by the number of tiles expected in each dimension
            shape = np.array(cfg.montage_target_shape) / np.array(cfg.region_shape)
            shapes = []
            for row in range(cfg.region_shape[0]):
                for col in range(cfg.region_shape[1]):
                    shapes.append({
                        'type': 'rect',
                        'x0': col * shape[1],
                        'y0': row * shape[0],
                        'x1': (col + 1) * shape[1],
                        'y1': (row + 1) * shape[0],
                        'line': {'color': 'rgba(0, 256, 0, .5)'}
                    })
                    # Invert row indexer since plotly figure is bottom up and cytokit convention is top down
                    if (col, cfg.region_shape[0] - row - 1) == ac['selection']['tile']['coords']:
                        shapes[-1]['fillcolor'] = 'rgba(0, 256, 0, .3)'

            layout['shapes'] = shapes

    figure = dict(data=fig_data, layout=layout)
    return _relayout(figure, relayout_data)


def _get_tile_hover_text(r):
    fields = []
    for f in cfg.CYTO_HOVER_FIELDS:
        fmt = '{}: {:.3f}'
        if f in cfg.CYTO_INT_FIELDS:
            # Avoid integer formats as values maybe floats and string formatting
            # will fail when specified as int and given float
            fmt = '{}: {:.0f}'
        fields.append(fmt.format(cfg.CYTO_FIELDS[f], r[f]))
    return '<br>'.join(fields)


def get_tile_graph_data_selection():
    df = get_graph_data_selection()
    if df is None:
        return None
    # Further restrict to only the selected tile
    df = df.set_index(['tile_x', 'tile_y']).sort_index()
    tx, ty = ac['selection']['tile']['coords']
    if (tx, ty) not in df.index:
        return None
    # Make sure to use list of tuples for slice to avoid series result with single matches
    df = df.loc[[(tx, ty)]]
    return df


def _get_tile_figure(relayout_data):
    fig_data = [{'x': [], 'y': [], 'mode': 'markers', 'type': 'scattergl'}]
    df = get_tile_graph_data_selection()

    ac['counts']['tile']['n'] = 0
    if df is not None:
        # Downsample if necessary
        if len(df) > cfg.max_tile_cells:
            logger.info('Sampling tile expression data for %s cells down to %s records', len(df), cfg.max_tile_cells)
            df = df.sample(n=cfg.max_tile_cells, random_state=cfg.random_state)
        ac['counts']['tile']['n'] = len(df)

        tx, ty = ac['selection']['tile']['coords']
        logger.info('Number of cells found in tile (x=%s, y=%s): %s', tx + 1, ty + 1, len(df))

        fig_data = [{
            'x': df['x'],
            'y': cfg.tile_shape[0] - df['y'],
            'mode': 'markers',
            'text': df.apply(_get_tile_hover_text, axis=1),
            'marker': {'opacity': 1., 'color': 'white'},
            'type': 'scattergl',
            'hoverinfo': 'text'
        }]
    figure = dict(data=fig_data, layout=ac['layouts']['tile'])
    return _relayout(figure, relayout_data)


@app.callback(Output('tile-title', 'children'), [Input('tile', 'figure')])
def update_tile_title(_):
    tx, ty = ac['selection']['tile']['coords']

    # Always include tile X/Y in title
    children = [html.H4('Tile X{}/Y{}'.format(tx + 1, ty + 1), style=TITLE_STYLE)]

    # Add informative count data (depending on whether or not any data selections have been made)
    if ac['counts']['graph_data_selection']['n'] > 0:
        children.append(html.P(
            'Showing {} cells of {} selected'.format(
                ac['counts']['tile']['n'],
                ac['counts']['graph_data_selection']['n']
            ), style=SUB_TITLE_STYLE
        ))

    return children


@app.callback(
    Output('tile', 'figure'),
    [Input('graph', 'selectedData'), Input('montage', 'clickData')] +
    [Input('tile-ch-{}-range'.format(i), 'value') for i in range(get_tile_nchannels())] +
    [Input('tile-ch-{}-color'.format(i), 'value') for i in range(get_tile_nchannels())],
    [State('tile', 'relayoutData')]
)
def update_tile(*args):
    selected_data, click_data, relayout_data = args[0], args[1], args[-1]

    # Persist updates to selected data
    handle_selected_data_update(selected_data)

    # Persist display settings updates
    nch = get_tile_nchannels()
    channel_ranges = args[2:(nch + 2)]
    channel_colors = args[(nch + 2):-1]
    handle_display_settings_update(channel_ranges, channel_colors)

    # Persist changes to selected tile
    handle_montage_click_data(click_data)

    ac['layouts']['tile'] = lib.get_interactive_image_layout(get_tile_image())
    return _get_tile_figure(relayout_data)


def get_single_cells_title(n_cells=0, n_tile=0):
    children = [html.H4('Selected Cells', style=TITLE_STYLE)]

    # Append count info to title/header only if it is informative
    if n_cells:
        children.append(html.P(
            'Showing {} cells of {} selected in tile'.format(n_cells, n_tile),
            style=SUB_TITLE_STYLE
        ))

    return children


@app.callback(Output('single-cells', 'children'), [Input('tile', 'figure')])
def update_single_cells(_):
    df = get_tile_graph_data_selection()

    channels = data.get_tile_image_channels()
    cell_boundary_channel = data.CH_SRC_CYTO + '_cell_boundary'

    if cell_boundary_channel not in channels:
        logger.warning('Cannot generate single cell images because extract does not contain cell boundary channel')
        return get_single_cells_title()
    if df is None:
        return get_single_cells_title()

    # Downsample if necessary
    n_cells_in_tile = len(df)
    if len(df) > cfg.max_single_cells:
        logger.info('Sampling single cell data for %s cells down to %s records', len(df), cfg.max_single_cells)
        df = df.sample(n=cfg.max_single_cells, random_state=cfg.random_state)

    # Fetch raw tile image with original channels, and extract cell boundaries
    img_tile = get_tile_image(apply_display_settings=False)
    img_cell = img_tile[channels.index(cell_boundary_channel)].copy()

    # Fetch RGB version of tile image
    img_tile = get_tile_image(apply_display_settings=True)

    # Eliminate cell objects not in sample
    img_cell[~np.isin(img_cell, df['id'].values)] = 0

    logger.debug(
        'Single cell tile shape = %s (%s), cell boundary shape = %s (%s)',
        img_tile.shape, img_tile.dtype, img_cell.shape, img_cell.dtype
    )

    # Extract regions in RGB image corresponding to cell labelings
    cells = lib.extract_single_cell_images(
        img_cell, img_tile, is_boundary=True,
        patch_shape=cfg.cell_image_size,
        apply_mask=True, fill_value=0)

    return get_single_cells_title(len(cells), n_cells_in_tile) + [
        html.Img(
            title='Cell ID: {}'.format(c['id']),
            src='data:image/png;base64,{}'.format(lib.get_encoded_image(c['image']))
        )
        for c in cells
    ]


if __name__ == '__main__':
    app.run_server(debug=True, port=cfg.app_port, host=cfg.app_host_ip)
