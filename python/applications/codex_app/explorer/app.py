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


def get_montage_image():
    return ac['processor']['montage'].run(data.get_montage_image())


def get_tile_image():
    return ac['processor']['tile'].run(data.get_tile_image(*ac['selection']['tile']['coords']))


def get_tile_nchannels():
    return len(data.get_tile_image_channels())


ac['shape']['montage'] = data.get_montage_image().shape
ac['processor']['montage'] = lib.ImageProcessor(ac['shape']['montage'][0])
ac['layouts']['montage'] = lib.get_interactive_image_layout(get_montage_image())

ac['shape']['tile'] = data.get_tile_image().shape
ac['processor']['tile'] = lib.ImageProcessor(ac['shape']['tile'][0])
ac['selection']['tile']['coords'] = (0, 0)
ac['layouts']['tile'] = lib.get_interactive_image_layout(get_tile_image())

ac['operation']['code'] = None


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
    if (ac['operation']['code'] or '').strip():
        n = len(df)
        logger.info('Applying custom code:\n%s', ac['operation']['code'])
        local_vars = {'df': df}
        exec(ac['operation']['code'], globals(), local_vars)
        df = local_vars['df']
        logger.info('Size of frame before custom operations = %s, after = %s', n, len(df))

    if len(df) > cfg.max_cells:
        logger.info('Sampling expression data for %s cells down to %s records', len(df), cfg.max_cells)
        df = df.sample(n=cfg.max_cells, random_state=cfg.random_state)
    return df


def apply_log(v):
    return np.log10(v)


def invert_log(v):
    return np.power(10, v)


def get_graph_data():
    selections = get_graph_axis_selections()

    if selections['xvar'] is None:
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
                marker={'opacity': .3},
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
    }
    fig = dict(data=fig_data, layout=fig_layout)
    return fig


def get_graph():
    return dcc.Graph(
        id='graph',
        figure=get_graph_figure(),
        animate=False
    )


def _ch_disp_name(ch):
    """Convert channel names like 'proc_Plasmid' to 'Plasmid'"""
    return '_'.join(ch.split('_')[1:])


def get_image_settings_layout(id_format, channel_names, class_name, type='tile'):

    ranges = data.db.get('app', type + '_channel_ranges')
    if ranges is None or len(ranges) != len(channel_names):
        channel_dtype = data.get_channel_dtype_map()
        assert len(channel_dtype) == len(channel_names), \
            'Channel data type map "{}" does not match length of channel list "{}"'\
            .format(channel_dtype, channel_names)
        ranges = [[0, np.iinfo(t).max] for t in channel_dtype.values()]

    colors = data.db.get('app', type + '_channel_colors')
    if colors is None or len(colors) != len(channel_names):
        colors = color.get_defaults(len(channel_names))

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
                    min=0,
                    max=ranges[i][1],
                    step=1,
                    value=ranges[i],
                    allowCross=False
                )
            ])
            for i in range(len(channel_names))
        ],
        className=class_name
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
            html.Div(axis.upper(), style={'width': '5%', 'display': 'inline-block'}),
            html.Div(
                dcc.Dropdown(
                    id='axis_' + axis + '_var',
                    options=[
                        {'label': v, 'value': k}
                        for k, v in field_names.items()
                    ],
                    value=var
                ),
                style={'width': '45%', 'display': 'inline-block'}
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
            ), style={'width': '45%', 'display': 'inline-block'}),

        ]
    )


def get_operation_code_layout():
    return [
        html.Div('Custom Operations:', style={'width': '100%'}),
        dcc.Textarea(
            id='operation-code',
            placeholder='Enter custom code',
            value='',
            style={'width': '90%', 'height': '100%'},
            wrap=False,
            rows=25
        ),
        html.Div('', id='code-message')
        # html.Button('Apply', id='apply-button')
    ]


app.layout = html.Div([
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
                html.Div(
                    lib.get_interactive_image('montage', ac['layouts']['montage'], style=MONTAGE_IMAGE_STYLE),
                    className='four columns'
                ),
                html.Div(get_operation_code_layout(), className='three columns'),
            ],
        ),
        # html.Pre(id='console', className='four columns'),
        html.Div(
            className='row',
            children=[
                html.Div(
                    lib.get_interactive_image('tile', ac['layouts']['tile'], style=TILE_IMAGE_STYLE),
                    className='ten columns'
                ),
                get_image_settings_layout('tile-ch-{}', data.get_tile_image_channels(), 'two columns')
            ]
        ),
        html.Div([
            html.Button('Save Settings', id='save-button'),
            html.Div('', id='message')
        ])
    ]
)


@app.callback(Output('code-message', 'children'), [Input('operation-code', 'value')])
def update_operation_code(code):
    ac['operation']['code'] = code
    return ''


@app.callback(
    Output('graph', 'figure'), [
        Input('axis_x_var', 'value'),
        Input('axis_x_scale', 'value'),
        Input('axis_y_var', 'value'),
        Input('axis_y_scale', 'value')
    ])
def update_graph(xvar, xscale, yvar, yscale):
    data.db.put('app', 'axis_x_var', xvar)
    data.db.put('app', 'axis_x_scale', xscale)
    data.db.put('app', 'axis_y_var', yvar)
    data.db.put('app', 'axis_y_scale', yscale)
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


def get_graph_data_selection(selected_data):
    type = selection_type(selected_data)
    if type is None:
        return None
    if type == 'lasso':
        # Fetch only cells/rows corresponding to selected data in graph (and index by tile loc)
        df = get_graph_data().iloc[[p['pointIndex'] for p in selected_data['points']]]
    elif type == 'range':
        df = get_graph_data()
        axis_selections = get_graph_axis_selections()
        var_range = selected_data['range']['x']
        if axis_selections['xscale'] == 'log':
            var_range = list(invert_log(np.array(var_range)))
        df = df[df[axis_selections['xvar']].between(*var_range)]
    else:
        raise ValueError('Selection type "{}" invalid'.format(type))

    # Sample if necessary
    if len(df) > cfg.max_montage_points:
        logger.info('Sampling montage data for %s cells down to %s records', len(df), cfg.max_montage_points)
        df = df.sample(n=cfg.max_montage_points, random_state=cfg.random_state)

    return df


@app.callback(Output('montage', 'figure'), [Input('graph', 'selectedData')], [State('montage', 'relayoutData')])
def update_montage(selected_data, relayout_data):
    df = get_graph_data_selection(selected_data)
    fig_data = [{'x': [], 'y': [], 'mode': 'markers', 'type': 'scattergl'}]
    if df is not None:
        # Update montage points using region x/y of cells
        x, y = _rescale_montage_coords(df['rx'], df['ry'])
        fig_data = [{
            'x': x,
            'y': cfg.montage_target_shape[0] - y,
            'mode': 'markers',
            'marker': {'opacity': .5, 'color': 'white'},
            'type': 'scattergl'
        }]
    figure = dict(data=fig_data, layout=ac['layouts']['montage'])
    return _relayout(figure, relayout_data)


def _get_tile_hover_text(r):
    return '<br>'.join(
        '{}: {:.2f}'.format(cfg.CYTO_FIELDS[f], r[f])
        for f in cfg.CYTO_HOVER_FIELDS if f in r
    )


def _get_tile_figure(selected_data, relayout_data):
    fig_data = [{'x': [], 'y': [], 'mode': 'markers', 'type': 'scattergl'}]
    df = get_graph_data_selection(selected_data)
    if df is not None:
        # Further restrict to only the selected tile
        df = df.set_index(['tile_x', 'tile_y']).sort_index()
        tx, ty = ac['selection']['tile']['coords']
        if (tx, ty) in df.index:
            df = df.loc[(tx, ty)]
            fig_data = [{
                'x': df['x'],
                'y': cfg.tile_shape[0] - df['y'],
                'mode': 'markers',
                'text': df.apply(_get_tile_hover_text, axis=1),
                'marker': {'opacity': 1., 'color': 'white'},
                'type': 'scattergl'
            }]
    figure = dict(data=fig_data, layout=ac['layouts']['tile'])
    return _relayout(figure, relayout_data)


# @app.callback(Output('tile', 'figure'), [Input('image', 'clickData')])
@app.callback(
    Output('tile', 'figure'),
    [Input('graph', 'selectedData'), Input('montage', 'clickData')] +
    [Input('tile-ch-{}-range'.format(i), 'value') for i in range(get_tile_nchannels())] +
    [Input('tile-ch-{}-color'.format(i), 'value') for i in range(get_tile_nchannels())]
    ,[
        State('tile', 'relayoutData')
    ])
def update_tile(*args):
    # montage click data:
    # {'points': [{'x': 114.73916, 'curveNumber': 0, 'pointNumber': 421, 'y': 306.4889, 'pointIndex': 421}]}
    selected_data, montage_data, relayout_data = args[0], args[1], args[-1]
    nch = get_tile_nchannels()
    channel_ranges = args[2:(nch + 2)]
    channel_colors = args[(nch + 2):-1]
    data.db.put('app', 'tile_channel_ranges', channel_ranges)
    data.db.put('app', 'tile_channel_colors', channel_colors)

    if montage_data:
        sy, sx = cfg.montage_target_shape
        ry, rx = cfg.region_shape
        py, px = montage_data['points'][0]['y'], montage_data['points'][0]['x']
        ac['selection']['tile']['coords'] = (int(px // (sx / rx)), ry - int(py // (sy / ry)) - 1)

    ac['processor']['tile'].ranges = channel_ranges
    ac['processor']['tile'].colors = [color.map(c) for c in channel_colors]

    ac['layouts']['tile'] = lib.get_interactive_image_layout(get_tile_image())
    return _get_tile_figure(selected_data, relayout_data)


if __name__ == '__main__':
    app.run_server(debug=True, port=cfg.app_port, host=cfg.app_host_ip)
