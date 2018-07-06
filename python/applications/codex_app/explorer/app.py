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
    if len(df) > cfg.max_cells:
        logger.info('Sampling expression data for %s cells down to %s records', len(df), cfg.max_cells)
        df = df.sample(n=cfg.max_cells, random_state=cfg.random_state)
    return df


def get_graph_data():
    selections = get_graph_axis_selections()

    # Create list of fields to extract that are always necessary
    cols = [
        'cell_diameter', 'nucleus_diameter', 'cell_size', 'nucleus_size', 'nucleus_solidity',
        'region_index', 'tile_x', 'tile_y', 'id', 'rid', 'rx', 'ry', 'x', 'y'
    ]
    # cols = [
    #     'cell_size', 'nucleus_size', 'nucleus_solidity',
    #     'region_index', 'tile_x', 'tile_y', 'id', 'rid', 'rx', 'ry', 'x', 'y'
    # ]
    if selections['xvar'] is None:
        return None

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

    # Graphable data should only include fields directly referable by name, which
    # means they should all be lower case with no prefix or grouping modifiers
    df = get_base_data().copy()
    df = strip_field_modifiers(df)
    df = normalize_field_names(df)
    df = df[cols]
    for k, v in vars.items():
        df[k] = df[v]

    if 'xvar' in df and selections['xscale'] == 'log':
        df['xvar'] = np.log10(df['xvar'])
    if 'yvar' in df and selections['yscale'] == 'log':
        df['yvar'] = np.log10(df['yvar'])

    return df


def strip_field_modifiers(df):
    return df.rename(columns=lambda c: c.replace('ch:', ''))


def normalize_field_name(field):
    return field.lower().replace(' ', '_')


def normalize_field_names(df):
    return df.rename(columns=normalize_field_name)


def normalize_channel_name(ch):
    """Convert channel names like 'proc_DAPI 1' to 'dapi_1'"""
    return ' '.join(ch.split('_')[1:]).replace(' ', '_').lower()


def _get_box_data(df):

    if len(df) > cfg.max_boxplot_records:
        logger.info('Sampling boxplot data for %s cells down to %s records', len(df), cfg.max_boxplot_records)
        df = df.sample(n=cfg.max_boxplot_records, random_state=cfg.random_state)

    vals = []
    labels = []
    for c in df:
        vals.extend(list(df[c].values))
        labels.extend([c] * len(df))
    return vals, labels


def get_distribution_figure(selected_points=None):
    # Extract expression stats only
    d = get_base_data().filter(regex='ch:')

    # Convert all names to snake case
    d = normalize_field_names(strip_field_modifiers(d))

    # Select only the expression fields present in the extract
    d = d.filter(items=[normalize_channel_name(c) for c in cfg.extract_channels])

    fig_data = []

    vals, labels = _get_box_data(d)
    fig_data.append(dict(
        x=vals,
        y=labels,
        name='Overall',
        orientation='h',
        boxpoints=False,
        type='box'
    ))

    if selected_points is not None:
        ds = d.iloc[selected_points]
        vals, labels = _get_box_data(ds)
        fig_data.append(dict(
            x=vals,
            y=labels,
            name='Selected',
            orientation='h',
            boxpoints=False,
            type='box'
        ))

    fig_layout = {
        'boxmode': 'group',
        'title': 'Expression Distribution',
        'margin': dict(b=0, t=50)
    }
    fig = dict(data=fig_data, layout=fig_layout)
    return fig


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


def get_distribution():
    return dcc.Graph(
        id='distribution',
        figure=get_distribution_figure(),
        animate=False
    )


def _ch_disp_name(ch):
    """Convert channel names like 'cyto_nucleus_boundary' to 'nucleus boundary'"""
    return ' '.join(ch.split('_')[1:]).lower()


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

    # Get list of possible field values based on "base" dataset, which may
    # include spatial coordinates, raw expression levels, and projections
    field_names = strip_field_modifiers(get_base_data().filter(regex='ch:')).columns.tolist()
    field_names += [
        'Nucleus Diameter', 'Nucleus Size', 'Nucleus Solidity',
        'Cell Diameter', 'Cell Size', 'Cell Solidity',
        'RX', 'RY', 'X', 'Y', 'Z'
    ]

    return html.Div([
            html.Div(axis.upper(), style={'width': '5%', 'display': 'inline-block'}),
            html.Div(
                dcc.Dropdown(
                    id='axis_' + axis + '_var',
                    options=[
                        # Note that the "value" should always be lower case since this
                        # is what is passed around for referring to fields
                        {'label': f, 'value': normalize_field_name(f)}
                        for f in field_names
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
                html.Div(
                    get_distribution(),
                    className='three columns'
                )
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
                get_image_settings_layout('tile-ch-{}', cfg.extract_channels, 'two columns')
            ]
        ),
        html.Div([
            html.Button('Save Settings', id='save-button'),
            html.Div('', id='message')
        ])
    ]
)


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


@app.callback(Output('distribution', 'figure'), [Input('graph', 'selectedData')])
def update_distribution(selected_data):
    selected_points = None
    if selected_data is not None:
        selected_points = [p['pointIndex'] for p in selected_data['points']]
    return get_distribution_figure(selected_points)


@app.callback(Output('montage', 'figure'), [Input('graph', 'selectedData')], [State('montage', 'relayoutData')])
def update_montage(selected_data, relayout_data):
    if selected_data is None:
        d = [{'x': [], 'y': [], 'mode': 'markers', 'type': 'scattergl'}]
    else:
        # Fetch data corresponding to selected points
        df = get_graph_data().iloc[[p['pointIndex'] for p in selected_data['points']]]

        # Sample if necessary
        if len(df) > cfg.max_montage_points:
            logger.info('Sampling montage data for %s cells down to %s records', len(df), cfg.max_montage_points)
            df = df.sample(n=cfg.max_montage_points, random_state=cfg.random_state)

        # Update montage points using region x/y of cells
        x, y = _rescale_montage_coords(df['rx'], df['ry'])
        d = [{
            'x': x,
            'y': cfg.montage_target_shape[0] - y,
            'mode': 'markers',
            'marker': {'opacity': .5, 'color': 'white'},
            'type': 'scattergl'
        }]
    figure = dict(data=d, layout=ac['layouts']['montage'])
    return _relayout(figure, relayout_data)


def _get_tile_hover_text(r):
    fields = ['Nucleus Diameter', 'Nucleus Solidity', 'Cell Diameter', 'Cell Size']
    return '<br>'.join(
        '{}: {:.2f}'.format(f, r[normalize_field_name(f)])
        for f in fields
        if normalize_field_name(f) in r
    )

def _get_tile_figure(selected_data, relayout_data):
    fig_data = [{'x': [], 'y': [], 'mode': 'markers', 'type': 'scattergl'}]
    if selected_data is not None:
        df = get_graph_data()

        # Fetch only cells/rows corresponding to selected data in graph (and index by tile loc)
        df = df.iloc[[p['pointIndex'] for p in selected_data['points']]] \
            .set_index(['tile_x', 'tile_y']).sort_index()

        # Further restrict to only the selected tile
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
    [Input('tile-ch-{}-range'.format(i), 'value') for i in range(cfg.extract_nchannels)] +
    [Input('tile-ch-{}-color'.format(i), 'value') for i in range(cfg.extract_nchannels)]
    ,[
        State('tile', 'relayoutData')
    ])
def update_tile(*args):
    # montage click data:
    # {'points': [{'x': 114.73916, 'curveNumber': 0, 'pointNumber': 421, 'y': 306.4889, 'pointIndex': 421}]}
    selected_data, montage_data, relayout_data = args[0], args[1], args[-1]
    nch = cfg.extract_nchannels
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
