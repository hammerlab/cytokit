import dash
import json
import fire
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
from codex_app.explorer import data as data
from codex_app.explorer.data import cache as ac
from codex_app.explorer import lib as lib
from codex_app.explorer.config import cfg
from codex_app.explorer import color
import pandas as pd
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


def get_graph_channels():
    return 'ch:CD4', 'ch:CD8'


def get_graph_data():
    c1, c2 = get_graph_channels()
    df = data.get_cytometry_stats()
    df = df[['region_index', 'tile_x', 'tile_y', 'id', 'rid', 'rx', 'ry', 'x', 'y', c1, c2]].copy()
    df = df.rename(columns={c1: 'c1', c2: 'c2'})
    return df


def get_graph():
    d = get_graph_data()
    data = [
        dict(
            x=d['c1'],
            y=d['c2'],
            mode='markers',
            marker={'opacity': .1},
            type='scattergl'
        )
    ]
    layout = {
        #'margin': dict(l=0, t=25, r=0, b=0, pad=0),
        'title': '',
        'titlefont': {'size': 12}
    }
    fig = dict(data=data, layout=layout)
    return dcc.Graph(
        id='graph',
        figure=fig,
        animate=True
    )


def get_image_settings_layout(id_format, channel_names, class_name, type='tile'):

    # Convert channel names like "cyto_nucleus_boundary" to "nucleus boundary"
    def ch_disp_name(ch):
        return ' '.join(ch.split('_')[1:]).title()

    ranges = data.db.get('app', type + '_channel_ranges')
    if ranges is None or len(ranges) != len(channel_names):
        ranges = [[0, 255]] * len(channel_names)

    colors = data.db.get('app', type + '_channel_colors')
    if colors is None or len(colors) != len(channel_names):
        colors = color.get_defaults(len(channel_names))

    return html.Div([
            html.Div([
                html.Div(ch_disp_name(channel_names[i]), style={'width': '50%', 'display': 'inline-block'}),
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
                    max=255,
                    step=1,
                    value=ranges[i],
                    allowCross=False
                    #marks={0: 'Cells'}
                )
            ])
            for i in range(len(channel_names))
        ],
        className=class_name
    )


app.layout = html.Div([
        html.Div(
            className='row',
            children=[
                html.Div([
                        dcc.Dropdown(
                            id='data-dropdown',
                            options=[
                                {'label': 'Selection', 'value': 'selection'},
                                {'label': 'Expression', 'value': 'expression'}
                            ],
                            value='selection'
                        ),
                        html.Div(
                            get_graph(),
                            id='data-output'
                        )
                    ],
                    className='five columns'
                ),
                html.Div(
                    lib.get_interactive_image('montage', ac['layouts']['montage'], style=MONTAGE_IMAGE_STYLE),
                    className='five columns'
                ),
                get_image_settings_layout('montage-ch-{}', cfg.montage_channels, 'two columns')
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
                get_image_settings_layout('tile-ch-{}', cfg.montage_channels, 'two columns')
            ]
        ),
        html.Div([
            html.Button('Save Settings', id='save-button'),
            html.Div('', id='message')
        ])
    ]
)


@app.callback(Output('data-output', 'children'), [Input('data-dropdown', 'value')])
def update_output(value):
    if value == 'selection':
        return get_graph()
    elif value == 'expression':
        return html.Div('Expression')
    return None


@app.callback(Output('message', 'children'), [Input('save-button', 'n_clicks')])
def save_state(n_clicks):
    if n_clicks is None:
        return ''
    path = data.db.save()
    return 'Application state saved to "{}"'.format(path)


# display the event data for debugging
# @app.callback(Output('console', 'children'), [Input('graph', 'selectedData')])
# def display_selected_data(selectedData):
#     return json.dumps(selectedData if selectedData else {}, indent=4)


def _rescale_montage_coords(x, y):
    sy, sx = cfg.montage_target_scale_factors
    return x * sx, y * sy


def _relayout(figure, relayout_data):
    if relayout_data:
        print(relayout_data)
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


@app.callback(Output('montage', 'figure'), [Input('graph', 'selectedData')], [State('montage', 'relayoutData')])
def update_montage(selected_data, relayout_data):
    # print(type(selectedData))
    if selected_data is None:
        d = [{'x': [], 'y': [], 'mode': 'markers', 'type': 'scattergl'}]
    else:
        # Fetch data corresponding to selected points
        df = get_graph_data().set_index(['c1', 'c2']).loc[[
            (p['x'], p['y'])
            for p in selected_data['points']
        ]]

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


def _get_tile_figure(selected_data, relayout_data):
    if selected_data is None:
        d = [{'x': [], 'y': [], 'mode': 'markers', 'type': 'scattergl'}]
    else:
        df = get_graph_data()

        # Fetch points in cyto data matching channel values in graph also
        # filtered to current tile
        tx, ty = ac['selection']['tile']['coords']
        df = df.set_index(['tile_x', 'tile_y', 'c1', 'c2']).loc[[
            (tx, ty) + (p['x'], p['y'])
            for p in selected_data['points']
        ]]

        d = [{
            'x': df['x'],
            'y': cfg.tile_shape[0] - df['y'],
            'mode': 'markers',
            'marker': {'opacity': 1., 'color': 'white'},
            'type': 'scattergl'
        }]
    figure = dict(data=d, layout=ac['layouts']['tile'])
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

    print('Channel ranges: ', channel_ranges)
    print('Channel colors: ', channel_colors)
    print('New tile coords: ', ac['selection']['tile']['coords'])
    ac['processor']['tile'].ranges = channel_ranges
    ac['processor']['tile'].colors = [color.map(c) for c in channel_colors]

    ac['layouts']['tile'] = lib.get_interactive_image_layout(get_tile_image())
    return _get_tile_figure(selected_data, relayout_data)


if __name__ == '__main__':
    app.run_server(debug=True, port=cfg.app_port, host=cfg.app_host_ip)
