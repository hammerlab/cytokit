import dash
import json
import fire
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
from codex_app.explorer import data as data
from codex_app.explorer import lib as lib
from codex_app.explorer.config import cfg
import pandas as pd
import logging
from skimage import io as sk_io
from skimage.transform import resize
logger = logging.getLogger(__name__)

data.initialize()

layouts = {}
app = dash.Dash()
app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/dZVMbK.css'})


def get_graph_channels():
    return 'ch:CD4', 'ch:CD8'


def get_tile_coords():
    """Current tile coordinates as (x, y)"""
    return 0, 0


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
        'margin': dict(l=0, t=25, r=0, b=0, pad=0),
        'title': 'Scatter',
        'titlefont': {'size': 12}
    }
    fig = dict(data=data, layout=layout)
    return dcc.Graph(
        id='graph',
        figure=fig,
        animate=True
    )


TILE_IMAGE_STYLE = dict(height='750px')
MONTAGE_IMAGE_STYLE = None

layouts['montage'] = lib.get_interactive_image_layout(data.get_montage_image())
layouts['tile'] = lib.get_interactive_image_layout(data.get_tile_image())


def get_channel_settings_layout(id, class_name):
    return html.Div(
        dcc.RangeSlider(
            id=id,
            min=0,
            max=20,
            step=0.5,
            value=[5, 15]
        ),
        className=class_name
    )


app.layout = html.Div([
        html.Div(
            className='row',
            children=[
                html.Div(get_graph(), className='five columns'),
                html.Div(
                    lib.get_interactive_image('montage', layouts['montage'], style=MONTAGE_IMAGE_STYLE),
                    className='five columns'
                ),
                get_channel_settings_layout('tile-channels', 'two columns')
            ],
        ),
        # html.Pre(id='console', className='four columns'),
        html.Div(
            className='row',
            children=[
                html.Div(
                    lib.get_interactive_image('tile', layouts['tile'], style=TILE_IMAGE_STYLE),
                    className='ten columns'
                ),
                get_channel_settings_layout('tile-channels', 'two columns')
            ]
        )
    ]
)

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
            'marker': {'opacity': .5},
            'type': 'scattergl'
        }]
    figure = dict(data=d, layout=layouts['montage'])
    return _relayout(figure, relayout_data)


# @app.callback(Output('tile', 'figure'), [Input('image', 'clickData')])
@app.callback(Output('tile', 'figure'), [Input('graph', 'selectedData')], [State('tile', 'relayoutData')])
def update_tile(selected_data, relayout_data):
    if selected_data is None:
        d = [{'x': [], 'y': [], 'mode': 'markers', 'type': 'scattergl'}]
    else:
        df = get_graph_data()

        # Fetch points in cyto data matching channel values in graph also
        # filtered to current tile
        tile_coords = get_tile_coords()
        df = df.set_index(['tile_x', 'tile_y', 'c1', 'c2']).loc[[
            tile_coords + (p['x'], p['y'])
            for p in selected_data['points']
        ]]

        d = [{
            'x': df['x'],
            'y': cfg.tile_shape[0] - df['y'],
            'mode': 'markers',
            'marker': {'opacity': .5},
            'type': 'scattergl'
        }]
    figure = dict(data=d, layout=layouts['tile'])
    return _relayout(figure, relayout_data)


if __name__ == '__main__':
    app.run_server(debug=True, port=cfg.app_port, host=cfg.app_host_ip)
