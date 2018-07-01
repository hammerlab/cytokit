import dash
import json
import fire
from dash.dependencies import Input, Output
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


def get_plot_data():
    x, y = 'ch:CD4', 'ch:CD8'
    df = data.get_cytometry_stats()
    df = df[['x', 'y', x, y]].copy()
    df.columns = ['x', 'y', 'c1', 'c2']
    return df


def get_expr_plot():
    d = get_plot_data()
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


layouts['montage'] = lib.get_interactive_image_layout(data.get_montage_image())
layouts['tile'] = lib.get_interactive_image_layout(data.get_tile_image())
app.layout = html.Div([
    html.Div(
        className='row',
        children=[
            html.Div(get_expr_plot(), className='four columns'),
            html.Div(lib.interactive_image('montage', layouts['montage']), className='eight columns')
        ],
        # style={'width': '2000px'}
    ),
    # html.Pre(id='console', className='four columns'),
    html.Div(dcc.Graph(id='tile', figure=dict(data=[])), className='twelve columns')
])

# display the event data for debugging
# @app.callback(Output('console', 'children'), [Input('graph', 'selectedData')])
# def display_selected_data(selectedData):
#     return json.dumps(selectedData if selectedData else {}, indent=4)


@app.callback(Output('montage', 'figure'), [Input('graph', 'selectedData')])
def update_montage(selectedData):
    # print(type(selectedData))
    if selectedData is None:
        d = [{'x': [], 'y': [], 'mode': 'markers', 'type': 'scattergl'}]
    else:
        df = get_plot_data().set_index(['c1', 'c2']).loc[[
            (p['x'], p['y'])
            for p in selectedData['points']
        ]]
        d = [{
            'x': df['x'],
            'y': data.get_montage_image().shape[0] - df['y'],
            'mode': 'markers',
            'marker': {'opacity': .5},
            'type': 'scattergl'
        }]
    return dict(data=d, layout=layouts['montage'])


# @app.callback(Output('tile', 'figure'), [Input('image', 'clickData')])
@app.callback(Output('tile', 'figure'), [Input('graph', 'selectedData')])
def update_tile(selectedData):
    if selectedData is None:
        d = [{'x': [], 'y': [], 'mode': 'markers', 'type': 'scattergl'}]
    else:
        df = get_plot_data()
        df = df.set_index(['c1', 'c2']).loc[[
            (p['x'], p['y'])
            for p in selectedData['points']
        ]]
        print('[tile] Subset size = ', len(df))

        d = [{
            'x': df['x'],
            'y': data.get_tile_image().shape[0] - df['y'],
            'mode': 'markers',
            'marker': {'opacity': .5},
            'type': 'scattergl'
        }]
    return dict(data=d, layout=layouts['tile'])


if __name__ == '__main__':
    app.run_server(debug=True, port=cfg.app_port, host=cfg.app_host_ip)
