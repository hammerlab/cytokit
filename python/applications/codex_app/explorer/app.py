import dash
import json
import fire
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import codex_app.explorer.data as data
import codex_app.explorer.lib as lib
import pandas as pd
import logging
from skimage import io as sk_io
from skimage.transform import resize
logger = logging.getLogger(__name__)

layouts = {}
app = dash.Dash()
app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/dZVMbK.css'})


def get_plot_data():
    x, y = 'CD4', 'CD8'
    df = data.cytometry_stats()
    df = df[[x, y]].copy()
    df.columns = ['c1', 'c2']
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
        'title': 'Marker Expression Decomposition<br>Markers: {}'.format(markers),
        'titlefont': {'size': 12}
    }
    fig = dict(data=data, layout=layout)
    return dcc.Graph(
        id='graph',
        figure=fig,
        animate=True
    )


# display the event data for debugging
# @app.callback(Output('console', 'children'), [Input('graph', 'selectedData')])
# def display_selected_data(selectedData):
#     return json.dumps(selectedData if selectedData else {}, indent=4)


# @app.callback(Output('image', 'figure'), [Input('graph', 'selectedData')])
# def update_histogram(selectedData):
#     print('In selected data:')
#     print(type(selectedData))
#     print(type(img_layout))
#     if selectedData is None:
#         data = [{'x': [], 'y': [], 'mode': 'markers', 'type': 'scattergl'}]
#     else:
#         df = df_exp.set_index(['c1', 'c2']).loc[[
#             (p['x'], p['y'])
#             for p in selectedData['points']
#         ]]
#         print('Subset size = ', len(df))
#         data = [{
#             'x': SCALE_FACTOR * df['X'],
#             'y': img.shape[0] - SCALE_FACTOR * df['Y'],
#             'mode': 'markers',
#             'marker': {'opacity': .5},
#             'type': 'scattergl'
#         }]
#     return dict(data=data, layout=img_layout)


# @app.callback(Output('tile', 'figure'), [Input('image', 'clickData')])
@app.callback(Output('tile', 'figure'), [Input('graph', 'selectedData')])
def update_tile(selectedData):
    if selectedData is None:
        data = [{'x': [], 'y': [], 'mode': 'markers', 'type': 'scattergl'}]
    else:
        d = get_plot_data()
        d = d.set_index(['c1', 'c2']).loc[[
            (p['x'], p['y'])
            for p in selectedData['points']
        ]]
        print('[tile] Subset size = ', len(df))

        data = [{
            'x': d['x'],
            'y': tile_img.shape[0] - df['y'],
            'mode': 'markers',
            'marker': {'opacity': .5},
            'type': 'scattergl'
        }]
    return dict(data=data, layout=layouts['tile'])


def run(exp_config_path, exp_data_dir, app_data_dir=None, port=8050):
    data.initialize(exp_config_path, exp_data_dir, app_data_dir=app_data_dir)

    layouts['montage'] = app_lib.get_interactive_image_layout(data.montage_image())
    layouts['tile'] = app_lib.get_interactive_image_layout(data.get_tile_image())
    app.layout = html.Div([
        html.Div(
            className='row',
            children=[
                html.Div(get_expr_plot(), className='four columns'),
                html.Div(app_lib.interactive_image('montage', layouts['montage']), className='eight columns')
            ],
            # style={'width': '2000px'}
        ),
        # html.Pre(id='console', className='four columns'),
        html.Div(dcc.Graph(id='tile', figure=dict(data=[])), className='twelve columns')
    ])

    app.run_server(debug=True, port=port)


if __name__ == '__main__':
    fire.Fire(run)
