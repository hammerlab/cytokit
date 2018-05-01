import dash
import json
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import codex_app.decompose.data as app_data
import codex_app.decompose.lib as app_lib
import logging
from skimage.transform import resize
logger = logging.getLogger(__name__)

SCALE_FACTOR = .3

#DATA_DIR = '/Users/eczech/data/research/hammer/smshare/experiment_data/7-7-17-multicycle/out-microv'
DATA_DIR = '/Users/eczech/data/research/hammer/smshare/experiment_data/20180426_D18_R1/views/v00-all/output'
app_data.set_data_dir(DATA_DIR)

app = dash.Dash()
app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/dZVMbK.css'})


exp_config = app_data.get_experiment_config()
markers = ['HOECHST1', 'CollagenIV', 'CD7', 'Ki67', 'CD4', 'CD3', 'CD8', 'CD38']

logger.info('Loading image tile')
tile_img = app_data.get_tile_image(0, 3, 6)
tile_img = tile_img[0, 0, 0]

from skimage import io as sk_io
#img = sk_io.imread('/Users/eczech/tmp/codex/decompose/cache/montage.tif')
img = sk_io.imread('/Users/eczech/data/research/hammer/smshare/experiment_data/20180426_D18_R1/views/v00-all/output/montage.tif')
#img = img[:1000, :1000]
img = resize(img, [int(s * SCALE_FACTOR) for s in img.shape])

logger.info('Loading expression file')
import pandas as pd
#df_exp = pd.read_csv('/Users/eczech/tmp/codex/decompose/cache/expression.csv', sep='\t').sample(10000)
df_exp = pd.read_csv('/Users/eczech/tmp/codex/decompose/cache/expression_20180426_D18_R1.csv', sep='\t').sample(50000, random_state=1)
print(df_exp.head())


def get_expr_plot(d):
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

img_layout = app_lib.get_interactive_image_layout(img)
tile_layout = app_lib.get_interactive_image_layout(tile_img)

app.layout = html.Div([
    html.Div(
        className='row',
        children=[
            html.Div(get_expr_plot(df_exp), className='four columns'),
            html.Div(app_lib.interactive_image('image', img_layout), className='eight columns')
        ],
        #style={'width': '2000px'}
    ),
    #html.Pre(id='console', className='four columns'),
    html.Div(dcc.Graph(id='tile', figure=dict(data=[])), className='twelve columns')
])


# display the event data for debugging
# @app.callback(Output('console', 'children'), [Input('graph', 'selectedData')])
# def display_selected_data(selectedData):
#     return json.dumps(selectedData if selectedData else {}, indent=4)


@app.callback(Output('image', 'figure'), [Input('graph', 'selectedData')])
def update_histogram(selectedData):
    print('In selected data:')
    print(type(selectedData))
    print(type(img_layout))
    if selectedData is None:
        data = [{'x': [], 'y': [], 'mode': 'markers', 'type': 'scattergl'}]
    else:
        df = df_exp.set_index(['c1', 'c2']).loc[[
            (p['x'], p['y'])
            for p in selectedData['points']
        ]]
        print('Subset size = ', len(df))
        data = [{
            'x': SCALE_FACTOR * df['X'],
            'y': img.shape[0] - SCALE_FACTOR * df['Y'],
            'mode': 'markers',
            'marker': {'opacity': .5},
            'type': 'scattergl'
        }]
    return dict(data=data, layout=img_layout)


# @app.callback(Output('tile', 'figure'), [Input('image', 'clickData')])
@app.callback(Output('tile', 'figure'), [Input('graph', 'selectedData')])
def update_tile(selectedData):
    if selectedData is None:
        data = [{'x': [], 'y': [], 'mode': 'markers', 'type': 'scattergl'}]
    else:
        #x, y = p['x'], p['y']
        #tx, ty = x // exp_config.tile_width, y // exp_config.tile_height
        tx, ty = 3, 6
        df = df_exp.set_index(['c1', 'c2']).loc[[
            (p['x'], p['y'])
            for p in selectedData['points']
        ]].copy()
        tw, th = exp_config.tile_width, exp_config.tile_height
        df = df[df['X'].between(tx * tw, (tx + 1) * tw)]
        df = df[df['Y'].between(ty * th, (ty + 1) * th)]
        df['X'] -= tx * tw
        df['Y'] -= ty * th
        print('[tile] Subset size = ', len(df))

        data = [{
            'x': df['X'],
            'y': tile_img.shape[0] - df['Y'],
            'mode': 'markers',
            'marker': {'opacity': .5},
            'type': 'scattergl'
        }]
    return dict(data=data, layout=tile_layout)


if __name__ == '__main__':
    app.run_server(debug=True)