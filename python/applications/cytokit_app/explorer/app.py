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
from cytokit_app.explorer import data as data
from cytokit_app.explorer.data import cache as ac
from cytokit_app.explorer import lib as lib
from cytokit_app.explorer.config import cfg
from cytokit.cytometry.cytometer import DEFAULT_PREFIXES
from cytokit.image import color
from collections import OrderedDict
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


data.initialize()

app = dash.Dash()
app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/dZVMbK.css'})


TILE_IMAGE_STYLE = dict(height='750px')
MONTAGE_IMAGE_STYLE = None
TITLE_STYLE = {'textAlign': 'center', 'margin-top': '0px', 'margin-bottom': '0px'}
SUB_TITLE_STYLE = {**{'font-style': 'italic'}, **TITLE_STYLE}


def get_montage_image():
    return ac['processor']['montage'].run(data.get_montage_image())


def get_tile_image(apply_display_settings=True, coords=None):
    if coords is None:
        coords = ac['selection']['tile']['coords']
    img = data.get_tile_image(*coords)
    return ac['processor']['tile'].run(img) if apply_display_settings else img


def get_tile_nchannels():
    return len(data.get_tile_image_channels())


ac['shape']['montage'] = data.get_montage_image().shape
ac['processor']['montage'] = lib.ImageProcessor(
    n_channels=ac['shape']['montage'][0],
    ranges=data.get_montage_image_ranges(),
    colors=data.get_montage_image_colors()
)
ac['layouts']['montage'] = lib.get_interactive_image_layout(get_montage_image())

ac['shape']['tile'] = data.get_tile_image().shape
ac['processor']['tile'] = lib.ImageProcessor(ac['shape']['tile'][0])
ac['selection']['tile']['coords'] = (0, 0)
ac['layouts']['tile'] = lib.get_interactive_image_layout(get_tile_image())

ac['flag']['message'] = False
ac['flag']['clear_buffer'] = False


def get_graph_axis_selections():
    xscale = data.db.get('app', 'axis_x_scale') or 'linear'
    yscale = data.db.get('app', 'axis_y_scale') or 'linear'
    return dict(
        xvar=data.db.get('app', 'axis_x_var'),
        xscale=xscale,
        xtrans=lib.get_transform_by_name(xscale),
        yvar=data.db.get('app', 'axis_y_var'),
        yscale=yscale,
        ytrans=lib.get_transform_by_name(yscale),
    )


def get_base_data():
    # Pull cell expression data frame and downsample if necessary
    df = data.get_cytometry_stats()

    # Apply custom code if any has been passed in
    if data.db.exists('app', 'operation_code'):
        code = data.db.get('app', 'operation_code')
        n = len(df)
        logger.debug('Applying custom code:\n%s', code)
        local_vars = {'df': df}
        exec(code, globals(), local_vars)
        df = local_vars['df']
        logger.debug('Size of frame before custom operations = %s, after = %s', n, len(df))

    return df


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

    df = get_base_data().copy().filter(items=cols)
    for k, v in vars.items():
        df[k] = df[v]

    ac['counts']['graph_data']['n'] = len(df)

    return df


def get_graph_figure():
    d = get_graph_data()

    x = d['xvar'] if d is not None and 'xvar' in d else []
    y = d['yvar'] if d is not None and 'yvar' in d else []

    selections = get_graph_axis_selections()
    x = selections['xtrans'].apply(x)
    y = selections['ytrans'].apply(y)

    fig_layout = {
        'margin': dict(t=25),
        'title': '',
        'titlefont': {'size': 12},
        'xaxis': {
            'title': (selections['xvar'] or '').upper(),
            'showgrid': False,
            'showline': True,
            'zeroline': False
        },
        'yaxis': {
            'title': (selections['yvar'] or '').upper(),
            'showgrid': False,
            'showline': True,
            'zeroline': False
        },
        'dragmode': 'select'
    }
    if len(y) > 0:
        try:
            fig_data = lib.get_density_scatter_plot_data(
                x, y, cfg.max_kde_cells,
                opacity=cfg.graph_point_opacity,
                colorscale='Portland',
                size=3
            )
            fig_layout['showlegend'] = False
        except:
            logger.warning(
                'An error occurred computing KDE estimates for sample; '
                'falling back on countour histogram overlay'
            )
            fig_data = lib.get_density_overlay_plot_data(
                x, y, opacity=cfg.graph_point_opacity, color='white')
            fig_layout['showlegend'] = True
            fig_layout['legend'] = dict(
                orientation='h',
                font=dict(color='white'),
                bgcolor='#3070A6',
                bordercolor='#FFFFFF',
                borderwidth=2
            )
    else:
        fig_data = [
            dict(x=x, type='histogram', marker={'color': '#3070A6'})
        ]

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
        colors = color.get_colors(len(channel_names))

    def get_value_range(i):
        if values:
            return values[i]
        if channel_names[i].startswith(lib.CH_SRC_CYTO):
            return [0, 1]
        return ranges[i]

    return html.Div([
            html.Div([
                html.Div(
                    _ch_disp_name(channel_names[i]),
                    style={'width': '33%', 'display': 'inline-block'}
                ),
                html.Div(
                    ' - '.join([str(v) for v in get_value_range(i)]),
                    id=id_format.format(str(i) + '-range-label'),
                    style={'width': '33%', 'display': 'inline-block', 'text-align': 'center'}
                ),
                html.Div(dcc.Dropdown(
                    id=id_format.format(str(i) + '-color'),
                    options=[{'label': c.title(), 'value': c} for c in color.get_color_names()],
                    value=colors[i],
                    clearable=False,
                    searchable=False
                ), style={'width': '33%', 'display': 'inline-block'}),
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
    # include features for different cell components
    regex = '|'.join(DEFAULT_PREFIXES)
    for c in get_base_data().filter(regex=regex).columns.tolist():
        # Only add the raw field name if it is not in the pre-mapped field list
        if c not in field_names:
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
                    {'label': 'Linear', 'value': 'linear'},
                    {'label': 'Log', 'value': 'log10'},
                    {'label': 'Hyperbolic Sine', 'value': 'asinh'},
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
        html.Div('Custom Operations', style={'width': '100%', 'textAlign': 'center'}),
        dcc.Textarea(
            id='operation-code',
            placeholder='Enter custom pre-processing code',
            value=code,
            style={'width': '100%', 'height': '100%'},
            wrap=False,
            rows=25
        ),
        html.Button('Apply', id='apply-button', style={'width': '100%'})
    ]


##############
# App Layout #
##############


app.layout = html.Div([
        html.Div(
            className='row',
            children=html.Div([
                html.Div(
                    'Cytokit Explorer 1.0',
                    className='four columns',
                    style={
                        'color': 'white', 'padding-left': '15px',
                        'padding-top': '10px', 'font': '400 16px system-ui'
                    }
                ),
                html.Div(
                    'Experiment: {}'.format(cfg.exp_name),
                    className='four columns',
                    style={
                        'color': 'white', 'text-align': 'center',
                        'padding-top': '10px', 'font': '400 16px system-ui'
                    }
                ),
                html.Div(
                    html.Button(
                        'Save Settings', id='save-button',
                        style={'color': 'white', 'border-width': '0px', 'float': 'right'}
                    ),
                    className='four columns'
                )
            ]),
            style={'backgroundColor': '#3070A6'}
        ),
        html.Div([
                html.Div(
                    '', id='message',
                    style={'float': 'left', 'color': 'white', 'padding-top': '10px', 'padding-left': '15px'}
                ),
                html.Button('X', id='close-message-button', style={'float': 'right', 'backgroundColor': 'white'})
            ],
            id='message-container',
            className='row',
            style={'display': 'none', 'backgroundColor': 'rgb(44, 160, 44)'}
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
        html.Div(
            className='row',
            children=[
                html.Div(
                    html.Div([
                            dcc.Checklist(
                                options=[
                                    {'label': '', 'value': 'enabled'}
                                ],
                                values=[],
                                id='enable-buffer',
                                style={
                                    'display': 'inline-block', 'vertical-align': 'top',
                                    'margin-top': '5px', 'margin-right': '5px'
                                }
                            ),
                            html.H4(
                                'Selected Cell Buffer',
                                style={**TITLE_STYLE, **{'display': 'inline-block'}}
                            )
                        ],
                        style={'float': 'right'}
                    ),
                    className='six columns'
                ),
                html.Div([
                        html.Button('Clear', id='clear-buffer', style={'float': 'left', 'margin-left': '10px'}),
                        html.Button('Load All', id='load-buffer', style={'float': 'left', 'margin-left': '10px'})
                    ],
                    className='six columns'
                ),
                html.Div(id='single-cells-buffer', className='twelve columns')
            ]
        ),
        html.P(id='clear-buffer-state'),
        html.P(id='load-buffer-state')
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
    # Currently, only app-related properties are worth saving (but this may include
    # large images/datasets in the future)
    path = data.db.save(groups=['app'])
    ac['flag']['message'] = True
    return html.P('Application state saved to "{}"'.format(path))


@app.callback(
    Output('message-container', 'style'),
    [Input('message', 'children'), Input('close-message-button', 'n_clicks')])
def toggle_message(*_):
    if not ac['flag']['message']:
        return {'display': 'none'}
    ac['flag']['message'] = False
    return {'display': 'block', 'backgroundColor': 'rgb(44, 160, 44)'}


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
    vmin, vmax = var_range
    vmin = axis_selections[axis + 'trans'].invert(vmin)
    vmax = axis_selections[axis + 'trans'].invert(vmax)
    return df[df[axis_selections[axis + 'var']].between(vmin, vmax)]


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
            'marker': {'opacity': .5, 'color': cfg.montage_point_color},
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
                        shapes[-1]['fillcolor'] = cfg.montage_grid_color

            layout['shapes'] = shapes

    figure = dict(data=fig_data, layout=layout)
    return _relayout(figure, relayout_data)


def _get_tile_hover_text(r):
    fields = []
    for f in cfg.CYTO_HOVER_FIELDS:
        # Ignore any hover-text fields not present in data
        if f not in r:
            continue
        fmt = '{}: {:.3f}'
        if f in cfg.CYTO_INT_FIELDS:
            # Avoid integer formats as values maybe floats and string formatting
            # will fail when specified as int and given float
            fmt = '{}: {:.0f}'
        fields.append(fmt.format(cfg.CYTO_FIELDS[f], r[f]))
    return '<br>'.join(fields)


def get_tile_coordinates_in_selection():
    """Return list of all tile x, y tuples currently present in selected cell data"""
    df = get_graph_data_selection()
    if df is None:
        return None
    return list(df.set_index(['tile_x', 'tile_y']).index.unique())


def get_tile_graph_data_selection(coords=None):
    df = get_graph_data_selection()
    if df is None:
        return None
    if coords is None:
        coords = ac['selection']['tile']['coords']
    tx, ty = coords

    # Further restrict selected data to only the requested tile
    df = df.set_index(['tile_x', 'tile_y']).sort_index()
    if (tx, ty) not in df.index:
        return None
    # Make sure to use list of tuples for slice to avoid series result with single matches
    return df.loc[[(tx, ty)]]


def _get_tile_figure_data(df):
    points_only = cfg.cell_marker_mode == 'point'

    # Create a trace with points at centroid of cells in tile image
    fig_data = [{
        'x': df['x'],
        'y': cfg.tile_shape[0] - df['y'],
        'mode': 'markers',
        'text': df.apply(_get_tile_hover_text, axis=1),
        'marker': {
            'opacity': 1 if points_only else 0,
            'color': cfg.cell_marker_point_color if points_only else cfg.cell_marker_mask_color,
            'size': cfg.cell_marker_point_size
        },
        'type': 'scattergl',
        'hoverinfo': 'text'
    }]
    if points_only:
        return fig_data, None

    # If configured to mark cells with masks instead of centroids, extract cell masks
    # from the raw labeled images and convert them to SVG paths
    cell_data, _ = get_single_cell_data(df=df, max_cells=np.inf, object_type=cfg.cell_marker_mask_object + '_boundary')
    shapes = []
    for cell in cell_data:
        prop = cell['properties']
        if prop.area > 0:
            # Sort coordinates counter-clockwise
            cell_coords = lib.get_sorted_boundary_coords(prop)

            # Coords in properties are (row, col) as they come from skimage.measure.regionprops
            # but SVG paths are expected in xy format
            cell_coords = ['{},{}'.format(r[1], cfg.tile_shape[0] - r[0]) for r in cell_coords]

            shapes.append({
                'type': 'path',
                # Path example: M 3,7 L2,8 L2,9 L5,9 L5,8 L4,7 Z (from https://plot.ly/python/shapes/)
                'path': 'M {} {} Z'.format(cell_coords[0], ' '.join('L' + v for v in cell_coords[1:])),
                'fillcolor': cfg.cell_marker_mask_fill,
                'line': {'color': cfg.cell_marker_mask_color}
            })
    return fig_data, shapes


def _get_tile_figure(relayout_data):
    fig_data = [{'x': [], 'y': [], 'mode': 'markers', 'type': 'scattergl'}]
    df = get_tile_graph_data_selection()

    ac['counts']['tile']['n'] = 0
    ac['layouts']['tile']['shapes'] = []
    if df is not None:
        # Downsample if necessary
        if len(df) > cfg.max_tile_cells:
            logger.info('Sampling tile expression data for %s cells down to %s records', len(df), cfg.max_tile_cells)
            df = df.sample(n=cfg.max_tile_cells, random_state=cfg.random_state)
        ac['counts']['tile']['n'] = len(df)

        tx, ty = ac['selection']['tile']['coords']
        logger.info('Number of cells found in tile (x=%s, y=%s): %s', tx + 1, ty + 1, len(df))

        fig_data, shapes = _get_tile_figure_data(df)
        if shapes:
            ac['layouts']['tile']['shapes'] = shapes

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


# Channel slider range label callbacks
for i in range(get_tile_nchannels()):
    @app.callback(
        Output('tile-ch-{}-range-label'.format(i), 'children'),
        [Input('tile-ch-{}-range'.format(i), 'value')]
    )
    def update_channel_range_label(val):
        if val is None:
            return None
        return ' - '.join([str(v) for v in val])


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


#######################
# Single Cell Functions
#######################

def get_single_cells_title(n_cells=0, n_tile=0):
    children = [html.H4('Selected Cells', style=TITLE_STYLE)]

    # Append count info to title/header only if it is informative
    if n_cells:
        children.append(html.P(
            'Showing {} cells of {} selected in tile'.format(n_cells, n_tile),
            style=SUB_TITLE_STYLE
        ))

    return children


@app.callback(
    Output('clear-buffer-state', 'children'),
    [Input('clear-buffer', 'n_clicks')]
)
def update_clear_buffer_state(n_clicks):
    if n_clicks:
        ac['flag']['clear_buffer'] = True
    return None


@app.callback(
    Output('load-buffer-state', 'children'),
    [Input('load-buffer', 'n_clicks')]
)
def update_load_buffer_state(n_clicks):
    if n_clicks:
        ac['flag']['load_buffer'] = True
    return None


def create_cell_image(cell):
    style = {'padding': '1px'}
    if cfg.cell_image_display_width is not None:
        style['width'] = '{:.0f}px'.format(cfg.cell_image_display_width)
    return html.Img(
        title='Cell ID: {}'.format(cell['id']),
        src='data:image/png;base64,{}'.format(lib.get_encoded_image(cell['image'])),
        style=style
    )


def get_single_cell_data(df=None, coords=None, max_cells=None, **kwargs):
    if df is None:
        df = get_tile_graph_data_selection(coords=coords)
    if df is None:
        return None, None

    # Downsample if necessary
    n_cells_in_tile = len(df)
    max_cells = max_cells or cfg.max_single_cells
    if len(df) > max_cells:
        logger.info(
            'Sampling single cell data for %s cells down to %s records %s',
            len(df), max_cells,
            '' if coords is None else '[tile = ' + str(coords) + ']'
        )
        df = df.sample(n=max_cells, random_state=cfg.random_state)

    raw_tile = get_tile_image(apply_display_settings=False, coords=coords)
    display_tile = get_tile_image(apply_display_settings=True, coords=coords)
    channels = data.get_tile_image_channels()
    cell_data = lib.get_single_cell_data(
        df, raw_tile, display_tile, channels,
        cell_image_size=cfg.cell_image_size,
        **kwargs
    )
    return cell_data, n_cells_in_tile


@app.callback(
    Output('single-cells-buffer', 'children'),
    [
        Input('single-cells', 'children'),
        Input('clear-buffer-state', 'children'),
        Input('load-buffer-state', 'children')
    ],
    [
        State('single-cells-buffer', 'children'),
        State('enable-buffer', 'values')
    ]
)
def update_single_cell_buffer(new_children, _1, _2, current_children, enabled):
    if not enabled:
        return []

    if ac['flag']['clear_buffer']:
        ac['flag']['clear_buffer'] = False
        return []
    res = (current_children or [])

    # If load buffer flag was set, ignore cells for selected tile and instead load all cells
    # across entire experiment
    if ac['flag']['load_buffer']:
        ac['flag']['load_buffer'] = False
        tile_coords = get_tile_coordinates_in_selection()
        for i, coords in enumerate(tile_coords):
            logger.info('Extracting cells (for buffer) from tile %s (%s of %s)', coords, i+1, len(tile_coords))
            cell_data, _ = get_single_cell_data(coords=coords, apply_mask=not cfg.cell_image_background_enabled)
            if cell_data is not None:
                res.extend([create_cell_image(cell) for cell in cell_data])
        return res

    # Otherwise, append new entries from tile just selected
    if new_children is not None:
        for c in new_children:
            if c['type'] == 'Div' and 'id' in c['props'] and c['props']['id'] == 'single-cell-images':
                res.extend(c['props']['children'])
    return res


@app.callback(
    Output('single-cells', 'children'),
    [Input('tile', 'figure')]
)
def update_single_cells(_):
    cell_data, n_cells_in_tile = get_single_cell_data(apply_mask=not cfg.cell_image_background_enabled)
    if cell_data is None:
        return get_single_cells_title()

    images = [create_cell_image(cell) for cell in cell_data]
    return get_single_cells_title(len(cell_data), n_cells_in_tile) + [
        html.Div(
            images,
            id='single-cell-images',
            style={'line-height': '.5', 'overflowY': 'scroll', 'max-height': '700px'}
        )
    ]


def run():
    app.run_server(debug=cfg.debug, port=cfg.app_port, host=cfg.app_host_ip)

if __name__ == '__main__':
    run()
