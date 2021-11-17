
import os
import dash
import dash_player
import dash_table
import dash_uploader as du
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State

import flask
from textwrap import dedent

from .text.general_text import markdown_text_disclaimer
from .dash_style_defs import upload_default_style
from .dash_style_defs import data_table_style_cell_conditional
from .dash_style_defs import data_table_style_data_conditional
from .dash_style_defs import data_table_header_style, data_table_style_cell
from .utils import build_df, set_results_dict, update_data_folder_tree, bar_chart
from .definitions import REPO_DIR, UPLOAD_DIR, STATIC_DIRNAME, STATIC_FULLPATH, FF_URL
from .api_calls import UploadFile, GetModelList, SubmitInferenceRequest, BuildInferenceRequest

from app import app, server


# Define empty string as default
empty_string = ''

# Debug flag for if no server connectivity
debug = False
#debug = True


# Get available model list from API
api_model_dict = GetModelList(url=FF_URL, debug=debug)
avail_model_list = api_model_dict['models']


# Server route for local video playing by dash-player
@server.route('{}/<path:path>'.format(STATIC_FULLPATH))
def serve_static(path):
    return flask.send_from_directory(STATIC_FULLPATH, path)


# Setup initial results dataframe 
init_results_df = build_df(model_list=avail_model_list,
                           results_dict={})


# Setup initial dropdown & video options
init_file_list = update_data_folder_tree()
init_dropdown_options=[{'label':os.path.basename(ifile), 
                        'value':ifile} for ifile in init_file_list]

init_url = empty_string if not init_file_list else init_file_list[0]


# Configure data uploader
video_filetypes = ['mp4']
du.configure_upload(app, UPLOAD_DIR)
def get_upload_component(id):
    return du.Upload(id=id,
                     max_file_size=1800,  # 1800 Mb
                     filetypes=video_filetypes,
                     upload_id='',
                     text='Drop or Click Here to Upload New Video',
                     default_style=upload_default_style
                    )


# Dash app layout defs
layout = html.Div([
    # Input section
    html.Div(id='video-input-parent',
        style={
            'width': '45%',
            'float': 'left',
            'margin': '0% 5% 1% 5%',
        },
        children=[

            dcc.Markdown(dedent('''## **Input Video**''')),

            # Movie player
            dash_player.DashPlayer(
                id='video-player',
                url=init_url,
                controls=True,
                playing=False,
                volume=1,
                #width='85%',
                #height='80%'
                style={
                    'width': '80%',
                    'margin': '0% 5% 1% 5%',
                    'float' : 'center'
                },
            ),

            # Combined file dropdown and uploader
            html.Div(
                style={
                    'float': 'center',
                    'margin': '2% 0% 0% 0%'
                },
                children=[
                    # Dropdown menu of test names
                    html.Div(
                        [
                            dcc.Dropdown(id='dropdown-file-names',
                                     value=init_url,
                                     #value='',
                                     options=init_dropdown_options,
                                     multi=False,
                                     placeholder='Select file...'),
                        ],
                        style={
                            'width': '50%',
                            'padding': '5px',
                            'margin': '0% 0% 0% 0%',
                            'vertical-align' : 'middle',
                            'display' : 'inline-block'},
                    ),
                    
                    # Uploader container
                    html.Div(
                        [
                            get_upload_component(id='dash-uploader'),
                            html.Div(id='callback-output'),
                        ],
                        style={  # wrapper div style
                            'textAlign': 'center',
                            'width': '50%',
                            'padding': '5px',
                            'display': 'inline-block',
                            'vertical-align' : 'middle',
                            'margin': '0% 0% 0% 0%',
                        }
                    ),

                ]
            ),


            # Time info of video
            html.Div(
                id='div-method-output',
                style={'margin': '10px 0px'}
            ),

            # Sliders
            html.P(dcc.Markdown(dedent("**Volume**:")),
                   style={'margin-top': '10px'}),
            dcc.Slider(
                id='slider-volume',
                min=0,
                max=1,
                step=0.05,
                value=0.5,
                updatemode='drag',
                marks={0: '0%', 1: '100%'}
            ),

            html.P(dcc.Markdown(dedent("**Playback Rate**:")),
                   style={'margin-top': '10px'}),
            dcc.Slider(
                id='slider-playback-rate',
                min=0,
                max=4,
                step=0.25,
                updatemode='drag',
                marks={i: str(i) + 'x' for i in
                       [0, 0.5, 1, 2, 3, 4]},
                value=1
            ),

            html.P(dcc.Markdown(dedent("**Seek To**:")),
                   style={'margin-top': '10px'}),
            dcc.Slider(
                id='slider-seek-to',
                min=0,
                max=1,
                step=0.01,
                updatemode='drag',
                marks={i: str(i * 100) + '%' for i in [0, 0.25, 0.5, 0.75, 1]},
                value=0
            ),
        ]
    ),
    
    # Inference Section
    html.Div(id='inference-parent',
        style={
            'width': '45%',
            'float': 'left',
            'margin': '0% 0% 0% 0%'
        },
        children=[
            dcc.Markdown(dedent('''## **Inference**''')),

            # Container for data table
            html.Div(id='table-div',
                style={
                    'width': '95%',
                    'float': 'center',
                    'margin': '2% 0% 2% 0%'
                },
                children=[
                    # Table of model options & inference results
                    dash_table.DataTable(
                        id='data-table',
                        columns=[{'name' : i, 'id' : i} for i in init_results_df.drop('Colors', axis=1).columns],
                        data=init_results_df.to_dict('records'),
                        page_size=10,
                        selected_rows=[],
                        row_selectable='multi',
                        style_as_list_view=True,
                        style_header=data_table_header_style,
                        style_cell=data_table_style_cell,
                        style_cell_conditional=data_table_style_cell_conditional,
                        style_data_conditional=data_table_style_data_conditional,
                    ),
                ]
            ),

            # Define selected models from checklist
            html.Div(id='model-checklist'),

            # List of selected models for submission
            html.Div(id='printed-model-list'),

            # Button to trigger upload & inference submission
            html.Button(children='Submit', id='send-to-volume', n_clicks=0),

            # Loading circle for upload feedback
            dcc.Loading(id='loading-upload',
                        color='#027bfc',
                        type='circle'),

            html.Br(),

            # Div for volume call
            html.Div(id='file-on-volume'),

            #html.Hr(),

            # Loading circle for submission feedback
            dcc.Loading(id='running-inference',
                        color='#027bfc',
                        type='circle',
                        ),

            # Div to get results output
            html.Div(id='inference-results'),

            # Div to prep results dictionary
            html.Div(id='plottable-data'),

            # Container for bar graph
            # Bar chart with plotted results
            html.Div(id='graph-div',
                style={
                    'width': '95%',
                    'float': 'center',
                    'margin': '2% 0% 0% 0%'
                },
                children=dcc.Graph(id='bar-chart-graph')
                ),

        ]
    ),
])



# Input Section callbacks

# Upload function callback
@du.callback(
    output=Output('callback-output', 'children'),
    id='dash-uploader',
)
def get_a_list(filename):
    print("get_a_list called")
    filenames = update_data_folder_tree()


# Dropdown video options
@app.callback(Output('dropdown-file-names', 'options'),
              [Input('video-input-parent', 'n_clicks')])
def update_options_list(n_clicks):
    '''Update dropdown file list on click'''
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    filenames = update_data_folder_tree()
    options = [
               {'label' : os.path.basename(i), 'value': i} for i in filenames
              ]
    return options



# Inference Section callbacks

# Get results from output
@app.callback(Output('plottable-data', 'data'),
              [Input('send-to-volume', 'n_clicks'),
               Input('model-checklist', 'value'),
               Input('inference-results', 'data')])
def update_data(button_clicks, model_list=[], results=[]):
    '''Define data dict from returned results'''
    data = set_results_dict(model_list=model_list,
                            inference_results=results)
    return data


# Define models with check boxes for inference submission
@app.callback(Output('model-checklist', 'value'),
              [Input('data-table', 'data'),
               Input('data-table', 'selected_rows')])
def print_row(data_df, selected_rows):
    '''Grab models selected by checkboxes'''
    selected_models = [data_df[idx]['Available Models'].lower() for idx in selected_rows]
    return selected_models


# Update data table display based on returned results
@app.callback(Output('data-table', 'data'),
              [Input('send-to-volume', 'n_clicks'),
               Input('inference-results', 'data')])
def display_table_output(button_clicks, results=[]):
    '''Update table based on inference results'''

    # Check if button pressed, clear table results
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'send-to-volume' in changed_id:
        results = []

    data = set_results_dict(model_list=avail_model_list,
                            inference_results=results)
    table_df = build_df(model_list=avail_model_list, 
                        results_dict=data).to_dict('records')
    return table_df


# Print file and model selection list
@app.callback(Output('printed-model-list', 'children'),
              [Input('model-checklist', 'value'),
              Input('dropdown-file-names', 'value')])
def print_file_and_model_list(model_list=[], filename=''):
    '''Print messages based on selected file & models'''
    # Check if a file is selected
    if not filename:
        message = 'Please select a file from the dropdown.'
    else:
        fbasename = os.path.basename(filename)
        message = 'Please select at least one inference model for evaluation of file **{}**.'.format(fbasename)
        # Check if any models selected
        if model_list:
            if len(model_list) == 1:
                message = 'Model '
            else:
                message = 'Models '
            message += 'selected for running inference on file **{}**: {}'.format(fbasename, model_list)
    
    return dcc.Markdown(dedent(message))


# Print feedback from file upload with loading circle
@app.callback([Output('loading-upload', 'children'),
               Output('file-on-volume', 'data')],
              [Input('send-to-volume', 'n_clicks'),
               State('dropdown-file-names', 'value')])
def upload_file(button_clicks, fname=''):
    '''Function to check if file already exists,
       otherwise upload it'''

    # Flag for file on volume is success (True) or failure (False)
    file_on_volume = False

    message = ''

    # Check if button pressed
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if not 'send-to-volume' in changed_id:
        return [dcc.Markdown(dedent(message)), file_on_volume]

    # Check if a file is selected
    if not fname:
        message = 'Please select a file from the dropdown menu.'
        return [dcc.Markdown(dedent(message)), file_on_volume]

    # If not on volume yet, upload
    if file_on_volume:
        message = 'File **{}** already uploaded.'.format(volume_obj_name)
    else:
        upload_success = UploadFile(file_name=fname)

        # Return messages based on success
        if upload_success:
            file_on_volume = True
            message = 'File {} uploaded.'.format(fname)
        else:
            message = 'File {fname} transfer **unsuccessful**. Check error log.'.format(fname)

    return [dcc.Markdown(dedent(message)), file_on_volume]


# Make inference submission to API
@app.callback([Output('running-inference', 'children'),
               Output('inference-results', 'data')],
              [Input('file-on-volume', 'data'),
               State('dropdown-file-names', 'value'),
               State('model-checklist', 'value')])
def submit_inference_request(file_on_volume=False, filename='', model_list=[]):
    '''Submit the request to FakeFinder inference API'''

    message = ''
    results = []

    if file_on_volume:
        volume_obj_name = os.path.basename(filename)

        # Build request
        request_list = BuildInferenceRequest(filename=volume_obj_name,
                                             model_list=model_list)
        print(f'{request_list}')
        # Submit request
        results = SubmitInferenceRequest(url=FF_URL,
                                         dict_list=request_list,
                                         debug=debug)
        if results:
            message = 'Inference submission **successful**. Returning results.'
        else:
            message = 'Inference submission **not successful**. Check error log.'

    return [html.Div([dcc.Markdown(message)]), results]


# Callback for graph display
@app.callback(Output('bar-chart-graph', 'figure'),
              [Input('send-to-volume', 'n_clicks'),
               Input('plottable-data', 'data')])
def update_graph(button_clicks, data={}):
    # Check if button pressed
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'send-to-volume' in changed_id:
        data = {}

    return bar_chart(data=data)



# Videoplayer specific callbacks

# Video selection from dropdown callback
@app.callback(Output('video-player', 'url'),
              [Input('dropdown-file-names', 'value')])
def update_src(value):
    if not value:
        value = init_url
    return value


@app.callback(Output('video-player', 'volume'),
              [Input('slider-volume', 'value')])
def update_volume(value):
    return value


@app.callback(Output('video-player', 'playbackRate'),
              [Input('slider-playback-rate', 'value')])
def update_playbackRate(value):
    return value


@app.callback(Output('div-method-output', 'children'),
              [Input('video-player', 'currentTime'),
               Input('video-player', 'secondsLoaded')],
              [State('video-player', 'duration')])
def update_methods(currentTime, secondsLoaded, duration):
    if currentTime is None:
        currentTime = 0.0
    if secondsLoaded is None:
        secondsLoaded = 0.0
    if duration is None:
        duration = 0.0

    message = '**Current Time**: {:.02f} s'.format(currentTime)
    message += ', **Seconds Loaded**: {:.02f} s'.format(secondsLoaded)
    message += ', **Duration**: {:.02f} s'.format(duration)
    return dcc.Markdown(dedent(message))


@app.callback(Output('video-player', 'intervalCurrentTime'),
              [Input('slider-intervalCurrentTime', 'value')])
def update_intervalCurrentTime(value):
    return value


@app.callback(Output('video-player', 'intervalSecondsLoaded'),
              [Input('slider-intervalSecondsLoaded', 'value')])
def update_intervalSecondsLoaded(value):
    return value


@app.callback(Output('video-player', 'intervalDuration'),
              [Input('slider-intervalDuration', 'value')])
def update_intervalDuration(value):
    return value


@app.callback(Output('video-player', 'seekTo'),
              [Input('slider-seek-to', 'value')])
def set_seekTo(value):
    return value



# Include open source css file
app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

if __name__ == '__main__':
    app.run_server(debug=debug)
