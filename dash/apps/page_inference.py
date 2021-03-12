from textwrap import dedent

from pathlib import Path
import os
import requests
import dash
import dash_player
import dash_table
import dash_uploader as du
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objects as go
import plotly.express as px
from dash.dependencies import Input, Output, State

import os
import base64
import numpy as np
from urllib.parse import quote as urlquote

import boto3
import flask

from .api_calls import UploadFileToS3, GetModelList, SubmitInferenceRequest, BuildInferenceRequest, CheckFileExistsS3
from .definitions import REPO_DIR, DATA_DIR, STATIC_DIRNAME, STATIC_FULLPATH, BUCKET_NAME, FF_URL
from .text.general_text import markdown_text_disclaimer

from app import app, server


# Define empty string as default
empty_string = ''


# Server route for local video playing by dash-player
@server.route('{}/<path:path>'.format(STATIC_FULLPATH))
def serve_static(path):
    return flask.send_from_directory(STATIC_FULLPATH, path)


# Generate the table from the pandas dataframe
def generate_table(model_list=[], inference_results=[], max_rows=10):
    if not model_list or not inference_results:
        return html.Table()
    # Simple header for table display
    header_text = ['Model', ' ', 'Real? ', ' ', 'Score']

    # Build single dict of results
    inference_dict = {}
    for i_result in inference_results:
        inference_dict.update(i_result)

    # Look for each model in results
    results_dict = {}
    for i_model in model_list:
        if i_model in inference_dict:
            score = inference_dict[i_model]['0']
            if score > 0.5:
                results_dict[i_model] = {'score' : score,
                                         'color' : 'red',
                                         'label' : 'No'}
            else:
                results_dict[i_model] = {'score' : score,
                                         'color' : 'green',
                                         'label' : 'Yes'}

    # Build HTML table
    return html.Table(
                       # Header
                       [html.Tr([html.Th(col) for col in header_text])] +

                       # Body
                       [html.Tr([
                           html.Td('{}'.format(i_model.upper())),
                           html.Td(' '),
                           html.Td('{}'.format(results_dict[i_model]['label'])),
                           html.Td(' '),
                           html.Td('{:.04f}'.format(results_dict[i_model]['score']))],
                           style={'color': results_dict[i_model]['color']}) for i_model in results_dict],
                           style={'border':'5px', 'font-size':'1.2rem'}
                      )


# Function to update list of video files in DATA_DIR
def update_data_folder_tree(extension='.mp4'):
    dirPath = Path(STATIC_FULLPATH)
    listOfFileNames = [ os.path.join (root, name) \
                        for root, dirs, files in os.walk(dirPath) \
                        for name in sorted(files) \
                        if name.endswith (extension) \
                      ]
    return listOfFileNames


# Setup initial dropdown & video options
init_file_list = update_data_folder_tree()
init_dropdown_options=[{'label':os.path.basename(ifile), 
                        'value':ifile} for ifile in init_file_list]

init_url = empty_string if not init_file_list else init_file_list[0]


# Configure data uploader
video_filetypes = ['mp4', 'dat']
du.configure_upload(app, STATIC_FULLPATH)
def get_upload_component(id):
    return du.Upload(
        id=id,
        max_file_size=1800,  # 1800 Mb
        filetypes=video_filetypes,
        upload_id='',
        text='Drop or Click Here to Upload New Video',
        default_style={
            'minHeight' : '30px',
            'lineHeight' : '30px',
            'height': '100%',
            'width': '90%',
            'float': 'center',
            'margin': '0% 0% 0% 0%'
        },
    )


# Get available model list from API
api_model_dict = GetModelList(url=FF_URL)
avail_model_list = api_model_dict['models']
#avail_model_list = ['selimsef', 'eighteen', 'wm']


# Dash app layout defs
layout = html.Div([
    html.Div(id='my-dropdown-parent',
        style={
            'width': '45%',
            'float': 'left',
            'margin': '0% 5% 1% 5%',
        },
        children=[

            dcc.Markdown(dedent('''## **Input Video**''')),

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
                                     value='',
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
                id='div-current-time',
                style={'margin': '10px 0px'}
            ),

            html.Div(
                id='div-method-output',
                style={'margin': '10px 0px'}
            ),

    #        dcc.Input(
    #            id='input-url',
    #            value='https://youtu.be/2svOtXaD3gg'
    #        ),

    #        html.Button('Change URL', id='button-update-url'),

            html.P(dcc.Markdown(dedent("**Volume**:")), style={'margin-top': '10px'}),
            dcc.Slider(
                id='slider-volume',
                min=0,
                max=1,
                step=0.05,
                value=None,
                updatemode='drag',
                marks={0: '0%', 1: '100%'}
            ),

            html.P(dcc.Markdown(dedent("**Playback Rate**:")), style={'margin-top': '10px'}),
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

            html.P(dcc.Markdown(dedent("**Seek To**:")), style={'margin-top': '10px'}),
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
    html.Div(id='my-inference-section',
        style={
            'width': '45%',
            'float': 'left',
            'margin': '0% 0% 0% 0%'
        },
        children=[
            dcc.Markdown(dedent('''## **Inference**''')),

            dcc.Markdown(dedent('''#### **Available Models**:''')),
            dcc.Checklist(
                id='model-checklist',
                options=[{'label': val.upper(), 'value': val} for val in avail_model_list],
                style={"margin-left": "15px"},
                labelStyle={'display': 'block'},
                value=[]
            ),

            html.Div(id='printed-model-list'),
       
            html.Button(children='Submit', id='send-to-aws'),

            html.Hr(),
       
            html.Div(id='submit-list'),
            html.Div(id='inference-results'),

            html.Div(id='table-cont'),

            dcc.Graph(id='graph-with-slider'),

        ]
    ),

])




# Inference callbacks

# Print selection list
@app.callback(Output('printed-model-list', 'children'),
              [Input('model-checklist', 'value'),
              Input('dropdown-file-names', 'value')])
def update_model_list(model_list=[], filename=''):
    fbasename = os.path.basename(filename)
    if not filename:
        message = 'Please select a file from the dropdown.'
    else:
        message = 'Please select at least one inference model for evaluation of file "{}".'.format(fbasename)
    if model_list:
        message = 'Models selected for inference: {}'.format(model_list)
    return dcc.Markdown(dedent(message))


# Make submission to API
@app.callback([Output('submit-list', 'children'),
               Output('inference-results', 'data')],
              [Input('send-to-aws', 'n_clicks'),
               Input('dropdown-file-names', 'value'),
               State('model-checklist', 'value')])
def submit_inference_request(button_clicks, fname='', model_list=[]):

    # Default message and results list
    message = 'Press the "Submit" button above to run the file through the inference models selected.'
    results = []

    # Ensure file requested
    if not fname:
        message = 'Please select a file from the dropdown.'
    # Check for at least one model for inference
    elif model_list:
        s3_obj_name = os.path.basename(fname)
        if CheckFileExistsS3(file_name=fname,
                             bucket=BUCKET_NAME,
                             object_name=s3_obj_name):
            message = 'File {} already uploaded.'.format(fname)
        else:
            message = 'Sending file {} to S3 Bucket {}'.format(fname, BUCKET_NAME)
            upload_success = UploadFileToS3(file_name=fname, 
                                            bucket=BUCKET_NAME,
                                            object_name=s3_obj_name)
        message += '\n\nMoving to inference.'

        # Build request
        request_list = BuildInferenceRequest(filename=s3_obj_name,
                                             bucket=BUCKET_NAME,
                                             model_list=model_list)
        # Submit request
        results = SubmitInferenceRequest(url=FF_URL,
                                         dict_list=request_list)

    return [html.Div([dcc.Markdown(message)]), results]
    

# Display inference results
@app.callback(
        dash.dependencies.Output('table-cont', 'children'),
        [Input('model-checklist', 'value'),
         Input('inference-results', 'data')]
    )
def display_table(model_list=[], results=[]):
    '''Display table via options to generate it'''
    return generate_table(model_list=model_list, inference_results=results)





@du.callback(
    output=Output('callback-output', 'children'),
    id='dash-uploader',
)
def get_a_list(filename):
    filenames = update_data_folder_tree()



@app.callback(
     Output('dropdown-file-names', 'options'),
     [Input('my-dropdown-parent', 'n_clicks')])
def update_options_list(n_clicks):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    
    filenames = update_data_folder_tree()
    options = [
               {'label' : os.path.basename(i), 'value': i} for i in filenames
              ]
    return options
    


# Bar chart of results
def bar_chart(data):
    ########### Hold in for real data feed
    #data = {'selimsef' : {'score' : 0.5},
    #        'eighteen' : {'score' : 0.9},
    #        'wm' : {'score' : 0.2}}
    x_vals = np.asarray([data[key]['score'] for key in data])
    y_vals = np.asarray([key for key in data])

    # Make a gradient colorscale based on score value
    color_scale = color_continuous_scale=[(0., 'rgb(2, 123, 252)'), (1., 'rgb(247, 80, 64)')]

    # Plot the score values for each model
    fig = px.bar(x=x_vals, y=y_vals, orientation='h', color=x_vals, color_continuous_scale=color_scale)

    # Vertical line delineating fake/real boundary
    fig.add_vline(x=0.5, line_width=2, line_dash="dash", line_color="rgb(247, 80, 64)"
    )
    # Set limits on colorbar
    fig.update_coloraxes(
        cmin=0.,
        cmax=1.,
    )
    # Titles, tickvals, etc
    fig.update_layout(
        title='Model Scores',
        xaxis_title='Scores',
        yaxis_title='',
        coloraxis_colorbar=dict(
            title='Score',
            ticks='outside',
            tickmode='array',
            tickvals=[0., 0.25, 0.5, 0.75, 1.],
            ticktext=['0 - Real', '0.25', '0.5', '.75', '1 - Fake']
            ),
        xaxis=dict(
            showgrid=True,
            showline=True,
            showticklabels=True,
            zeroline=True,
            domain=[0., 1.]
        ),
        yaxis=dict(
            showgrid=True,
            showline=True,
            showticklabels=True,
            zeroline=True,
        ),
        paper_bgcolor='rgb(248, 248, 255)',
        plot_bgcolor='rgb(248, 248, 255)',
        margin=dict(l=80, r=80, t=80, b=80),
        showlegend=False,
    )

    return fig

# Callback for graph display
@app.callback(Output('graph-with-slider', 'figure'),
              [Input('inference-results', 'data')])
def update_graph(input_data_dict):
    return bar_chart(data=input_data_dict)





# Videoplayer callbacks

# Video selection from dropdown callback
@app.callback(Output('video-player', 'url'),
              [Input('dropdown-file-names', 'value')])
def update_src(value):
    if not value:
        value = init_url
    return value


@app.callback(Output('video-player', 'playing'),
              [Input('radio-bool-props', 'value')])
def update_prop_playing(value):
    return 'playing' in value


@app.callback(Output('video-player', 'loop'),
              [Input('radio-bool-props', 'value')])
def update_prop_loop(value):
    return 'loop' in value


@app.callback(Output('video-player', 'controls'),
              [Input('radio-bool-props', 'value')])
def update_prop_controls(value):
    return 'controls' in value


@app.callback(Output('video-player', 'muted'),
              [Input('radio-bool-props', 'value')])
def update_prop_muted(value):
    return 'muted' in value


@app.callback(Output('video-player', 'volume'),
              [Input('slider-volume', 'value')])
def update_volume(value):
    return value


@app.callback(Output('video-player', 'playbackRate'),
              [Input('slider-playback-rate', 'value')])
def update_playbackRate(value):
    return value


#@app.callback(Output('video-player', 'url'),
#              [Input('button-update-url', 'n_clicks')],
#              [State('input-url', 'value')])
#def update_url(n_clicks, value):
#    return value


# Instance Methods
@app.callback(Output('div-current-time', 'children'),
              [Input('video-player', 'currentTime')])
def update_time(currentTime):
    if currentTime is None:
        currentTime = 0.0
    return dcc.Markdown(dedent('**Current Time**: {:.02f} s'.format(currentTime)))


@app.callback(Output('div-method-output', 'children'),
              [Input('video-player', 'secondsLoaded')],
              [State('video-player', 'duration')])
def update_methods(secondsLoaded, duration):
    if secondsLoaded is None:
        secondsLoaded = 0.0
    if duration is None:
        duration = 0.0
    return dcc.Markdown(dedent('**Second Loaded**: {:.02f} s, **Duration**: {:.03f} s'.format(secondsLoaded, duration)))


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
    app.run_server(debug=True)
