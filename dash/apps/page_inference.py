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

import pandas as pd

import os
import base64
import numpy as np
from urllib.parse import quote as urlquote

import boto3
import flask

from .text.general_text import markdown_text_disclaimer
from .dash_style_defs import bar_chart_color_scale, upload_default_style
from .dash_style_defs import data_table_header_style, data_table_style_cell
from .dash_style_defs import data_table_style_cell_conditional, data_table_style_data_conditional
from .definitions import REPO_DIR, DATA_DIR, STATIC_DIRNAME, STATIC_FULLPATH, BUCKET_NAME, FF_URL
from .api_calls import UploadFileToS3, GetModelList, SubmitInferenceRequest, BuildInferenceRequest, CheckFileExistsS3

from app import app, server


# Define empty string as default
empty_string = ''

# Debug flag for if no server connectivity
#debug = False
debug = True


# Get available model list from API
api_model_dict = GetModelList(url=FF_URL, debug=debug)
avail_model_list = api_model_dict['models']


# Server route for local video playing by dash-player
@server.route('{}/<path:path>'.format(STATIC_FULLPATH))
def serve_static(path):
    return flask.send_from_directory(STATIC_FULLPATH, path)


# Generate simple dict of results for other functions
def set_results_dict(model_list=[], inference_results=[]):
    if not model_list or not inference_results:
        return {}
    
    # Build single dict of results
    inference_dict = {}
    for i_result in inference_results:
        inference_dict.update(i_result)
    
    # Look for each model in results
    results_dict = {}
    for i_model in model_list:
        if i_model in inference_dict:
            score = inference_dict[i_model]['0']
            if score == 0.5:
                results_dict[i_model] = {'score' : score,
                                         'color' : 'white',
                                         'label' : 'Undetermined'}
            elif score > 0.5:
                results_dict[i_model] = {'score' : score,
                                         'color' : '#f75040',
                                         'label' : 'Fake'}
            else:
                results_dict[i_model] = {'score' : score,
                                         'color' : '#7dc53e',
                                         'label' : 'Real'}

    return results_dict


# Generate the table from the pandas dataframe
def build_df(results_dict={}):
    # Column header names
   
    # Instantiate empty lists for table
    model_names = avail_model_list
    model_names_upper = []
    num_models = len(model_names)
    model_labels = [[]]*num_models
    model_scores = [[]]*num_models
    model_colors = ['white']*num_models

    # If there are results, built list structures
    for idx, imodel in enumerate(model_names):
        model_names_upper.append(imodel.upper())
        if imodel in results_dict:
            imodel_dict = results_dict[imodel]
            model_scores[idx] = imodel_dict['score']
            model_labels[idx] = imodel_dict['label']
            model_colors[idx] = imodel_dict['color']

    data_dict = {'Available Models' : model_names_upper,
                 'Real or Fake': model_labels,
                 'Probability of Being Fake' : model_scores,
                 'Colors' : model_colors}

    results_df = pd.DataFrame.from_dict(data_dict)
    return results_df

init_results_df = build_df()


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
video_filetypes = ['mp4']
du.configure_upload(app, STATIC_FULLPATH)
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
            html.Button(children='Submit', id='send-to-aws'),

            # Loading circle for upload feedback
            dcc.Loading(id='loading-s3upload',
                        color='#027bfc',
                        type='circle'),

            html.Br(),
           
            # Div for s3 call 
            html.Div(id='file-on-s3'),
       
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


# Get results from output
@app.callback(Output('plottable-data', 'data'),
              [Input('send-to-aws', 'n_clicks'),
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
              [Input('inference-results', 'data')])
def display_table_output(results=[]):
    '''Update table based on inference results'''
    data = set_results_dict(model_list=avail_model_list,
                            inference_results=results)
    table_df = build_df(data).to_dict('records')
    return table_df


# Inference callbacks

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


## Message for upload loading
#@app.callback(Output('printing-upload-loading', 'children'),
#              [Input('send-to-aws', 'n_clicks'),
#               Input('loading-s3upload', 'loading_state'),
#               Input('running-inference', 'loading_state')])
#def print_upload_loading(button_clicks, upload_loading_state, inference_loading_state):
#    message = ''
#    print('Testing')
#    print(upload_loading_state, inference_loading_state)
#    if upload_loading_state:
#        if upload_loading_state['is_loading']:
#            message = 'Uploading file'
#            print(message)
#    return dcc.Markdown(dedent(message))



# Print feedback from file upload with loading circle
@app.callback([Output('loading-s3upload', 'children'),
               Output('file-on-s3', 'data')],
              [Input('send-to-aws', 'n_clicks'),
               State('dropdown-file-names', 'value')])
def upload_file_to_s3(button_clicks, fname=''):
    '''Function to check if file already in s3 bucket, 
       otherwise upload it'''

    # Flag for file on s3 is success (True) or failure (False)
    file_on_s3 = False

    message = ''
    # Check if a file is selected
    if not fname:
        return [dcc.Markdown(dedent(message)), file_on_s3]

    # Check if file on AWS
    s3_obj_name = os.path.basename(fname)
    file_on_s3 = CheckFileExistsS3(file_name=fname,
                                   bucket=BUCKET_NAME,
                                   object_name=s3_obj_name,
                                   debug=debug)

    # If not on s3 yet, upload
    if file_on_s3:
        message = 'File **{}** already uploaded.'.format(s3_obj_name)
    else:
        upload_success = UploadFileToS3(file_name=fname, 
                                        bucket=BUCKET_NAME,
                                        object_name=s3_obj_name,
                                        debug=debug)

        # Return messages based on success
        if upload_success:
            file_on_s3 = True
            message = 'File {} sent to S3 Bucket {}'.format(fname, BUCKET_NAME)
        else:
            message = 'File transfer **unsuccessful**. Check error log.'.format(fname, BUCKET_NAME)

    return [dcc.Markdown(dedent(message)), file_on_s3]


# Make inference submission to API
@app.callback([Output('running-inference', 'children'),
               Output('inference-results', 'data')],
              [Input('file-on-s3', 'data'),
               State('dropdown-file-names', 'value'),
               State('model-checklist', 'value')])
def submit_inference_request(file_on_s3=False, filename='', model_list=[]):
    '''Submit the request to FakeFinder inference API'''

    message = ''
    results = []

    if file_on_s3:
        s3_obj_name = os.path.basename(filename)

        # Build request
        request_list = BuildInferenceRequest(filename=s3_obj_name,
                                             bucket=BUCKET_NAME,
                                             model_list=model_list)

        # Submit request
        results = SubmitInferenceRequest(url=FF_URL,
                                         dict_list=request_list,
                                         debug=debug)
        if results:
            message = 'Inference submission **successful**. Returning results.'
        else:
            message = 'Inference submission **not successful**. Check error log.'

    return [html.Div([dcc.Markdown(message)]), results]
    



@du.callback(
    output=Output('callback-output', 'children'),
    id='dash-uploader',
)
def get_a_list(filename):
    filenames = update_data_folder_tree()



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
    


# Bar chart of results
def bar_chart(data={}):
    '''Build a bar chart figure with inference results'''

    if not data:
        data = {'' : {'score' : 0.5}}

    # Rescale score as confidence
    x_vals = [2. * data[key]['score'] - 1. for key in data]
    y_vals = [key.upper() for key in data]
    n_vals = len(x_vals)

    # Include model average
    if n_vals > 1:
        x_vals.append(sum(x_vals)/n_vals)
        y_vals.append('Average')

    x_vals = np.asarray(x_vals)
    y_vals = np.asarray(y_vals)

    # Make a gradient colorscale based on score value
    #color_scale = [(0., 'rgb(2, 123, 252)'), (1., 'rgb(247, 80, 64)')]
    color_scale = bar_chart_color_scale
    #color_scale = [(0., 'rgb(2, 123, 252)'), (1., 'rgb(247, 80, 64)')]
    #color_scale = color_continuous_scale=[(0., 'rgb(2, 123, 252)'), (1., 'rgb(247, 80, 64)')]
    color_midpoint = 0.
    #color_midpoint = 0.5
    #color_midpoint = ['rgb(259, 259, 259)']

    # Plot the score values for each model
    fig = px.bar(x=x_vals, y=y_vals, orientation='h', color=x_vals, 
                 color_continuous_scale=color_scale, 
                 color_continuous_midpoint=color_midpoint)

    # Vertical line delineating fake/real boundary
    fig.add_vline(x=0.0, line_width=2, 
                  line_dash="dash", line_color="rgb(247, 80, 64)")
    #fig.add_annotation(x=1, y=0,
    #                   text="Text annotation with arrow",
    #                   showarrow=True,
    #                   arrowhead=1)
    # Set limits on colorbar
    fig.update_coloraxes(
        cmin=-1.,
        cmax=1.,
    )
    # Titles, tickvals, etc
    fig.update_layout(
        title='Confidence Scores',
        xaxis_title='Confidence Score',
        yaxis_title='',
        coloraxis_colorbar=dict(
            title='Confidence',
            ticks='outside',
            tickmode='array',
            tickvals=[-1., -0.5, 0.0, 0.5, 1.],
            ticktext=['-1 - Real', '-0.5', '0 - Unsure', '0.5', '1 - Fake']
            ),
        xaxis=dict(
            showgrid=True,
            showline=True,
            showticklabels=True,
            zeroline=True,
            range=[-1.,1.],
            domain=[0., 1.]
        ),
        yaxis=dict(
            showgrid=True,
            showline=True,
            showticklabels=True,
            zeroline=True,
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        #paper_bgcolor='rgb(248, 248, 255)',
        #plot_bgcolor='rgb(248, 248, 255)',
        margin=dict(l=80, r=80, t=80, b=80),
        showlegend=False,
    )

    return fig


# Callback for graph display
@app.callback(Output('bar-chart-graph', 'figure'),
              [Input('send-to-aws', 'n_clicks'),
               Input('plottable-data', 'data')])
def update_graph(button_clicks, data={}):

    #changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    #if 'btn-nclicks-1' in changed_id:
    #    data = {} 

    return bar_chart(data=data)





# Videoplayer callbacks
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
    app.run_server(debug=True)
