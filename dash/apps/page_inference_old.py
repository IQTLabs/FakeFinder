import os
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

import json
import time
import textwrap
import urllib.parse
import numpy as np
import pandas as pd
import plotly.express as px
import dash_player as dash_player
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#from flask import Flask, Response

from .definitions import REPO_DIR
from .text.general_text import markdown_text_disclaimer

from app import app

# Load COVID csv data into pandas dataframe
#REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.split(REPO_DIR)[0], 'data/')
video_fname = os.path.join(DATA_DIR, 'Home Stallone [DeepFake].mp4')

# Define empty string as default
empty_string = ''


# Uncomment for standalone page use.
#app = dash.Dash()
#
#app.layout = html.Div(children=[
#layout = html.Div(children=[
layout = html.Div([

      dbc.Container([
        #dcc.Store(id="store-choices"),
        #dcc.Store(id="store-figs"),
        #dcc.Location(id="result_url", refresh=False),

        #dbc.Row([
        #         dbc.Col(html.H1("How to Use this App", className="text-center"), className="mb-5 mt-5")
        #        ]),
    
        html.Hr(),

        html.Br(),

        html.H4('Load video link'),

        dcc.Upload(
            id='upload-image',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            # Allow multiple files to be uploaded
            multiple=True
        ),
        html.Div(id='output-image-upload'),
        

        #html.Video(src="https://www.youtube.com/watch?v=2svOtXaD3gg&t=33s&ab_channel=CtrlShiftFace"),

        #html.H4('Select a Prevalence'),
        #
        #html.H5(
        #    [dcc.Markdown(id="prev_text"),],
        #     id="prevalences",
        #     className="mini_container",
        #),
        
        html.Br(),
        #html.Div(
        #         className='video-outer-container',
        #         children=html.Div(
        #             style={'width': '100%', 'paddingBottom': '56.25%', 'position': 'relative'},
        #             children=dash_player.DashPlayer(
        #                 id='video-display',
        #                 style={'position': 'absolute', 'width': '100%',
        #                        'height': '100%', 'top': '0', 'left': '0', 'bottom': '0', 'right': '0'},
        #                 #url=video_fname,
        #                 url='https://www.youtube.com/watch?v=2svOtXaD3gg&t=35s&ab_channel=CtrlShiftFace',
        #                 controls=True,
        #                 playing=False,
        #                 volume=1,
        #                 width='50%',
        #                 height='50%'
        #             )
        #         )
        #),

        html.Hr(),

        html.H4('Select a Test Result'),

        html.Br(),
        



        dash_player.DashPlayer(
            id='video-player',
            #url='http://media.w3.org/2010/05/bunny/movie.mp4',
            url=video_fname,
            controls=True,
            playing=False,
            volume=1,
            width='50%',
            height='50%'
        ),
        
        html.Button('Set seekTo to 10', id='button-seek-to'),
        html.Button('Switch video', id='button-switch-video'),
        
        html.Div(id='div-current-time', style={'margin-bottom': '20px'}),
        
        html.Div(id='div-method-output'),




        html.Hr(),
        html.Br(),

        #html.Hr(),

        ## Link to download data to csv file
        #html.A(dbc.Button(
        #       'Download Selected Data',
        #       color='primary',
        #       className='three columns',
        #      ),
        #      id='download-link',
        #      download="rawdata.csv",
        #      href="",
        #      target="_blank"
        #),

        html.Hr(),
        
        # Intro markdown text from above
        dcc.Markdown(children=markdown_text_disclaimer, id='markdown-disc'),

        html.Hr(),

        # Data dump of table on screen
        html.Div(id='table-container'),

        html.Hr(),

        html.Hr(),

        ]
    )
])


def parse_contents(contents):
    return html.Div([

        # HTML images accept base64 encoded strings 
        # in the same format that is supplied by the upload
        html.Img(src=contents),
        html.Hr(),
        html.Div('Raw Content'),
        html.Pre(contents[:100] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])
@app.callback(dash.dependencies.Output('output-image-upload', 'children'),
              [dash.dependencies.Input('upload-image', 'contents')])
def update_output(images):
    if not images:
        return

    for i, image_str in enumerate(images):
        image = image_str.split(',')[1]
        data = decodestring(image.encode('ascii'))
        with open(f"image_{i+1}.jpg", "wb") as f:
            f.write(data)

    children = [parse_contents(i) for i in images]
    print(images)
    return children


# Footage Selection
@app.callback(dash.dependencies.Output("video-display", "url"),
              [dash.dependencies.Input('dropdown-footage-selection', 'value'),
               dash.dependencies.Input('dropdown-video-display-mode', 'value')])
def select_footage(footage, display_mode):
    # Find desired footage and update player video
    url = url_dict[display_mode][footage]
    return url


def parse_url_params(string):
  param_dict ={}
  if len(string) > 0 and string[0] == '?':
    string = string[1:]
  decoded = urllib.parse.parse_qs(string)

  for key in decoded:
    if isinstance(decoded[key], list) and len(decoded[key]) == 1:
      param_dict[key] = decoded[key][0]
    else:
      param_dict[key] = decoded[key]

  return param_dict









@app.callback(dash.dependencies.Output('div-current-time', 'children'),
              [dash.dependencies.Input('video-player', 'currentTime')])
def update_time(currentTime):
    return 'Current Time: {}'.format(currentTime)


@app.callback(dash.dependencies.Output('div-method-output', 'children'),
              [dash.dependencies.Input('video-player', 'secondsLoaded')],
              [dash.dependencies.State('video-player', 'duration')])
def update_methods(secondsLoaded, duration):
    return 'Second Loaded: {}, Duration: {}'.format(secondsLoaded, duration)


@app.callback(dash.dependencies.Output('video-player', 'seekTo'),
              [dash.dependencies.Input('button-seek-to', 'n_clicks')])
def set_seekTo(n_clicks):
    return 10


@app.callback(dash.dependencies.Output('video-player','url'),
              [dash.dependencies.Input('button-switch-video','n_clicks')])
def switch_video(n_clicks):
    if (n_clicks is None) or (n_clicks % 2):
        return "static/movie.mp4"
    else:
        return "static/movie_orig.mp4"

  

# Include open source css file
app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

if __name__ == '__main__':
    app.run_server(debug=True)

