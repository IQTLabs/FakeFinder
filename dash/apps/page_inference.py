from textwrap import dedent

import os
import dash_player
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State

from .definitions import REPO_DIR, DATA_DIR
from .text.general_text import markdown_text_disclaimer

from app import app

video_fname = os.path.join(DATA_DIR, 'Home-Stallone-[DeepFake].mpd')
print(video_fname)
#video_fname = os.path.join(DATA_DIR, 'Home-Stallone-[DeepFake].mp4')

# Define empty string as default
empty_string = ''


layout = html.Div([
    html.Div(
        style={
            'width': '40%',
            'float': 'left',
            'margin': '0% 5% 1% 5%'
        },
        children=[
            dash_player.DashPlayer(
                id='video-player',
                url=video_fname,
                #url='https://youtu.be/2svOtXaD3gg',
                controls=True,
                playing=False,
                volume=1,
                width='100%'
            ),
            html.Div(
                id='div-current-time',
                style={'margin': '10px 0px'}
            ),

            html.Div(
                id='div-method-output',
                style={'margin': '10px 0px'}
            ),

            dcc.Markdown(dedent('''
            ### Other Video Examples
            * https://youtu.be/UuaYDZd8-cI
            * https://www.youtube.com/watch?v=VhFSIR7r7Yo&ab_channel=AftabHaiderMughal 
            * https://youtu.be/f2YbT5SGmGA
            '''))
        ]
    ),

    html.Div(
        style={
            'width': '30%',
            'float': 'left'
        },
        children=[
            dcc.Input(
                id='input-url',
                value=video_fname
                #value='https://youtu.be/2svOtXaD3gg'
            ),

            html.Button('Change URL', id='button-update-url'),

            dcc.Checklist(
                id='radio-bool-props',
                options=[{'label': val.capitalize(), 'value': val} for val in [
                    'playing',
                    'loop',
                    'controls',
                    'muted'
                ]],
                value=['controls']
            ),

            html.P("Volume:", style={'margin-top': '10px'}),
            dcc.Slider(
                id='slider-volume',
                min=0,
                max=1,
                step=0.05,
                value=None,
                updatemode='drag',
                marks={0: '0%', 1: '100%'}
            ),

            html.P("Playback Rate:", style={'margin-top': '25px'}),
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

            #html.P("Update Interval for Current Time:", style={'margin-top': '30px'}),
            #dcc.Slider(
            #    id='slider-intervalCurrentTime',
            #    min=40,
            #    max=1000,
            #    step=None,
            #    updatemode='drag',
            #    marks={i: str(i) for i in [40, 100, 200, 500, 1000]},
            #    value=100
            #),

            #html.P("Update Interval for seconds loaded:", style={'margin-top': '30px'}),
            #dcc.Slider(
            #    id='slider-intervalSecondsLoaded',
            #    min=200,
            #    max=2000,
            #    step=None,
            #    updatemode='drag',
            #    marks={i: str(i) for i in [200, 500, 750, 1000, 2000]},
            #    value=500
            #),

            #html.P("Update Interval for duration:",
            #       style={'margin-top': '30px'}),
            #dcc.Slider(
            #    id='slider-intervalDuration',
            #    min=200,
            #    max=2000,
            #    step=None,
            #    updatemode='drag',
            #    marks={i: str(i) for i in [200, 500, 750, 1000, 2000]},
            #    value=500
            #),

            html.P("Seek To:", style={'margin-top': '30px'}),
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

    html.Hr(),
    html.Br(),
    html.Br(),
    html.Div(
        style={
            'width': '40%',
            'float': 'left',
            'margin': '0% 5% 1% 5%'
        },
        children=[
            dcc.Markdown(dedent('''
            ### Output from Inference
            ''')),

            #dash_player.DashPlayer(
            #    id='video-player',
            #    #url=video_fname,
            #    url='https://youtu.be/2svOtXaD3gg',
            #    controls=True,
            #    playing=False,
            #    volume=1,
            #    width='100%'
            #),
            #html.Div(
            #    id='div-current-time',
            #    style={'margin': '10px 0px'}
            #),

            #html.Div(
            #    id='div-method-output',
            #    style={'margin': '10px 0px'}
            #),

        ]
    ),

])


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


@app.callback(Output('video-player', 'url'),
              [Input('button-update-url', 'n_clicks')],
              [State('input-url', 'value')])
def update_url(n_clicks, value):
    return value


# Instance Methods
@app.callback(Output('div-current-time', 'children'),
              [Input('video-player', 'currentTime')])
def update_time(currentTime):
    if currentTime is None:
        currentTime = 0.0
    return 'Current Time: {:.02f}'.format(currentTime)


@app.callback(Output('div-method-output', 'children'),
              [Input('video-player', 'secondsLoaded')],
              [State('video-player', 'duration')])
def update_methods(secondsLoaded, duration):
    if secondsLoaded is None:
        secondsLoaded = 0.0
    if duration is None:
        duration = 0.0
    return 'Second Loaded: {:.02f}, Duration: {:.03f}'.format(secondsLoaded, duration)


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
