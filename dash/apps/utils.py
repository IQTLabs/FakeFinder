
import os
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from .definitions import STATIC_FULLPATH
from .dash_style_defs import bar_chart_color_scale


# Function to update list of video files in DATA_DIR
def update_data_folder_tree(extension='.mp4'):
    dirPath = Path(STATIC_FULLPATH)
    listOfFileNames = [ os.path.join (root, name) \
                        for root, dirs, files in os.walk(dirPath) \
                        for name in sorted(files) \
                        if name.endswith (extension) \
                      ]
    return listOfFileNames


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
def build_df(model_list=[], results_dict={}):
    # Column header names
   
    # Instantiate empty lists for table
    model_names = model_list
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


# Bar chart of inference results
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

    # Plot the score values for each model
    fig = px.bar(x=x_vals, y=y_vals, orientation='h', color=x_vals, 
                 color_continuous_scale=bar_chart_color_scale, 
                 color_continuous_midpoint=0.)

    # Vertical line delineating fake/real boundary
    fig.add_vline(x=0.0, line_width=1, 
                  line_dash="dash", line_color="black")
    
    # Set limits on colorbar
    fig.update_coloraxes(
        cmin=-1.,
        cmax=1.,
    )
    # Titles, tickvals, etc
    fig.update_layout(
        title='<b>Confidence Scores</b>',
        xaxis_title='<b>Confidence Score</b>',
        yaxis_title='',
        coloraxis_colorbar=dict(
            title='<b>Confidence</b>',
            ticks='outside',
            tickmode='array',
            tickvals=[-1., -0.5, 0.0, 0.5, 1.],
            ticktext=['-1 - Real', '-0.5', '0 - Uncertain', '0.5', '1 - Fake']
            ),
        xaxis=dict(
            showgrid=True,
            showline=True,
            showticklabels=True,
            zeroline=True,
            range=[-1.,1.],
            domain=[0., 1.],
            tickvals=[-1., -0.5, 0.0, 0.5, 1.],
            ticktext=['-1<br><b>Real</b></br>', '-0.5', 
                      '0<br><b>Uncertain</b></br>', 
                      '0.5', '1<br><b>Fake</b></br>'],
            tickfont=dict(size=14)
        ),
        yaxis=dict(
            showgrid=True,
            showline=True,
            showticklabels=True,
            zeroline=True,
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=80, r=80, t=80, b=80),
        showlegend=False,
    )

    return fig

