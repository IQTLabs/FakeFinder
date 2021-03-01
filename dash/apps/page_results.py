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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app import app

# Load COVID csv data into pandas dataframe
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.split(REPO_DIR)[0], 'data/')
COMPANY_DATA_DIR = os.path.join(DATA_DIR, 'bycompany/')

# JSON file matching company names to unique datafiles
co_names_json = 'company_names.json'
co_names_filepath = os.path.join(DATA_DIR, co_names_json)

with open(co_names_filepath) as fi:
  co_name_datafile_dict = json.load(fi)

# Get number of unique tests
n_unique_tests = co_name_datafile_dict['n_unique_tests']

# Remove key for unique tests from dictionary
del co_name_datafile_dict['n_unique_tests']

# Get unique company names
unique_companies = list(co_name_datafile_dict.keys())
n_unique_companies = len(unique_companies)

# Load initial data for printed table
data_file_path = os.path.join(COMPANY_DATA_DIR,
                              co_name_datafile_dict[unique_companies[0]])
df = pd.read_csv(data_file_path)

# Setup formatting for display
formats = {'company' : '{}',
           'test_name' : '{}',
           'prevalence': '{:.03f}',
           'ppv': '{:.03f}',
           'ppv_95ci_low': '{:.03f}',
           'ppv_95ci_high' : '{:.03f}',
           'npv' : '{:.03f}',
           'npv_95ci_low' : '{:.03f}',
           'npv_95ci_high' : '{:.03f}',
           'p_pos' : '{:.03f}',
           'p_pos_low' : '{:.03f}',
           'p_pos_high' : '{:.03f}',
           'fdr' : '{:.03f}',
           'fdr_95ci_low' : '{:.03f}',
           'fdr_95ci_high' : '{:.03f}',
           'fomr' : '{:.03f}',
           'fomr_95ci_low' : '{:.03f}',
           'fomr_95ci_high' : '{:.03f}'}

# Generate the table from the pandas dataframe
def generate_table(dataframe, max_rows=10):
    dataframe_text = dataframe.copy()

    # Convert table values to text
    for col, f in formats.items():
        dataframe_text[col] = dataframe_text[col].map(lambda x: f.format(x))
    
    # Return formated table
    return dash_table.DataTable(
                                data=dataframe_text.to_dict('records'),
                                columns=[{'id': c, 'name': c} for c in dataframe_text.columns],
                                virtualization=True,
                                fixed_rows={'headers': True},
                                style_table={
                                    'height': 200,
                                    'overflowX': 'auto',
                                    'overflowY': 'auto',
                                    'minWidth': '100%',
                                },
                               )
# Load flu data
flu_filename = 'flu_plot_mat.csv'
flu_file_path = os.path.join(DATA_DIR, flu_filename)
flu_df = pd.read_csv(flu_file_path)


# Slider range
slider_range = np.linspace(0.05, 1, 20, dtype=np.float)

# Set initial slider value to something low-ish
init_slider_val = 0.05

# Table range
prev_range = np.linspace(0.005, 1, 200, dtype=np.float)

# Define empty string as default
empty_string = ''

# Define an empty figure object / dict as default
empty_figure = {
                 "layout": {
                            "xaxis": {
                                      "title" : '',
                                      "range": [0,1.0],
                                      "showticklabels" : False,
                                      'zeroline': False,
                                      'visible': False,
                                      "ticks" : 'inside',
                                     },
                            "yaxis": {
                                      "title" : '',
                                      "range": [0,1.0],
                                      "showticklabels" : False,
                                      'zeroline': False,
                                      'visible': False,
                                      "ticks" : 'inside',
                                     },
                            "annotations": [
                                            {
                                             "text": "Please select a company and test.",
                                             "xref": "paper",
                                             "yref": "paper",
                                             "showarrow": False,
                                             "font": {"size": 18}
                                            }
                                           ]
                           }
               }

markdown_text_intro = '''
## How to use this app
Researchers may not know the prevalence of novel coronavirus in any one place or time.
But users can explore how diagnostic tests would be expected to perform based on their reported clinical trial data.
The tool below allows you to select one of {} diagnostic tests for novel coronavirus and examine its expected PPV and NPV across a range of disease prevalence, from 0.5% to 100%.

You can choose a prevalence level by adjusting a slider. You'll see a series of images like the one below.  

It's helpful to compare coronavirus diagnostic tests with a baseline.
We include data about an influenza rapid diagnostic test for baseline comparison.
'''.format(n_unique_tests)


markdown_text_end = '''
## Where did the data come from?

The data populated for this app come from three sources:

1. Novel coronavirus diagnostic test data comes from either the FDA or FindDx. 
  - [The FDA publishes data provided by vendors of coronavirus diagnostic tests given Emergency Use Authorization](https://www.fda.gov/emergency-preparedness-and-response/mcm-legal-regulatory-and-policy-framework/emergency-use-authorization)
  - [FindDx is compiling a database of novel coronavirus diagnostic tests from around the world.](https://www.finddx.org/covid-19/dx-data/)
2. Data for the influenza rapid diagnostic comes from the study _[Accuracy of Rapid Influenza Diagnostic Tests: A Meta-Analysis](https://pubmed.ncbi.nlm.nih.gov/22371850/)_

## Completeness of the data

As of this writing, we only include data on molecular PCR tests. We plan to include data on antigen and antibody tests in a future version. 

Also, not every vendor reports clinical trial data, some only report analytical data. In this case, we do not include any performance data. However, a future version of this app may include analytical data as well.

Where there is data from both the FDA's EUA website and FindDX, we choose data from the FDA. That is not to say that we think it's better. Rather, we desire to report data from the manufacturer to give a picutre of what's reported to the FDA. 

FindDx sometimes reports multiple performance studies for a single test. For the tests exclusive to FindDx that have multiple studies, we choose to report the one with the largeest sample size. 

## Calculation details

We calculate confidence intervals for sensitivity and specificity using the [Wilson score interval](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval) for binomial proportions based on reported clinical trial data. 

We calculate PPV and NPV using [Bayes theorem](http://sphweb.bumc.bu.edu/otlt/MPH-Modules/BS/BS704_Probability/BS704_Probability6.html).
We construct the confidence intervals for PPV and NPV by carying through the low and high ends of the 95% confidence intervals on both sensitivity and specificity instead of just the point estimates.

Note that these calculations are based on clinical trial information reported by the diagnostic test manufacturers.
Real world performance may differ based on deviations in how samples are collected, equipment used, and other factors.
'''


markdown_text_disclaimer = '''
## Disclaimer

This page and the related application were developed and are being released by IQT in collaboration with B.Next.
Please note that the application displays standard statistical calculations of 3rd party data as reported.
B.Next and IQT cannot guarantee the accuracy of the reported data.
'''


# Function to generate header text
def generate_header_text(test_val):
    header_text = ''
    if test_val == 'pos':
        header_text = textwrap.dedent('''
                      #### ** Lower prevalence of coronavirus increases the frequency of false positive test results **
                      If many more healthy people in the population are being tested,
                      then the number of false positive tests rivals the number of true positive tests.''')

    if test_val == 'neg':
        header_text = textwrap.dedent('''
                       #### ** Higher prevalence of coronavirus increases the frequency of false negative test results **
                       At low prevalence levels, false negative test results are less of an issue.
                       However, at high prevalence levels the risks of false negatives increase.''')

    return header_text

# Function to generate text with data and results details
def generate_column_text(dataframe, test_val, slider_prevalence_value, inter_idx, company_test_name):

    n_people = 100000
    mkd_text = ''

    p_pos_low = dataframe["p_pos_low"].values[inter_idx]
    p_pos_high = dataframe["p_pos_high"].values[inter_idx]

    ppv_low = dataframe["ppv_95ci_low"].values[inter_idx]
    ppv_high = dataframe["ppv_95ci_high"].values[inter_idx]

    npv_low = dataframe["npv_95ci_low"].values[inter_idx]
    npv_high = dataframe["npv_95ci_high"].values[inter_idx]

    if test_val == 'pos':
        mkd_text = '''
            ** {} **

            At {:.1%} prevalence.
            For every {:,} people tested, one should expect between {:.1%} and {:.1%} to test positive. 
            That's between {:,.0f} and {:,.0f} positive tests.
            
            Of those, between {:,.1%} and {:,.1%} are likely true positives, i.e., 
            the people who tested positive truly have the disease. 
            However, between {:,.1%} and {:,.1%} could be false positives, i.e., 
            those people who tested positive actually don't have the disease.
            
            In other words, in a worst case scenario, if {:,} people are tested, 
            up to {:,.0f} people could receive positive results and up to {:,.0f} of those could be errors.

            '''.format(company_test_name, slider_prevalence_value, n_people, 
                       p_pos_low, p_pos_high, p_pos_low * n_people, p_pos_high * n_people,
                       ppv_low, ppv_high, 1-ppv_high, 1-ppv_low,
                       n_people, p_pos_high * n_people, p_pos_high * (1 - ppv_low) * n_people)

    if test_val == 'neg':
        mkd_text = '''
            ** {} **

            At {:.1%} prevalence.
            For every {:,} people tested, one should expect between {:.1%} and {:.1%} to test negative.
            That's between {:,.0f} and {:,.0f} negative tests.
            
            Of those, between {:,.1%} and {:,.1%} are likely true negatives, i.e.,
            the people who tested negative truly don't have the disease. 
            However, between {:,.1%} and {:,.1%} could be false negatives, i.e., 
            those people who tested negative actually have the disease.
            
            In other words, in a worst case scenario, if {:,} people are tested, 
            up to {:,.0f} people could receive negative results and up to {:,.0f} of those could be errors.
            '''.format(company_test_name, slider_prevalence_value, n_people, 
                       1-p_pos_low, 1-p_pos_high, (1-p_pos_low) * n_people, (1-p_pos_high) * n_people,
                       npv_low, npv_high, 1-npv_high, 1-npv_low,
                       n_people, (1-p_pos_low) * n_people, (1-p_pos_low) * (1 - npv_low) * n_people)

    return mkd_text


# Properties for pos/neg figures
label_defs = {
              'pos' : ['ppv', 'PPV', 'Positive Predictive Value (PPV)', [227, 74, 51], [253, 187, 132]],
              'neg' : ['npv', 'NPV', 'Negative Predictive Value (NPV)', [49, 130, 189], [158, 202, 225]]
             }


# Generate the table from the pandas dataframe
def generate_figure_x(dataframe, test_val='pos', slider_prevalence_value=init_slider_val, company_test_name=''):
    
    # If not provided data, provide empty figure
    if len(dataframe) == 0:
        return empty_string, empty_figure

    fig_labels = label_defs[test_val]
    pred_str = fig_labels[0]
    c_hi_str = '%s_95ci_high'%pred_str
    c_lo_str = '%s_95ci_low'%pred_str
    y_label_str = fig_labels[1]
    title_str = fig_labels[2]
    rgb_l = fig_labels[3]
    rgb_i = fig_labels[4]

    # COVID
    # Get prevalence values
    prev = dataframe['prevalence'].values
    # Get positive pred values
    ppv = dataframe[pred_str].values
    # Get hi/lo intervals on ppv
    c_hi = dataframe[c_hi_str].values
    c_lo = dataframe[c_lo_str].values
    
    # Plot confidence intervals first for fill-between
    conf_int_lo = go.Scatter(x=prev,
                             y=c_lo,
                             fill='none',
                             fillcolor='rgba(%i, %i, %i, 0.1)'%(rgb_i[0], rgb_i[1], rgb_i[2]),
                             line=dict(color='rgba(255,255,255,0)'),
                             showlegend=False,
                             name='conf_int_lo',
                            )
    
    conf_int_hi = go.Scatter(x=prev,
                             y=c_hi,
                             fill='tonexty',
                             fillcolor='rgba(%i, %i, %i, 0.25)'%(rgb_i[0], rgb_i[1], rgb_i[2]),
                             line=dict(color='rgba(255,255,255,0)'),
                             showlegend=False,
                             name='conf_int_hi',
                            )

    # PPV values
    med = go.Scatter(x=prev, y=ppv,
                     line=dict(color='rgba(%s, %s, %s, 1)'%(rgb_l[0], rgb_l[1], rgb_l[2])),
                     showlegend=False,
                     name=title_str)

    # Prevalence value -> index for intercepts
    arr_idx = np.searchsorted(prev_range, slider_prevalence_value, side="left")
    up_intercept = c_hi[arr_idx]
    lo_intercept = c_lo[arr_idx]

    # Vertical line indicating prevalence value
    dashed_prev = go.Scatter(x=slider_prevalence_value*np.ones(len(prev)),
                             y=np.linspace(0,1.1,len(prev)),
                             line=dict(color='rgba(0,0,0,1)',
                                       dash='dot', width=1.5),
                             showlegend=False,
                             name='dashed_prevalence',
                            )
    
    # Horizontal lines indicating hi/lo intercepts
    dashed_lo_intercept = go.Scatter(x=np.linspace(0,1.1,len(prev)),
                                     y=lo_intercept*np.ones(len(prev)),
                                     line=dict(color='rgba(0,0,0,1)',
                                               dash='dot', width=1.5),
                                     showlegend=False,
                                     name='dashed_up_intercept',
                                    )
    dashed_up_intercept = go.Scatter(x=np.linspace(0,1.1,len(prev)),
                                     y=up_intercept*np.ones(len(prev)),
                                     line=dict(color='rgba(0,0,0,1)',
                                               dash='dot', width=1.5),
                                     showlegend=False,
                                     name='dashed_up_intercept',
                                    )
    
    # Influenza
    # Get prevalence values
    flu_prev = flu_df['prevalence'].values
    # Get positive pred values
    flu_ppv = flu_df[pred_str].values
    # Get hi/lo intervals on ppv
    flu_c_hi = flu_df[c_hi_str].values
    flu_c_lo = flu_df[c_lo_str].values
    
    # Plot confidence intervals first for fill-between
    flu_conf_int_lo = go.Scatter(x=flu_prev,
                                 y=flu_c_lo,
                                 fill='none',
                                 fillcolor='rgba(189, 189, 189, 0.1)',
                                 line=dict(color='rgba(255,255,255,0)'),
                                 showlegend=False,
                                 name='flu_conf_int_lo',
                                )
    
    flu_conf_int_hi = go.Scatter(x=flu_prev,
                                 y=flu_c_hi,
                                 fill='tonexty',
                                 fillcolor='rgba(189, 189, 189, 0.25)',
                                 line=dict(color='rgba(255,255,255,0)'),
                                 showlegend=False,
                                 name='flu_conf_int_hi',
                                )

    # PPV values
    flu_med = go.Scatter(x=flu_prev, y=flu_ppv,
                         line=dict(color='rgba(99, 99, 99, 1)'),
                         showlegend=False,
                         name=title_str)

    # Prevalence value -> index for intercepts
    flu_arr_idx = np.searchsorted(prev_range, slider_prevalence_value, side="left")
    flu_up_intercept = flu_c_hi[flu_arr_idx]
    flu_lo_intercept = flu_c_lo[flu_arr_idx]

    # Vertical line indicating prevalence value
    flu_dashed_prev = go.Scatter(x=slider_prevalence_value*np.ones(len(prev)),
                                 y=np.linspace(0,1.1,len(prev)),
                                 line=dict(color='rgba(0,0,0,1)',
                                           dash='dot', width=1.5),
                                 showlegend=False,
                                 name='flu_dashed_prevalence',
                                )
    
    # Horizontal lines indicating hi/lo intercepts
    flu_dashed_lo_intercept = go.Scatter(x=np.linspace(0,1.1,len(prev)),
                                         y=flu_lo_intercept*np.ones(len(prev)),
                                         line=dict(color='rgba(0,0,0,1)',
                                                   dash='dot', width=1.5),
                                         showlegend=False,
                                         name='flu_dashed_up_intercept',
                                        )
    flu_dashed_up_intercept = go.Scatter(x=np.linspace(0,1.1,len(prev)),
                                         y=flu_up_intercept*np.ones(len(prev)),
                                         line=dict(color='rgba(0,0,0,1)',
                                                   dash='dot', width=1.5),
                                         showlegend=False,
                                         name='flu_dashed_up_intercept',
                                        )
   
    # Save list of plots for figure
    covid_data = [med, conf_int_lo, conf_int_hi,
                  dashed_prev, dashed_lo_intercept, dashed_up_intercept]

    # Save list of plots for Influenza figure
    flu_data = [flu_med, flu_conf_int_lo, flu_conf_int_hi,
                flu_dashed_prev, flu_dashed_lo_intercept, flu_dashed_up_intercept]

    flu_title = '%s - Influenza <br> Influenza Rapid Diagnostic'%title_str
    covid_title = textwrap.wrap('%s - COVID-19 %s'%(title_str, company_test_name), width=42)
    extra_breaks = len(covid_title) - 2
    for n in range(extra_breaks):
        flu_title += '<br>    '
    
    # Subplot method
    fig = make_subplots(rows=1, cols=2, start_cell="top-left",
                        subplot_titles=('<br>'.join(covid_title), flu_title))
    fig.update_layout(transition_duration=500,
                      plot_bgcolor='rgb(255,255,255)')
    #fig.update_layout(plot_bgcolor='rgb(255,255,255)')
   
    # First subplot 
    for cdat in covid_data:
        fig.add_trace(cdat, row=1, col=1)

    # Second subplot 
    for fdat in flu_data:
        fig.add_trace(fdat, row=1, col=2)

    # Update both subplot axes
    xaxis=dict(title='Prevalence',
               gridcolor='rgb(210,210,210)',
               range=[0,1.0],
               showgrid=True,
               showline=True,
               mirror=True,
               linecolor='black',
               showticklabels=True,
               tickcolor='rgb(127,127,127)',
               ticks='inside',
               zeroline=True
              )
    yaxis=dict(title=y_label_str,
               gridcolor='rgb(210,210,210)',
               range=[0,1.0],
               showgrid=True,
               showline=True,
               mirror=True,
               linecolor='black',
               showticklabels=True,
               tickcolor='rgb(127,127,127)',
               ticks='inside',
               zeroline=True,
              )
    fig.update_xaxes(xaxis)
    fig.update_yaxes(yaxis)
    fig.update_layout(width=1000, height=500,
                      autosize=False,
                      plot_bgcolor='rgb(255,255,255)')
    fig.update_layout(transition_duration=500,
                      plot_bgcolor='rgb(255,255,255)')
    #fig['layout']['transition']['duration'] = 500

    # Define output text
    header_text = generate_header_text(test_val)
    covid_text = generate_column_text(dataframe, test_val, slider_prevalence_value, arr_idx, company_test_name)
    flu_text = generate_column_text(flu_df, test_val, slider_prevalence_value, flu_arr_idx, 'Rapid Influenza Diagnostic')
    markdown_text = html.Div([
                     dcc.Markdown(children=header_text, id='markdown'),
                     dbc.Row([
                              dbc.Col(html.Div(dcc.Markdown(children=covid_text, id='markdown')), width=6),
                              dbc.Col(html.Div(dcc.Markdown(children=flu_text, id='markdown')), width=6),
                             ], justify="between")])
    
    return markdown_text, fig

# Uncomment for standalone page use.
#app = dash.Dash()
#
#app.layout = html.Div(children=[
#layout = html.Div(children=[
layout = html.Div([

      dbc.Container([
        dcc.Store(id="store-choices"),
        dcc.Store(id="store-figs"),
        dcc.Location(id="result_url", refresh=False),

        #dbc.Row([
        #         dbc.Col(html.H1("How to Use this App", className="text-center"), className="mb-5 mt-5")
        #        ]),
    
        # Intro markdown text from above
        dcc.Markdown(children=markdown_text_intro, id='markdown'),

        html.Hr(),

        # Example plot explanation
        dbc.Row([
                 html.Img(src='assets/plot_explanation.png', height="500px", width="675px" ),
                ], justify='around', className="mb-5"),
        
        html.Hr(),

        html.Br(),

        html.H4('Select a Company and a Test'),

        dcc.Dropdown(id='dropdown-figure-company', 
                     options=[{'label': i, 'value': i} for i in unique_companies],
                     multi=False,
                     placeholder='Filter by company...'),

        html.Br(),

        dcc.Dropdown(id='dropdown-figure-test-name',
                     value='',
                     multi=False,
                     placeholder='Filter by test...'),
        
        html.Br(),

        html.Hr(),

        html.H4('Select a Prevalence'),
        
        html.H5(
            [dcc.Markdown(id="prev_text"),],
             id="prevalences",
             className="mini_container",
        ),
        
        html.Br(),

        dcc.Slider(
            id='prevalence-slider',
            min=0.005,
            max=1.,
            step=0.005,
            value=init_slider_val,
            marks={str(prevalence): str(np.round(prevalence,3)) for prevalence in slider_range},
        ),
        
        html.Br(),

        html.Hr(),

        html.H4('Select a Test Result'),

        html.Br(),

        dbc.Tabs( 
                 [
                  dbc.Tab(label="Positive Result", tab_id="pos_fig"),
                  dbc.Tab(label="Negative Result", tab_id="neg_fig"),
                 ],
            className='nav nav-pills',
            id="tabs",
            active_tab="pos_fig",
        ),
        html.Div(id="tab-content", className="p-4"),

        html.Hr(),

        # Link to download data to csv file
        html.A(dbc.Button(
               'Download Selected Data',
               color='primary',
               className='three columns',
              ),
              id='download-link',
              download="rawdata.csv",
              href="",
              target="_blank"
        ),

        html.Hr(),

        # Data dump of table on screen
        html.Div(id='table-container'),

        html.Hr(),

        # Intro markdown text from above
        dcc.Markdown(children=markdown_text_end, id='markdown-end'),

        html.Hr(),

        # Intro markdown text from above
        dcc.Markdown(children=markdown_text_disclaimer, id='markdown-disc'),

        html.Hr(),

        ]
    )
])

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
  
@app.callback(
    dash.dependencies.Output('dropdown-figure-company', 'value'),
    [dash.dependencies.Input("result_url", "search")],
    [dash.dependencies.State("dropdown-figure-company", "value")],
)
def set_company_value(url_params, current):
    if current != '':
      pass
    if not url_params:
        return ''
    parsed = parse_url_params(url_params)
    url_company = parsed['company'] if 'company' in parsed else None
    selected_company=''
    for s in unique_companies:
      if s.lower() == url_company.lower():
        selected_company = url_company

    return selected_company

# Callback function for company name
@app.callback(
    dash.dependencies.Output('dropdown-figure-test-name', 'options'),
    [dash.dependencies.Input('dropdown-figure-company', 'value')])
def set_test_options(selected_company):
    '''Get and set the company name'''
    if selected_company is None or selected_company is '':
        return []

    co_data_filename = co_name_datafile_dict[selected_company]
    co_data_fullpath = os.path.join(COMPANY_DATA_DIR, co_data_filename)
    df_company_filtered = pd.read_csv(co_data_fullpath)

    # Identify only unique test names for given company
    avail_tests = df_company_filtered.test_name.unique()
    return [{'label': i, 'value': i} for i in avail_tests]


# Callback function for test name(s)
@app.callback(
    dash.dependencies.Output('dropdown-figure-test-name', 'value'),
    [dash.dependencies.Input('dropdown-figure-test-name', 'options'),
     dash.dependencies.Input("result_url", "search")])
def set_test_value(available_options, url_params):
    parsed = parse_url_params(url_params)
    url_test = parsed['test'] if 'test' in parsed else None
    '''Get and set the diagnostic test name'''
    # If no options available, or too many, don't autofill
    if available_options is None or len(available_options) == 0:
        return None
    if len(available_options) == 1:
        return available_options[0]['value']
    if url_test:
      for o in available_options:
        if o['value'].lower() == url_test.lower():
          return o['value']

    # If only one option available, then ok to autofill
    return None


# Selectors -> prev_text
@app.callback(
    dash.dependencies.Output("prev_text", "children"),
    [dash.dependencies.Input("prevalence-slider", "value"),],
)
def update_prev_text(slider_prevalence_value):
    '''Print out current prevalence value'''
    mkd_text = 'Current Value: {} or {}%'.format(slider_prevalence_value, 100. * slider_prevalence_value)
    return mkd_text


@app.callback(
        dash.dependencies.Output("tab-content", "children"),
        [dash.dependencies.Input("tabs", "active_tab"),
         dash.dependencies.Input("store-figs", "data")],
)
def render_tab_content(active_tab, data):
    """
    This callback takes the 'active_tab' property as input, as well as the
    stored graphs, and renders the tab content depending on what the value of
    'active_tab' is.
    """
    if active_tab and data is not None:
        if active_tab == "pos_fig":
            tab_cols = [dbc.Col([dcc.Graph(figure=data["pos_fig"])]),
                       dbc.Col([data['pos_text']])]
            return tab_cols
        elif active_tab == "neg_fig":
            tab_cols = [dbc.Col([dcc.Graph(figure=data["neg_fig"])]),
                       dbc.Col([data['neg_text']])]
            return tab_cols
    return "No tab selected"

@app.callback(
         dash.dependencies.Output("store-figs", "data"), 
        [dash.dependencies.Input('dropdown-figure-company', 'value'),
         dash.dependencies.Input('dropdown-figure-test-name', 'value'),
         dash.dependencies.Input('prevalence-slider', 'value')]
)
def generate_graphs(dropdown_value, dropdown_test_value, slider_prevalence_value):
    """
    This callback generates subplots from covid + flu data.
    """
    table = []
    if dropdown_value is None or dropdown_value is '' or dropdown_test_value is None:
        dff = []
        # Default - display first company data table
        table = generate_table(df)
    else:
        # Access company specific file
        co_data_filename = co_name_datafile_dict[dropdown_value]
        co_data_fullpath = os.path.join(COMPANY_DATA_DIR, co_data_filename)
        df_company = pd.read_csv(co_data_fullpath)
        # Grab specific test results
        dff = df_company[df_company.test_name.str.contains(dropdown_test_value,
                                                           regex=False)]

        table = generate_table(dff)

    # simulate expensive graph generation process
    time.sleep(0.5)

    uid_name = '{} - {}'.format(dropdown_value, dropdown_test_value)

    text_pos, pos_fig = generate_figure_x(dff, 'pos', slider_prevalence_value, uid_name)
    text_neg, neg_fig = generate_figure_x(dff, 'neg', slider_prevalence_value, uid_name)
    # save figures & text in a dictionary for sending to the dcc.Store
    return {"pos_fig": pos_fig, "neg_fig": neg_fig, "pos_text": text_pos, "neg_text": text_neg, 'table' : table}


# Table callback function
@app.callback(
        dash.dependencies.Output('table-container', 'children'),
        [dash.dependencies.Input('store-figs', 'data')]
    )
#def display_table(dropdown_value, dropdown_test_value):
def display_table(data):
    '''Display table via options to generate it'''
    return data['table']


# Include open source css file
app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

if __name__ == '__main__':
    app.run_server(debug=True)

