import os
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

import json
import urllib
import pandas as pd
from app import app

pd.set_option("display.precision", 2)

# Load COVID csv data into pandas dataframe
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.split(REPO_DIR)[0],'data/')
COMPANY_DATA_DIR = os.path.join(DATA_DIR, 'bycompany/')

# JSON file matching company names to unique datafiles
co_names_json = 'company_names.json'
co_names_filepath = os.path.join(DATA_DIR, co_names_json)

with open(co_names_filepath) as fi:
  co_name_datafile_dict = json.load(fi)

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

# Convert strings to numeric values
def format_cols(dataframe=[]):
    dataframe = dataframe.apply(pd.to_numeric, errors='ignore')
    for col, f in formats.items():
        dataframe[col] = dataframe[col].map(lambda x: f.format(x))
    return dataframe


markdown_text_disclaimer = '''
## Disclaimer

This page and the related application were developed and are being released by IQT in collaboration with B.Next.
Please note that the application displays standard statistical calculations of 3rd party data as reported.
B.Next and IQT cannot guarantee the accuracy of the reported data.
'''

# Generate the table from the pandas dataframe
def generate_table(dataframe, max_rows=10):
    dataframe = format_cols(dataframe)
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            #html.Td('{}'.format(dataframe.iloc[i][col])) for col in dataframe.columns
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )


#app = dash.Dash()
#
#app.layout = html.Div(children=[
layout = html.Div(children=[
      dbc.Container([
        html.Br(),
        
        # Header
        html.H4('Select a Company and a Test'),

        # Dropdown menu of company names
        dcc.Dropdown(id='dropdown-company', 
                     options=[{'label': i, 'value': i} for i in unique_companies],
                     multi=False,
                     placeholder='Filter by company...'),

        html.Br(),

        # Dropdown menu of test names
        dcc.Dropdown(id='dropdown-test-name',
                     value='',
                     multi=False,
                     placeholder='Filter by test...'),
        
        html.Br(),

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
        html.Div(id='table-cont'),

        html.Hr(),

        # Intro markdown text from above
        dcc.Markdown(children=markdown_text_disclaimer, id='markdown'),

        html.Hr(),

        ]
      )
  ]
)


# Callback function for company name
@app.callback(
    dash.dependencies.Output('dropdown-test-name', 'options'),
    [dash.dependencies.Input('dropdown-company', 'value')])
def set_test_options(selected_company):
    '''Get and set the company name'''
    if selected_company is None or selected_company is '':
        return []
    #comp_cond = df.company.str.contains(selected_company)
    #df_comp_cond = df[comp_cond]
    
    co_data_filename = co_name_datafile_dict[selected_company]
    co_data_fullpath = os.path.join(COMPANY_DATA_DIR, co_data_filename)
    df_company_filtered = pd.read_csv(co_data_fullpath)

    # Identify only unique test names for given company
    avail_tests = df_company_filtered.test_name.unique()
    return [{'label': i, 'value': i} for i in avail_tests]


# Callback function for test name(s)
@app.callback(
    dash.dependencies.Output('dropdown-test-name', 'value'),
    [dash.dependencies.Input('dropdown-test-name', 'options')])
def set_test_value(available_options):
    '''Get and set the diagnostic test name'''
    # If no options available, or too many, don't autofill
    if len(available_options) is None:
        return None
    if len(available_options) == 0 or len(available_options) > 1:
        return None
    # If only one option available, then ok to autofill
    return available_options[0]['value']


# Full callback function
@app.callback(
        dash.dependencies.Output('table-cont', 'children'),
        [dash.dependencies.Input('dropdown-company', 'value'),
         dash.dependencies.Input('dropdown-test-name', 'value')]
    )
def display_table(dropdown_value, dropdown_test_value):
    '''Display table via options to generate it'''

    if dropdown_value is None or dropdown_value is '' or dropdown_test_value is None:
        return generate_table(df)

    # Access company specific file
    co_data_filename = co_name_datafile_dict[dropdown_value]
    co_data_fullpath = os.path.join(COMPANY_DATA_DIR, co_data_filename)
    df_company = pd.read_csv(co_data_fullpath)
    # Grab specific test results
    dff = df_company[df_company.test_name.str.contains(dropdown_test_value,
                                                       regex=False)]

    #dff = df[df.company.str.contains(dropdown_value) &
    #         df.test_name.str.contains(dropdown_test_value)]
    return generate_table(dff)


# Callback for saving csv string of selected data
@app.callback(
    dash.dependencies.Output('download-link', 'href'),
    [dash.dependencies.Input('dropdown-company', 'value'),
     dash.dependencies.Input('dropdown-test-name', 'value')]
)
def update_download_link(dropdown_value, dropdown_test_value):
    '''
       Save selected data to csv 
       Source: https://community.plotly.com/t/download-raw-data/4700/8
    '''
    if dropdown_value is None or dropdown_value is '' or dropdown_test_value is None:
        dff = df

    else:
        # Access company specific file
        co_data_filename = co_name_datafile_dict[dropdown_value]
        co_data_fullpath = os.path.join(COMPANY_DATA_DIR, co_data_filename)
        df_company = pd.read_csv(co_data_fullpath)
        # Grab specific test results
        dff = df_company[df_company.test_name.str.contains(dropdown_test_value,
                                                           regex=False)]

    csv_string = dff.to_csv(index=False, encoding='utf-8')
    csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
    return csv_string


# Include open source css file
app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

if __name__ == '__main__':
    app.run_server(debug=True)

