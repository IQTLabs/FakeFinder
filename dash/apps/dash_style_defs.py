

# Uploader Style
upload_default_style={
    'minHeight' : '30px',
    'lineHeight' : '30px',
    'height': '100%',
    'width': '90%',
    'float': 'center',
    'margin': '0% 0% 0% 0%'
}



# DataTable

# Style definitions for dash data table
data_table_header_style = {
     'backgroundColor': '#c4e0ff',
     'fontWeight': 'bold',
     'textAlign': 'center'
    }

data_table_style_cell = {
     'overflow': 'hidden',
     'textOverflow': 'ellipsis',
     'maxWidth': 0,
     'textAlign': 'center'
    }

data_table_style_cell_conditional = [
    {
        'if': {'column_id': 'Available Models'},
        'textAlign': 'left',
    }
]

data_table_style_data_conditional = [
    {
        'if': {
            'filter_query': '{Colors} = #f75040'
        },
        'backgroundColor': '#f75040',
        #'color': 'white'
        'color': 'black'
    },
    {
        'if': {
            'filter_query': '{Colors} = #7dc53e'
        },
        'backgroundColor': '#7dc53e',
        #'color': 'white'
        'color': 'black'
    },
    {
        'if': {
            'filter_query': '{Colors} = white'
        },
        'backgroundColor': 'white',
        'color': 'black'
    },
]

