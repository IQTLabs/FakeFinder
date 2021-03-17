


# Color scale for bar plot
bar_chart_color_scale = [(0.0, 'rgb(2, 123, 252)'), 
                         (0.08333333333333333, 'rgb(94, 142, 252)'), 
                         (0.16666666666666666, 'rgb(135, 162, 252)'), 
                         (0.25, 'rgb(167, 183, 252)'), 
                         (0.3333333333333333, 'rgb(196, 205, 252)'),
                         (0.4166666666666667, 'rgb(223, 227, 251)'), 
                         (0.5, 'rgb(249, 249, 249)'), 
                         (0.5833333333333334, 'rgb(255, 223, 216)'), 
                         (0.6666666666666666, 'rgb(255, 196, 184)'), 
                         (0.75, 'rgb(255, 170, 153)'), 
                         (0.8333333333333334, 'rgb(255, 142, 122)'), 
                         (0.9166666666666666, 'rgb(253, 113, 93)'), 
                         (1.0, 'rgb(247, 80, 64)')]


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
     'textAlign': 'center',
     'font-family':'Open Sans',
     'fontSize' : 14,
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
        'color': '#f75040',
        'backgroundColor': 'white'
    },
    {
        'if': {
            'filter_query': '{Colors} = #7dc53e'
        },
        'color': '#027bfc',
        'backgroundColor': 'white'
    },
    {
        'if': {
            'filter_query': '{Colors} = white'
        },
        'backgroundColor': 'white',
        'color': 'black'
    },
]

