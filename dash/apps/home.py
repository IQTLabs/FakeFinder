import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from config import APP_PATH
from .text.home_text import markdown_text_intro, markdown_text_body
from .text.general_text import markdown_text_disclaimer

# needed only if running this as a single page app
#external_stylesheets = [dbc.themes.LUX]




#app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# change to app.layout if running as single page app instead
layout = html.Div(children=[
    dbc.Container([
        dbc.Row([
                 dbc.Col(html.H1("The FakeFinder DeepFake Project", className="text-center"), className="mb-5 mt-5")
                ]),
    
        # Intro markdown text from above
        dcc.Markdown(children=markdown_text_intro, id='markdown'),
        
        html.Br(),
        
        # Example plot explanation
        dbc.Row([
                 html.Img(src='assets/preprocessing.png', height="500px", width="650px" ),
                ], justify='around', className="mb-5"),
        
        ## Example plot explanation
        #dbc.Row([
        #         html.Img(src='assets/preprocess_example.png', height="300px", width="850px" ),
        #        ], justify='around', className="mb-5"),
        
        html.Br(),
        
        # Link to results tool
        html.A(dbc.Button(
               'Go to Inference Tool',
               color='primary',
               className='three columns',
              ),
              id='inference-link',
              href="{}/page_inference".format(APP_PATH),
              target="_self"
        ),

        html.Br(),
        html.Br(),

        # Markdown text from above
        dcc.Markdown(children=markdown_text_body, id='markdown'),

        # Example plot explanation
        dbc.Row([
                 html.Img(src='assets/stylegan2_example.png', height="500px", width="500px" ),
                ], justify='around', className="mb-5"),
        
        # Link to results tool
        html.A(dbc.Button(
               'Go to Inference Tool',
               color='primary',
               className='three columns',
              ),
              id='inference-link',
              href="{}/page_inference".format(APP_PATH),
              target="_self"
        ),

        html.Hr(),

        # Lower page buttons to links
        dbc.Row([
                 dbc.Col(dbc.Card(children=[html.H3(children='Access data & code used to build this dashboard',
                                                    className="text-center"),
                                            html.A([
                                            html.Img(src="{}/assets/github_logo.png".format(APP_PATH),
                                                     style={
                                                            'height' : '7vw',
                                                            'min-height' : '1vw',
                                                            'padding-top' : 10,
                                                            'padding-bottom' : 10,
                                                           }
                                                    )
                                                   ], className='text-center', href='https://github.com/IQTLabs/FakeFinder'),
                                           ], body=True, color="dark", outline=True), width=2, lg=4, className="mb-4"),

                 #dbc.Col(dbc.Card(children=[html.H3(children='Explore other work from B.Next',
                 #                                   className="text-center"),
                 #                           html.A([
                 #                           html.Img(src="{}/assets/BNext_Logo.png".format(APP_PATH),
                 #                                    style={
                 #                                           'height' : '7vw',
                 #                                           'min-height' : '1vw',
                 #                                           'padding-top' : 10,
                 #                                           'padding-bottom' : 10,
                 #                                           'className': 'text-center'
                 #                                          }
                 #                                   )
                 #                                  ], className='text-center', href='https://www.bnext.org/'),
                 #                          ], body=True, color="dark", outline=True), width=2, lg=4, className="mb-4"),

                 dbc.Col(dbc.Card(children=[html.H3(children='Explore other research areas of IQT Labs',
                                                    className="text-center"),
                                            html.A([
                                            html.Img(src="{}/assets/IQT_Labs_Logo.png".format(APP_PATH),
                                                     style={
                                                            'height' : '7vw',
                                                            'min-height' : '1vw',
                                                            'padding-top' : 10,
                                                            'padding-bottom' : 10,
                                                           }
                                                    )
                                                   ], className='text-center', href='https://www.iqt.org/labs/'),
                                           ], body=True, color="dark", outline=True), width=2, lg=4, className="mb-4")
                ], justify='around', className="mb-5"),
                #], justify='around', align='stretch', className="mb-5"),
        
        html.Hr(),
        
        # Intro markdown text from above
        dcc.Markdown(children=markdown_text_disclaimer, id='markdown-disc'),

        html.Hr(),

        #html.A("Special thanks to ..."),
        
        #html.Hr(),

    ])

])

# needed only if running this as a single page app
# if __name__ == '__main__':
#     app.run_server(host='127.0.0.1', debug=True)
