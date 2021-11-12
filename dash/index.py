
from config import APP_PATH
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

from app import app
# import pages for the app
from apps import home, page_inference

server = app.server

#color_scheme = '#f75040'
color_scheme = '#130047'
#color_scheme = '#027bfc'
#color_scheme = 'light'

# building the upper navigation bar
dropdown = dbc.Navbar(children=[
                dbc.NavItem(dbc.NavLink("About FakeFinder", href="{}/home".format(APP_PATH), external_link=True)),
                dbc.NavItem(dbc.NavLink("|", active=False, disabled=True)),
                dbc.NavItem(dbc.NavLink("Inference Tool", href="{}/page_inference".format(APP_PATH), external_link=True)),
               ],
               color=color_scheme,
               dark=True,
               className="ml-1",
               style={'font-size':'1.5em'})

logo_bar = dbc.Container(
                        [
                         dbc.Row([
                                  dbc.Col(html.A([
                                          html.Img(src="{}/assets/IQT_Labs_Logo.png".format(APP_PATH),
                                                   style={
                                                          'height' : '100px',
                                                          'padding-top' : 10,
                                                          'padding-bottom' : 10,
                                                          'padding-left' : 30,
                                                         }
                                                  )
                                                 ], href='https://www.iqt.org/labs/'),
                                          width=4
                                         ),
                                 ],
                                 justify="between")
                        ]
                       )

navbar = dbc.Navbar(

    dbc.Container(
        [
            html.A([
                    html.Img(src="{}/assets/FakeFinder_Logo.png".format(APP_PATH),
                             style={
                                    'height' : '100px',
                                    'padding-top' : 10,
                                    'padding-bottom' : 10,
                                    'padding-right' : 40,
                                    'padding-left' : 0,
                                   }
                            )
                   ], href='https://www.iqt.org/labs/'),
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(dbc.NavbarBrand("FakeFinder DeepFake Inference Tool", className="ml-1")),
                    ],
                    align="center",
                ),
                href="{}/home".format(APP_PATH),
            ),
            dbc.NavbarToggler(id="navbar-toggler2"),
            dbc.Collapse(
                dbc.Nav(
                    # right align menu with ml-auto className
                    [dropdown], className="ml-auto", navbar=True
                ),
                id="navbar-collapse2",
                navbar=True,
            ),
            html.A([
                    html.Img(src="{}/assets/iqt-labs-white-color.png".format(APP_PATH),
                             style={
                                    'height' : '100px',
                                    'padding-top' : 10,
                                    'padding-bottom' : 10,
                                    'padding-right' : 0,
                                    'padding-left' : 40,
                                   }
                            )
                   ], href='https://www.iqt.org/labs/'),
        ]
    ),
    color=color_scheme,
    dark=True,
    className="mb-4",
)

def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

for i in [2]:
    app.callback(
        Output(f"navbar-collapse{i}", "is_open"),
        [Input(f"navbar-toggler{i}", "n_clicks")],
        [State(f"navbar-collapse{i}", "is_open")],
    )(toggle_navbar_collapse)

# embedding the navigation bar
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    #logo_bar,
    navbar,
    html.Div(id='page-content')
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '{}/page_table'.format(APP_PATH):
        return page_table.layout
    elif pathname == '{}/page_inference'.format(APP_PATH):
        return page_inference.layout
    else:
        return home.layout

if __name__ == '__main__':
    #app.run_server(port=PORT, debug=False)
   # app.run_server(host='127.0.0.1', debug=True)
    app.run_server(host='0.0.0.0', debug=True, dev_tools_silence_routes_logging = False)
