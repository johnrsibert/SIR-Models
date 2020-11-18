# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
#import plotly.tools as tls
import pandas as pd

import js_covid as cv
import cv20

#df = pd.read_csv('https://gist.githubusercontent.com/chriddyp/c78bf172206ce24f77d6363a2d754b59/raw/c353e8ef842413cae56ae3920b8fd78468aa4cb2/usa-agricultural-exports-2011.csv')

cv.population_dat = pd.read_csv(cv.census_data_path,header=0,comment='#')
#df = cv.population_dat
#BC_dat =  pd.read_csv( 'http://www.bccdc.ca/Health-Info-Site/Documents/BCCDC_COVID19_Dashboard_Case_Details.csv',header=0) 
#df = BC_dat
#NYT_dat = pd.read_csv('https://github.com/nytimes/covid-19-data/blob/master/us-counties.csv',header=0)
NYT_dat = pd.read_csv('/home/other/nytimes-covid-19-data/us-counties.csv',header=0)
#df = NYT_dat
#print(df)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div(['Geography: ',
              dcc.Input(id='Geog-state', type='text', value='Los Angeles')]),
    html.Div(['Surrounded By: ',
              dcc.Input(id='SurroundedBy-state', type='text', value='California')]),
    html.Button(id='submit-button-state', n_clicks=0, children='Submit'),
    html.Div(id='output-state'),
    dcc.Graph(id='prevalence-graphic')
])


@app.callback(#Output('output-state', 'children'),
              Output('prevalence-graphic', component_property='src'),
              [Input('submit-button-state', 'n_clicks')],
              [State('Geog-state', 'value'),
               State('SurroundedBy-state', 'value')])
def update_output(n_clicks, gname, surrounding):
    dat = cv.population_dat
    print(cv.population_dat)
    state_filter = cv.population_dat['state'].isin([surrounding])
    county_filter = cv.population_dat['county'].isin([gname])
    county_row = state_filter & county_filter
    print(cv.population_dat[county_row].values)
    try:
        population = int(pd.to_numeric(cv.population_dat[county_row]['population'].values))
        code = cv.population_dat[county_row]['code'].values
        fips = int(cv.population_dat[county_row]['fips'].values)
    except:
        print('get_county_pop() failed for:')
        print(gname,surrounding)
        population = 0


    moniker = str(gname+code)
    moniker = moniker.replace(' ','_',5) 
    print('population:',population,'; area code:',code,'; fips:',fips,'; moniker:',moniker)
    test = cv20.Geography(name=gname,enclosed_by=surrounding,code=code)
    test.read_nyt_data()
#   def plot_prevalence(self,yscale='linear', per_capita=False, delta_ts=True,
#                       window=[11], plot_dt = False, cumulative = True,
#                       show_order_date = True,
#                       annotation = True, signature = False, 
#                       save = True, dashboard = False):
        
    mpl_uri = test.plot_prevalence(yscale='linear', per_capita=False, delta_ts=True,
                        window=[11], plot_dt = False, cumulative = True,
                        show_order_date = False,
                        annotation = True, signature = True, 
                        save = False, dashboard = True)
             

#   print(mpl_uri)

    if (n_clicks < 1):
        return ('Please click Submit')
    else:
        return mpl_uri
#       return u'''
#           The Button has been pressed {} times,
#           geography is "{}",
#           surrounded by "{}".\n
#           Draw prevalence plot for "{}"
#       '''.format(n_clicks, gname, surrounding, moniker)


if __name__ == '__main__':
    app.run_server(debug=True)


