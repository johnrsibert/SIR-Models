import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import base64
import flask

import pandas as pd
import js_covid as cv
import cv20

cv.population_dat = pd.read_csv(cv.census_data_path,header=0,comment='#')

image_filename = 'default.png'
static_image_route = '/static/'

app = dash.Dash()

app.layout = html.Div([
    html.Div(['Geography: ',
              dcc.Input(id='Geog-state', type='text', value='Los Angeles')]),
    html.Div(['Surrounded By: ',
              dcc.Input(id='SurroundedBy-state', type='text', value='California')]),
    html.Button(id='submit-button-state', n_clicks=0, children='Submit'),
    html.Img(id='prevalence-graphic', style={'height':'4.5in', 'width':'6.5in'})
#   html.Img(id='prevalence-graphic',src='data:image/png;base64,{}'.format(base64.b64encode(
#            open(cv.graphics_path+image_filename, 'rb').read()).decode()), 
#            style={'height':'4.5in', 'width':'6.5in'})
])

@app.callback(Output('prevalence-graphic','src'),
              [Input('submit-button-state', 'n_clicks')],
              [State('Geog-state', 'value'),
               State('SurroundedBy-state', 'value')])
def update_image_src(value):
    return static_image_route + value



def update_output(n_clicks, gname, surrounding):
    if (n_clicks > 0):
        print('got',gname,surrounding)
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
           
            moniker = str(gname+code)
            moniker = moniker.replace(' ','_',5) 
            print('population:',population,'; area code:',code,'; fips:',fips,'; moniker:',moniker)
            test = cv20.Geography(name=gname,enclosed_by=surrounding,code=code)
            print('HERE 2')
            test.read_nyt_data()
            print('HERE 3')
            test.print_metadata()
            try:
                print('calling '+test.moniker+'.plot_prevalence(yscale="linear", per_capita=False, delta_ts=True,...')
                test.plot_prevalence(yscale='linear', per_capita=False, delta_ts=True,
                            window=[11], plot_dt = False, cumulative = True,
                            show_order_date = False,
                            annotation = True, signature = True, 
                            save = False, dashboard = True)
    
            except:
                print('plot_prevalence() failed for:')
                print(test.moniker,gname,surrounding)
    
        except:
            print('get_county_pop() failed for:')
            print(gname,surrounding)
            population = 0

        image_name = 'test.png'
        print('returning',image_name,cv.graphics_path)
     #  return flask.send_from_directory(image_directory, image_name)
        return flask.send_from_directory(cv.graphics_path, image_name)


#    moniker = str(gname+code)
#    moniker = moniker.replace(' ','_',5) 
#    print('population:',population,'; area code:',code,'; fips:',fips,'; moniker:',moniker)
#    test = cv20.Geography(name=gname,enclosed_by=surrounding,code=code)
#    print('HERE')
#    test.read_nyt_data()
#    print('HERE 1')
#    test.print_metadata()
#    print('calling test.plot_prevalence(yscale="linear", per_capita=False, delta_ts=True,...')
#    test.plot_prevalence(yscale='linear', per_capita=False, delta_ts=True,
#                        window=[11], plot_dt = False, cumulative = True,
#                        show_order_date = False,
#                        annotation = True, signature = True, 
#                        save = False, dashboard = True)


if __name__ == '__main__':
    app.run_server(debug=True)
