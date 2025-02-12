'''
Author: Brian Duggan
Date: 28 March 2023
A dashboard app allowing the user to to select and visualize stock prices over a selected time period.
Based on the selection, investments can be calculated and the price of a portfolio, given an asset allocation, can be visualized over time.

To add:
- while scrolling through ticker, add bar to make it faster
- initial allocation values while maintaining selection of stocks
'''

import pandas as pd
import pandas_datareader.data as web
import pandas_datareader.tiingo as tingo
import numpy as np
from datetime import datetime, date, timedelta
from dash import dash, dash_table
from dash import dcc
from dash import html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc


tic_symbols = pd.read_csv('supported_tickers.csv') #Initial supported ticker symbols
tic_symbols.set_index('ticker',inplace=True)
options = list(tic_symbols.index)   #dataframe containing stock symbol options
price_data = [] #Dataframe for storing stock data

#Default values for Dashboard
default_stock_symbols = ["TSLA", "CS"]
default_allocation = {
  "Stock symbol": default_stock_symbols,
  "Allocation": [1, 99]
  }
default_allocation = pd.DataFrame(default_allocation)
default_allocation = default_allocation.to_dict('records')


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MINTY],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}]
                )

server=app.server

#Layout section
# ------------------------------------------------------------------------------------------------------------------
app.layout = dbc.Container([
    dbc.Row([#Header Row
        dbc.Col(#Single column containing title
            html.H1(
                "Stock Dashboard",
                className='text-center mb-4'
            ),
            width=12
        )
    ]),

    dbc.Row([#Row for stock selector, date picker
        dbc.Col([#Col for stock symbol selection
            html.H5('Enter a stock symbol'),
            dcc.Dropdown(
                        id='my_stock_picker',
                        options = options,
                        value=default_stock_symbols,
                        multi = True
            ),
        ],
            xs=12, sm=12, md=12, lg=5, xl=5,className='mt-4'
        ),

        dbc.Col([#Col for selecting start and end date
            html.H5('Select a start and end date'),
            dmc.DateRangePicker(
                id='my_date_picker',
                minDate=date(1950,1,1),
                value=[datetime(2018,1,1), datetime.today()],
            )
        ],width={'offset':2},
            xs=12, sm=12, md=12, lg=5, xl=5, className='mt-4'
        ),

        dbc.Col(#Col for submit button
            html.Button(id='submit-button',
                        n_clicks=0,
                        children='Submit',
                        #style={'fontSize':24, 'marginLeft':'30px'},
                        className='btn btn-primary btn-lg'
            ),
            xs=2, sm=2, md=2, lg=1, xl=1, className='mt-4'
        )

    ], align = 'end'),

    dbc.Row(#Row containing graph displaying stock value over time
        dbc.Col(
            dmc.LoadingOverlay(
                dcc.Graph(id='my_graph',
                            figure={
                                'data':[
                                    {'x':[1,2],'y':[3,1]}
                                ],
                                'layout':{
                                    'title': 'Stock closing prices over time',
                                    'template':'simple_white'
                                }
                            }
                )
            )
            , width={'size':12}, className='mt-5 g-0'
        )
    ),

    dbc.Row([#Title of investment calculator section
        dbc.Col(
            html.H2('Investment calculator'),
        )
    ]),

    dbc.Row([#Input investments and allocation
        dbc.Col([#Input initial investment
            html.H5('Initial investment'),
            dcc.Input(
                id='initial_investment',
                placeholder='USD',
                type='number',
                value=10000
            )
        ],
            width={'offset':2},
            xs=12, sm=12, md=12, lg=3, xl=3, className='mt-4'
        ),

        dbc.Col([#Input monthly investment
            html.H5('Monthly investment'),
            dcc.Input(
                id='monthly_investment',
                placeholder='USD',
                type='number',
                value=200
            )
        ],
            width={'offset':2},
            xs=12, sm=12, md=12, lg=3, xl=3, className='mt-4'
        ),

        dbc.Col([#Input desired allocation
            dbc.Row(html.H5('Enter allocation in %')),
            dbc.Row( ## Allocation table
                    dash_table.DataTable(#Default table
                        id='allocation_table',
                        columns = [
                            {
                                'name':'Stock symbol',
                                'id':'Stock symbol',
                            },
                            {
                                'name':'Allocation',
                                'id':'Allocation',
                                'type':'numeric',
                                'format':{"specifier": ",.0f"},
                                "editable": True,
                                "on_change": {"failure": "default"},
                                "validation": {"default": 0}
                            }
                        ],
                        data = default_allocation,
                        fill_width=False
                    )
            )
        ],
            width={'offset':2},
            xs=12, sm=12, md=12, lg=3, xl=3, className='mt-4'
        ),

        dbc.Col(#Button to initiate calculation
                dmc.HoverCard(#Hovercard when hovering with mouse over the button gives information
                    withArrow=True,
                    width=150,
                    shadow="md",
                    children=[
                        dmc.HoverCardTarget(#contains the button to calculate value
                            html.Button(id='calculate-button',
                                        n_clicks=0,
                                        children='Calculate',
                                        className='btn btn-primary btn-lg'
                            ),
                        ),
                        dmc.HoverCardDropdown(#Text discplayed in the hovercard
                            dmc.Text(
                                "Click to discover historical and current value of your portfolio",
                                size="sm",
                            )
                        ),
                    ],
                ),
            width={'offset':4},
            xs=2, sm=2, md=2, lg=1, xl=1, className='mt-4'
        )
    ]),

    dbc.Row([#Resulting final value of investment
        dbc.Col([
            html.H5('Investment value today'),
            html.H3(id='investment_output', className='text-success')
        ],
        width=12, className='mt-4'
        ),
    ]),

    dbc.Row([#Container for Portfolio value over time
        dbc.Col(html.H5('Portfolio value over time'), width=12),

        dbc.Col(
            dmc.LoadingOverlay(
                dcc.Graph(
                id='portfolio_growth'
                )
            )
            ,width=12
        )
    ])


])

#Callback section
# ------------------------------------------------------------------------------------------------------------------
# Callback function for calculation of investment growth over time
@app.callback(  [
                    Output('investment_output', 'children'),
                    # Output('summary_table','children'),
                    Output('portfolio_growth','figure')
                ],
                [
                    Input('calculate-button', 'n_clicks')
                ],
                [
                    State('initial_investment', 'value'),
                    State('monthly_investment', 'value'),
                    State('allocation_table', 'data'),
                ]
            )
def display_growth(n_clicks, initial_investment, month_invest, allocation):
    alloc_df = pd.DataFrame(allocation)#check allocation equals 100%

    #Create a summary table with the information of each stock needed to calculate returns
    summary_df = pd.DataFrame(columns = ['stock','start_date','initial_price','end_date','end_price','allocation(%)', 'Shares (initial)', 'Shares (monthly)','value ($)'])
    daily_value_df = pd.DataFrame(columns = ['symbol','date','adjClose','daily_value'])
    traces = [] #traces for plotting
    i = 0   #creating summary_df row by row

    for tic in price_data['symbol'].unique():
        price_at_start = round(price_data[price_data['symbol']==tic].iloc[0]['adjClose'],2)
        price_at_end = price_data[price_data['symbol']==tic].iloc[-1]['adjClose']
        start_date = price_data[price_data['symbol']==tic].iloc[0]['date']
        end_date = price_data[price_data['symbol']==tic].iloc[-1]['date']

        tic_allocation = alloc_df[alloc_df['Stock symbol']==tic]['Allocation'].iloc[0]

        #initial investment calculation value
        init_shares = (initial_investment*tic_allocation/100)/price_at_start

        #Monthly investment calculation (start off with initial investment)
        monthly_shares = init_shares

        #Get a monthly price, assuming a monthly investment is made
        price_data['year'] = price_data['date'].apply(lambda n: n.year)
        price_data['month'] = price_data['date'].apply(lambda n: n.month)
        price_data['day'] = price_data['date'].apply(lambda n: n.day)

        for (year, month),data in price_data[price_data['symbol']==tic].groupby(['year','month']):
            month_price = data['adjClose'].iloc[0]
            monthly_shares += (month_invest*tic_allocation/100)/month_price

            #Calculate the value of the shares held for every stock for every day_data
            data['daily_value'] = np.vectorize(daily_value)(monthly_shares,data['adjClose'])
            daily_value_df = pd.concat([daily_value_df,data[['symbol','date','adjClose','daily_value']]])

        #Shareprice_value calculation
        value = round(price_at_end*(monthly_shares),2)


        summary_df.loc[i] = [tic, start_date, price_at_start, end_date, price_at_end, tic_allocation, round(init_shares,2), round(monthly_shares,2), value]
        i += 1

    #Figure of portfolio growth -> check jupyter notebook on how to make the portfolio growth graph
    traces.append({'x': daily_value_df.groupby('date').sum(value).reset_index()['date'],
                    'y': daily_value_df.groupby('date').sum(value).reset_index()['daily_value'], 'name': 'Portfolio value'})

    #Creating a figure object to plot
    fig = {
        'data': traces,
        'layout': go.Layout(
        yaxis={'title':'Value in USD'},
        template = 'simple_white'
        )
    }

    cash_out_value = summary_df['value ($)'].sum()

    # return round(cash_out_value,2), dash_table.DataTable(summary_df.to_dict('records')),fig
    return "{} USD".format(round(cash_out_value,2)),fig


# Callback function to update the graph based on the stock ticker chosen
# Also generates allocation table depending on stocks chosen
@app.callback(  [
                Output('my_graph','figure'),
                Output('allocation_table','data')
                ],
                [Input('submit-button', 'n_clicks')],
                [State('my_date_picker', 'value'),
                State('my_stock_picker', 'value')
                ])
def update_graph(n_clicks, date_range, stock_ticker=default_stock_symbols):

    ### ---------------UPDATE GRAPH FOR STOCK TICKER VALUE-------------------------------------------------------------------------------------
    # the callback gets it as a string, therefore needs to be reconverted to datetime for the api call
    start_date = date_range[0]
    end_date = date_range[1]
    start = datetime.strptime(start_date[:10],'%Y-%m-%d')
    end = datetime.strptime(end_date[:10],'%Y-%m-%d')
    global price_data
    price_data = []

    traces = []
    for tic in stock_ticker:
        df = web.get_data_tiingo(tic, start, end, api_key = '7105aa11ad28bc8fa37d405d829d00adc66d910d')
        df.reset_index(inplace=True)
        price_data.append(df)
        traces.append({'x': df[df['symbol']==tic]['date'], 'y': df[df['symbol']==tic]['adjClose'], 'name': tic})

    price_data = pd.concat(price_data)

    fig = {
        'data': traces,
        'layout': go.Layout(
        yaxis={'title':'Stock Prices in USD'},
        title='Stock closing prices over time',
        template = 'simple_white'
        ),
    }

    ### ---------------CREATE ALLOCATION TABLE BASED ON SELECTION-------------------------------------------------------------------------------------
    initial_alloc_table = pd.DataFrame(columns = ['Stock symbol','Allocation'])
    initial_alloc_table['Stock symbol']=stock_ticker

    return fig, initial_alloc_table.to_dict('records')

#Other functions
# ------------------------------------------------------------------------------------------------------------------
def daily_value(monthly_shares,day_price):
    return monthly_shares*day_price
  
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 3000))
    app.run_server(debug=False, host="0.0.0.0", port=port)
