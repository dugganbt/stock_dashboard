from dash import dash, dash_table
from dash import dcc
from dash import html
import plotly.graph_objs as go
import pandas as pd
import pandas_datareader.data as web
import pandas_datareader.tiingo as tingo
from datetime import datetime
from numpy import random
import numpy as np
from dash.dependencies import Input, Output, State
import requests
from dateutil.relativedelta import relativedelta
import dash_bootstrap_components as dbc

##COMMENTS
##Take dividends into account?
##Integration of costs (TER)
##Change datepicker such that dates from many different years can be chosen easily
##Remove the time from the allocation overview table
##how to update the stock_ticker labels automatically, and only have the actual ticker and not numbers?

app = dash.Dash(external_stylesheets=[dbc.themes.MINTY])
server = app.server

tic_symbols = pd.read_csv('supported_tickers.csv')
tic_symbols.set_index('ticker',inplace=True)
options = list(tic_symbols.index)   #dataframe containing stock symbol options
price_data = [] #Dataframe for storing stock data


#### CODE TO GENERATE EXAMPLE DATA SET
#temporary dataframe acquisition to avoid API calls
# start = datetime(1995,1,2)
# end = datetime(2023,2,21)
# other token: 7105aa11ad28bc8fa37d405d829d00adc66d910d
# example_tics = ['TSLA','AAPL','MSFT','ASCN','VOO']
# for tic in example_tics:
#     df = web.get_data_tiingo(tic, start, end, api_key = '7105aa11ad28bc8fa37d405d829d00adc66d910d')
#     df.reset_index(inplace=True)
#     price_data.append(df)
#
# price_data = pd.concat(price_data)
# price_data.to_csv('example_stock_data.csv')
# df = web.get_data_tiingo('TSLA', start, end, api_key = 'b08c6021d2b9635edac117ef8347df3d684b33e0')
# df.to_csv('TSLA_data.csv')
# df = pd.read_csv('example_stock_data.csv')

app.layout = html.Div(
                [
                    # Header of the division
                    html.Br(),
                    html.H2('Stock ticker overview'),
                    html.Br(),
                    html.Div(
                        [
                            # Title for the stock symbol picker
                            html.H5('Enter a stock symbol:', style={'paddingRight':'20px'}),
                            dcc.Dropdown(
                                        id='my_stock_picker',
                                        options = options,
                                        value=["TSLA"],
                                        multi = True
                            ),

                        ],
                        style={
                            'display':'inline-block',
                            'verticalAlign':'top',
                             'width':'30%'
                             }
                    ),

                        # Division for the date choosing
                        html.Div(
                            [
                                html.H5('Select a start and end date:'),
                                dcc.DatePickerRange(id='my_date_picker',
                                                    min_date_allowed=datetime(2015,1,1),
                                                    max_date_allowed=datetime.today(),
                                                    start_date=datetime(2018,1,1),
                                                    end_date = datetime.today()
                                )
                            ],
                            style={
                                'display':'inline-block'
                            }
                        ),

                        # adding a submit button upon which the information updates
                        html.Div(
                            [
                            html.Button(id='submit-button',
                                        n_clicks=0,
                                        children='Submit',
                                        style={'fontSize':24, 'marginLeft':'30px'})

                            ],
                            style={
                                'display':'inline-block'
                            }
                        ),

                    # The time series graph showing the price of the stock
                    dcc.Graph(id='my_graph',
                                figure={
                                    'data':[
                                        {'x':[1,2],'y':[3,1]}
                                    ],
                                    'layout':{
                                        'title': 'Default Title'
                                    }
                                }
                    ),

                    html.Br(),
                    html.H4('Investment calculator'),
                    html.Div(## Investment calculation
                        [
                            html.Div(## Input initial investment
                                [
                                    html.H5('Initial investment ($)', style={'paddingRight':'20px'}),
                                    dcc.Input(
                                        id='initial_investment',
                                        placeholder='Enter a value...',
                                        type='number',
                                        value=''
                                    ),
                                ],
                                style={
                                    "display": "inline-block",
                                    "width": "20%"
                                }
                            ),

                            html.Div(## Input monthly investment
                                [
                                    html.H5('Monthly investment ($)', style={'paddingRight':'20px'}),
                                    dcc.Input(
                                        id='monthly_investment',
                                        placeholder='Enter a value...',
                                        type='number',
                                        value=''
                                    )
                                ],
                                style={
                                    'display':'inline-block',
                                    "width": "20%"
                                }
                            ),
                        ]
                    ),

                    html.Br(),
                    html.H5('Input desired allocation of investment'),
                    html.Div( ## Allocation table
                        id='allocation_table_container'
                    ),

                    html.Br(),
                    html.Div(## calculate investment button
                        [
                        html.Button(id='calculate-button',
                                    n_clicks=0,
                                    children='Calculate',
                                    style={'fontSize':24})

                        ],
                        style={
                            'display':'inline-block',
                            "width": "20%"
                        }
                    ),
                    html.Br(),

                    html.Div( ## Result of calculation
                        [
                            html.H4('Resulting assets ($)', style={'paddingRight':'20px'}),
                            html.Div(id='investment_output', style={'paddingRight':'20px'})
                        ]
                    ),

                    html.Br(),
                    html.Div( ## Div for Summary table of chosen assets and allocation
                        [
                            html.H4('Allocation overview'),
                            html.Div( ## Table of allocation
                                id='summary_table'
                            ),
                            html.Br(),
                            html.Div( ## Plot of portfolio value over time
                                [
                                    html.H4('Portfolio value over time'),
                                    dcc.Graph(
                                    id='portfolio_growth'
                                    )
                                ]
                            )

                        ]
                    )
                ]
            )

# Callback function for calculation of investment growth over time and summary table
@app.callback(  [
                    Output('investment_output', 'children'),
                    Output('summary_table','children'),
                    Output('portfolio_growth','figure')
                ],
                [
                    Input('calculate-button', 'n_clicks')
                ],
                [
                    State('initial_investment', 'value'),
                    State('monthly_investment', 'value'),
                    State('allocation_table', 'data')
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

        tic_allocation = alloc_df[alloc_df['symbol']==tic]['alloc'].iloc[0]

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
        print(summary_df.tail())
        i += 1

    #Figure of portfolio growth -> check jupyter notebook on how to make the portfolio growth graph
    traces.append({'x': daily_value_df.groupby('date').sum(value).reset_index()['date'],
                    'y': daily_value_df.groupby('date').sum(value).reset_index()['daily_value'], 'name': 'Portfolio value'})

    #Creating a figure object to plot
    fig = {
        'data': traces,
        'layout': go.Layout(
        # title = ', '.join(tic_name) +' ('+ ', '.join(stock_ticker) +')' + ' Closing Prices',
        yaxis={'title':'Value ($)'})
        }

    cash_out_value = summary_df['value ($)'].sum()

    return round(cash_out_value,2), dash_table.DataTable(summary_df.to_dict('records')),fig


# Callback function to update the graph based on the stock ticker chosen
# Also generates allocation table depending on stocks chosen
@app.callback(  [
                Output('my_graph','figure'),
                Output('allocation_table_container','children')
                ],
                [Input('submit-button', 'n_clicks')],
                [State('my_stock_picker', 'value'),
                State('my_date_picker', 'start_date'),
                State('my_date_picker', 'end_date')
                ])
def update_graph(n_clicks, stock_ticker, start_date, end_date):

    ### ---------------UPDATE GRAPH FOR STOCK TICKER VALUE-------------------------------------------------------------------------------------
    # the callback gets it as a string, therefore needs to be reconverted to datetime for the api call
    start = datetime.strptime(start_date[:10],'%Y-%m-%d')
    end = datetime.strptime(end_date[:10],'%Y-%m-%d')
    global price_data
    price_data = []

    traces = []
    for tic in stock_ticker:
        df = web.get_data_tiingo(tic, start, end, api_key = 'b08c6021d2b9635edac117ef8347df3d684b33e0')
        df.reset_index(inplace=True)
        price_data.append(df)
        # df_meta = tingo.TiingoMetaDataReader(tic, start, end, api_key = 'b08c6021d2b9635edac117ef8347df3d684b33e0')
        # tic_name = df_meta.read().loc['name']
        # df.index = df.index.get_level_values('date')
        # traces.append({'x': df.index, 'y': df['close'], 'name': tic_name})
        traces.append({'x': df[df['symbol']==tic]['date'], 'y': df[df['symbol']==tic]['adjClose'], 'name': tic})

    price_data = pd.concat(price_data)

    fig = {
        'data': traces,
        'layout': go.Layout(
        # title = ', '.join(tic_name) +' ('+ ', '.join(stock_ticker) +')' + ' Closing Prices',
        yaxis={'title':'Stock Prices in USD'})
        }

    ### ---------------CREATE ALLOCATION TABLE BASED ON SELECTION-------------------------------------------------------------------------------------

    initial_alloc_table = pd.DataFrame(columns = ['symbol','alloc'])
    initial_alloc_table['symbol']=stock_ticker


    alloc_table = dash_table.DataTable(
        id='allocation_table',
        columns = [
            {
                'name':'Stock symbol',
                'id':'symbol',
            },
            {
                'name':'Allocation (%)',
                'id':'alloc',
                'type':'numeric',
                'format':{"specifier": ",.0f"},
                "editable": True,
                "on_change": {"failure": "default"},
            "validation": {"default": 0}
            }
        ],
        data = initial_alloc_table.to_dict('records')
    )

    return fig, alloc_table

def daily_value(monthly_shares,day_price):
    return monthly_shares*day_price


if __name__ == '__main__':
    app.run_server(debug=True)
