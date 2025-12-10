'''
Author: Brian Duggan
Date: 28 March 2023
Updated for Dash Mantine Components 0.14+ compatibility
'''

from pathlib import Path
import pandas as pd
import os
import pandas_datareader.data as web
import numpy as np
from datetime import datetime, date
from dash import Dash, dash_table, dcc, html, Input, Output, State, _dash_renderer
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc

# 1. FIX: Set React version for DMC 0.14+
_dash_renderer._set_react_version("18.2.0")

BASE_DIR = Path(__file__).resolve().parent
TIINGO_API_KEY = os.environ.get("TIINGO_API_KEY")
TICKER_FILE = BASE_DIR / "supported_tickers.csv"

# Load tickers
try:
    tic_symbols = pd.read_csv(TICKER_FILE)
    tic_symbols.set_index('ticker', inplace=True)
    options = list(tic_symbols.index)
except Exception as e:
    print(f"Warning: Could not load tickers from {TICKER_FILE}. Using defaults. Error: {e}")
    options = ["TSLA", "CS"]

price_data = [] 

# Default values
default_stock_symbols = ["TSLA", "CS"]
default_allocation = {
  "Stock symbol": default_stock_symbols,
  "Allocation": [1, 99]
}
default_allocation_df = pd.DataFrame(default_allocation)
default_allocation_dict = default_allocation_df.to_dict('records')

app = Dash(__name__, external_stylesheets=[dbc.themes.MINTY],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}]
                )

server = app.server

# Layout section
app.layout = dmc.MantineProvider(
    dbc.Container([
        dbc.Row([
            dbc.Col(
                html.H1("Stock Dashboard", className='text-center mb-4'),
                width=12
            )
        ]),

        dbc.Row([
            dbc.Col([
                html.H5('Enter a stock symbol'),
                dcc.Dropdown(
                    id='my_stock_picker',
                    options=options,
                    value=default_stock_symbols,
                    multi=True
                ),
            ], xs=12, sm=12, md=12, lg=5, xl=5, className='mt-4'),

            dbc.Col([
                html.H5('Select a start and end date'),
                dmc.DatePickerInput(
                    id='my_date_picker',
                    type="range",  
                    minDate=date(1950, 1, 1),
                    value=[datetime(2018, 1, 1), datetime.today()],
                    style={"width": "100%"}
                )
            ], width={'offset': 2}, xs=12, sm=12, md=12, lg=5, xl=5, className='mt-4'),

            dbc.Col(
                html.Button(id='submit-button',
                            n_clicks=0,
                            children='Submit',
                            className='btn btn-primary btn-lg'),
                xs=2, sm=2, md=2, lg=1, xl=1, className='mt-4'
            )
        ], align='end'),

        dbc.Row(
            dbc.Col(
                dcc.Loading(
                    dcc.Graph(id='my_graph',
                              figure={
                                  'data': [{'x': [1, 2], 'y': [3, 1]}],
                                  'layout': {'title': 'Stock closing prices over time', 'template': 'simple_white'}
                              })
                ), width={'size': 12}, className='mt-5 g-0'
            )
        ),

        dbc.Row([dbc.Col(html.H2('Investment calculator'))]),

        dbc.Row([
            dbc.Col([
                html.H5('Initial investment'),
                dcc.Input(id='initial_investment', placeholder='USD', type='number', value=10000)
            ], width={'offset': 2}, xs=12, sm=12, md=12, lg=3, xl=3, className='mt-4'),

            dbc.Col([
                html.H5('Monthly investment'),
                dcc.Input(id='monthly_investment', placeholder='USD', type='number', value=200)
            ], width={'offset': 2}, xs=12, sm=12, md=12, lg=3, xl=3, className='mt-4'),

            dbc.Col([
                dbc.Row(html.H5('Enter allocation in %')),
                dbc.Row(
                    dash_table.DataTable(
                        id='allocation_table',
                        columns=[
                            {'name': 'Stock symbol', 'id': 'Stock symbol'},
                            {'name': 'Allocation', 'id': 'Allocation', 'type': 'numeric',
                             'format': {"specifier": ",.0f"}, "editable": True}
                        ],
                        data=default_allocation_dict,
                        fill_width=False
                    )
                )
            ], width={'offset': 2}, xs=12, sm=12, md=12, lg=3, xl=3, className='mt-4'),

            dbc.Col(
                dmc.HoverCard(
                    withArrow=True,
                    width=150,
                    shadow="md",
                    children=[
                        dmc.HoverCardTarget(
                            html.Button(id='calculate-button', n_clicks=0, children='Calculate',
                                        className='btn btn-primary btn-lg')
                        ),
                        dmc.HoverCardDropdown(
                            dmc.Text("Click to discover historical and current value of your portfolio", size="sm")
                        ),
                    ],
                ), width={'offset': 4}, xs=2, sm=2, md=2, lg=1, xl=1, className='mt-4'
            )
        ]),

        dbc.Row([
            dbc.Col([
                html.H5('Investment value today'),
                html.H3(id='investment_output', className='text-success')
            ], width=12, className='mt-4'),
        ]),

        dbc.Row([
            dbc.Col(html.H5('Portfolio value over time'), width=12),
            dbc.Col(
                dcc.Loading(dcc.Graph(id='portfolio_growth')),
                width=12
            )
        ])
    ])
)

# Callback section
@app.callback(
    [Output('investment_output', 'children'), Output('portfolio_growth', 'figure')],
    [Input('calculate-button', 'n_clicks')],
    [State('initial_investment', 'value'), State('monthly_investment', 'value'), State('allocation_table', 'data')]
)
def display_growth(n_clicks, initial_investment, month_invest, allocation):
    if not allocation or not price_data:
        return "0 USD", {}
        
    alloc_df = pd.DataFrame(allocation)
    
    # Ensure all stocks in allocation are in price_data
    available_stocks = [df['symbol'].iloc[0] for df in price_data if not df.empty] if isinstance(price_data, list) else price_data['symbol'].unique()
    
    # Simple check to prevent errors on empty data
    if len(available_stocks) == 0:
        return "0 USD", {}

    summary_df = pd.DataFrame(columns=['stock', 'start_date', 'initial_price', 'end_date', 'end_price', 'allocation(%)', 'Shares (initial)', 'Shares (monthly)', 'value ($)'])
    daily_value_df = pd.DataFrame(columns=['symbol', 'date', 'adjClose', 'daily_value'])
    traces = []
    
    # Process price_data (Handling list vs concat DF)
    working_price_data = pd.concat(price_data) if isinstance(price_data, list) else price_data

    # Helper function
    def daily_value_func(monthly_shares, day_price):
        return monthly_shares * day_price

    i = 0
    for tic in alloc_df['Stock symbol'].unique():
        if tic not in working_price_data['symbol'].values:
            continue
            
        tic_data = working_price_data[working_price_data['symbol'] == tic].copy()
        if tic_data.empty: continue

        price_at_start = round(tic_data.iloc[0]['adjClose'], 2)
        price_at_end = tic_data.iloc[-1]['adjClose']
        start_date = tic_data.iloc[0]['date']
        end_date = tic_data.iloc[-1]['date']

        tic_allocation = alloc_df[alloc_df['Stock symbol'] == tic]['Allocation'].iloc[0]

        init_shares = (initial_investment * tic_allocation / 100) / price_at_start
        monthly_shares = init_shares

        tic_data['year'] = tic_data['date'].apply(lambda n: n.year)
        tic_data['month'] = tic_data['date'].apply(lambda n: n.month)

        temp_daily_values = []
        
        # Calculate monthly updates
        # Optimization: Don't loop rows if possible, but keeping logic similar to original for safety
        for (year, month), data in tic_data.groupby(['year', 'month']):
            month_price = data['adjClose'].iloc[0]
            monthly_shares += (month_invest * tic_allocation / 100) / month_price
            
            # Calculate value for these days
            # We must use the 'monthly_shares' current value for these specific days
            vals = data['adjClose'] * monthly_shares
            temp_df = data[['symbol', 'date', 'adjClose']].copy()
            temp_df['daily_value'] = vals
            daily_value_df = pd.concat([daily_value_df, temp_df])

        value = round(price_at_end * monthly_shares, 2)
        summary_df.loc[i] = [tic, start_date, price_at_start, end_date, price_at_end, tic_allocation, round(init_shares, 2), round(monthly_shares, 2), value]
        i += 1

    # Plotting
    if not daily_value_df.empty:
        # Sum daily values across all stocks
        portfolio_daily = daily_value_df.groupby('date')['daily_value'].sum().reset_index()
        traces.append({'x': portfolio_daily['date'], 'y': portfolio_daily['daily_value'], 'name': 'Portfolio value'})

    fig = {
        'data': traces,
        'layout': go.Layout(yaxis={'title': 'Value in USD'}, template='simple_white')
    }

    cash_out_value = summary_df['value ($)'].sum()
    return "{} USD".format(round(cash_out_value, 2)), fig

@app.callback(
    [Output('my_graph', 'figure'), Output('allocation_table', 'data')],
    [Input('submit-button', 'n_clicks')],
    [State('my_date_picker', 'value'), State('my_stock_picker', 'value')]
)
def update_graph(n_clicks, date_range, stock_ticker):
    if not stock_ticker:
        stock_ticker = default_stock_symbols
        
    if not TIINGO_API_KEY:
        # Return empty/default if no key (prevents crash on build)
        return {}, default_allocation_dict

    # Date parsing: dmc 0.14+ DatePickerInput usually returns ISO strings or None
    start_date = date_range[0]
    end_date = date_range[1]
    
    # Ensure dates are datetime objects (Tiingo needs datetime)
    if isinstance(start_date, str):
        start = datetime.fromisoformat(start_date).replace(tzinfo=None)
    else:
        start = start_date
        
    if isinstance(end_date, str):
        end = datetime.fromisoformat(end_date).replace(tzinfo=None)
    else:
        end = end_date

    global price_data
    price_data = []
    traces = []

    for tic in stock_ticker:
        try:
            df = web.get_data_tiingo(tic, start, end, api_key=TIINGO_API_KEY)
            df.reset_index(inplace=True)
            price_data.append(df)
            traces.append({'x': df[df['symbol'] == tic]['date'], 'y': df[df['symbol'] == tic]['adjClose'], 'name': tic})
        except Exception as e:
            print(f"Error fetching {tic}: {e}")

    if price_data:
        price_data = pd.concat(price_data)

    fig = {
        'data': traces,
        'layout': go.Layout(
            yaxis={'title': 'Stock Prices in USD'},
            title='Stock closing prices over time',
            template='simple_white'
        ),
    }

    initial_alloc_table = pd.DataFrame({'Stock symbol': stock_ticker, 'Allocation': [0]*len(stock_ticker)})
    # Set default allocation if 2 stocks
    if len(stock_ticker) == 2:
        initial_alloc_table['Allocation'] = [50, 50]

    return fig, initial_alloc_table.to_dict('records')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port, debug=False)