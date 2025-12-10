'''
Author: Brian Duggan
Date: 28 March 2023
Updated for Vercel:
- Compatible with Dash Mantine Components 0.14+ (DatePickerInput)
- Uses yfinance for reliable data fetching
'''

from pathlib import Path
import pandas as pd
import os
import yfinance as yf  
import numpy as np
from datetime import datetime, date
from dash import Dash, dash_table, dcc, html, Input, Output, State, _dash_renderer
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc

_dash_renderer._set_react_version("18.2.0")

BASE_DIR = Path(__file__).resolve().parent
TICKER_FILE = BASE_DIR / "supported_tickers.csv"

# Load tickers
try:
    tic_symbols = pd.read_csv(TICKER_FILE)
    tic_symbols.set_index('ticker', inplace=True)
    options = list(tic_symbols.index)
except Exception as e:
    # Fallback if CSV is missing or CS is delisted
    options = ["TSLA", "AAPL", "MSFT", "GOOGL"]

price_data = [] 

# Default values
default_stock_symbols = ["TSLA", "AAPL"]
default_allocation = {
  "Stock symbol": default_stock_symbols,
  "Allocation": [50, 50]
}
default_allocation_df = pd.DataFrame(default_allocation)
default_allocation_dict = default_allocation_df.to_dict('records')

app = Dash(__name__, external_stylesheets=[dbc.themes.MINTY],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}]
                )

server = app.server

# 2. CRITICAL: Wrap layout in MantineProvider for DMC 0.14+
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
                # 3. CRITICAL: Use DatePickerInput (type="range") instead of old DateRangePicker
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

@app.callback(
    [Output('investment_output', 'children'), Output('portfolio_growth', 'figure')],
    [Input('calculate-button', 'n_clicks')],
    [State('initial_investment', 'value'), State('monthly_investment', 'value'), State('allocation_table', 'data')]
)
def display_growth(n_clicks, initial_investment, month_invest, allocation):
    if not allocation or not price_data:
        return "0 USD", {}
        
    alloc_df = pd.DataFrame(allocation)
    
    # Safety check for empty price data
    working_price_data = pd.concat(price_data) if isinstance(price_data, list) else price_data
    if working_price_data.empty:
         return "0 USD", {}

    summary_df = pd.DataFrame(columns=['stock', 'start_date', 'initial_price', 'end_date', 'end_price', 'allocation(%)', 'Shares (initial)', 'Shares (monthly)', 'value ($)'])
    daily_value_df = pd.DataFrame(columns=['symbol', 'date', 'adjClose', 'daily_value'])
    traces = []
    
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
        
        # Calculate monthly updates
        for (year, month), data in tic_data.groupby(['year', 'month']):
            month_price = data['adjClose'].iloc[0]
            monthly_shares += (month_invest * tic_allocation / 100) / month_price
            
            vals = data['adjClose'] * monthly_shares
            temp_df = data[['symbol', 'date', 'adjClose']].copy()
            temp_df['daily_value'] = vals
            daily_value_df = pd.concat([daily_value_df, temp_df])

        value = round(price_at_end * monthly_shares, 2)
        summary_df.loc[i] = [tic, start_date, price_at_start, end_date, price_at_end, tic_allocation, round(init_shares, 2), round(monthly_shares, 2), value]
        i += 1

    if not daily_value_df.empty:
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
        
    start_date = date_range[0]
    end_date = date_range[1]
    
    # 4. FIX: Ensure dates are proper datetime objects
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

            df = yf.download(tic, start=start, end=end, progress=False)
            df.reset_index(inplace=True)
            
            # yfinance creates 'Adj Close', but our code uses 'adjClose'
            if 'Adj Close' in df.columns:
                df.rename(columns={'Adj Close': 'adjClose'}, inplace=True)
            elif 'Close' in df.columns:
                # Fallback if Adj Close isn't present
                df['adjClose'] = df['Close']
            
            # yfinance 'Date' column is usually correct, but ensure consistency
            if 'Date' in df.columns:
                df.rename(columns={'Date': 'date'}, inplace=True)
                
            df['symbol'] = tic
            
            price_data.append(df)
            traces.append({'x': df['date'], 'y': df['adjClose'], 'name': tic})
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
    if len(stock_ticker) == 2:
        initial_alloc_table['Allocation'] = [50, 50]

    return fig, initial_alloc_table.to_dict('records')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port, debug=False)