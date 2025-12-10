'''
Author: Brian Duggan
Date: 28 March 2023
Updated for Vercel:
- Compatible with Dash Mantine Components 0.14+ (DatePickerInput)
- Uses yfinance for reliable data fetching
- Fixes DataFrame boolean error
'''

from pathlib import Path
from typing import Optional
import pandas as pd
import os
import yfinance as yf  
import numpy as np
from datetime import datetime, date, timedelta
from dash import Dash, dash_table, dcc, html, Input, Output, State, _dash_renderer
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc

# 1. Set React version for DMC 0.14+ compatibility
_dash_renderer._set_react_version("18.2.0")

BASE_DIR = Path(__file__).resolve().parent
TICKER_FILE = BASE_DIR / "supported_tickers.csv"
TICKER_REFRESH_INTERVAL = timedelta(days=1)
DEFAULT_STOCK_SYMBOLS = ["TSLA", "AAPL"]
price_cache = {}


def empty_figure(message: str = "No data to display yet"):
    """Simple empty-state figure with a centered note."""
    return {
        "data": [],
        "layout": go.Layout(
            title=message,
            template="simple_white",
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[{
                "text": message,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 14}
            }],
        ),
    }


def format_currency(value: float) -> str:
    """Format numbers as currency strings."""
    try:
        return f"${value:,.2f}"
    except Exception:  # noqa: BLE001
        return "$0.00"


def build_stats_cards(total_contributions: float, current_value: float, gain: float, ret_pct: float):
    """Render quick stats cards for the portfolio view."""
    metrics = [
        ("Total contributed", format_currency(total_contributions)),
        ("Current value", format_currency(current_value)),
        ("Net gain / loss", format_currency(gain)),
        ("Return vs contributions", f"{ret_pct:,.2f}%"),
    ]
    cards = []
    for label, display in metrics:
        cards.append(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.Small(label, className="text-muted"),
                        html.H5(display, className="mt-2"),
                    ]),
                    className="h-100 shadow-sm border-0"
                ),
                sm=6, lg=3
            )
        )
    return dbc.Row(cards, className="g-3")


def fetch_supported_ticker_list() -> Optional[pd.DataFrame]:
    """
    Pull ticker symbols from multiple yfinance sources to keep the dropdown current.
    """
    ticker_sources = [
        ("sp500", getattr(yf, "tickers_sp500", None)),
        ("nasdaq", getattr(yf, "tickers_nasdaq", None)),
        ("other", getattr(yf, "tickers_other", None)),
    ]

    symbols = set()
    for name, func in ticker_sources:
        if not callable(func):
            continue
        try:
            symbols.update(func())
        except Exception as exc:  # noqa: BLE001
            print(f"[ticker-refresh] {name} fetch failed: {exc}")

    cleaned_symbols = sorted({sym.strip().upper() for sym in symbols if isinstance(sym, str) and sym.strip()})
    return pd.DataFrame({"ticker": cleaned_symbols}) if cleaned_symbols else None


def load_supported_tickers(force_refresh: bool = False) -> pd.DataFrame:
    """
    Load supported tickers from disk, refreshing them if the file is stale or missing.
    """
    should_refresh = force_refresh or not TICKER_FILE.exists()
    if TICKER_FILE.exists() and not force_refresh:
        last_modified = datetime.fromtimestamp(TICKER_FILE.stat().st_mtime)
        should_refresh = datetime.now() - last_modified > TICKER_REFRESH_INTERVAL

    refreshed_df = None
    if should_refresh:
        refreshed_df = fetch_supported_ticker_list()
        if refreshed_df is not None:
            try:
                refreshed_df.to_csv(TICKER_FILE, index=False)
            except Exception as exc:  # noqa: BLE001
                print(f"[ticker-refresh] Failed to persist ticker file: {exc}")

    try:
        df = refreshed_df if refreshed_df is not None else pd.read_csv(TICKER_FILE)
    except Exception as exc:  # noqa: BLE001
        print(f"[ticker-refresh] Falling back to defaults: {exc}")
        df = pd.DataFrame({"ticker": DEFAULT_STOCK_SYMBOLS})

    if "ticker" not in df.columns:
        df = pd.DataFrame({"ticker": DEFAULT_STOCK_SYMBOLS})

    df = pd.concat([df, pd.DataFrame({"ticker": DEFAULT_STOCK_SYMBOLS})], ignore_index=True)
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df = df.dropna(subset=["ticker"]).drop_duplicates(subset=["ticker"])
    return df


def get_price_data(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    """
    Pull price data with a tiny in-memory cache to cut down on repeat downloads.
    """
    key = (ticker, start.date().isoformat(), end.date().isoformat())
    if key in price_cache:
        return price_cache[key].copy()

    df = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
        actions=False,
        group_by="column"
    )
    price_cache[key] = df.copy()
    return df

tic_symbols = load_supported_tickers()
tic_symbols.set_index('ticker', inplace=True)
options = list(tic_symbols.index)

price_data = [] 

# Default values
default_allocation = {
  "Stock symbol": DEFAULT_STOCK_SYMBOLS,
  "Allocation": [50, 50]
}
default_allocation_df = pd.DataFrame(default_allocation)
default_allocation_dict = default_allocation_df.to_dict('records')

app = Dash(__name__, external_stylesheets=[dbc.themes.MINTY],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}]
                )

server = app.server

# 2. Wrap layout in MantineProvider
app.layout = dmc.MantineProvider(
    dbc.Container([
        dcc.Interval(id='ticker-refresh', interval=24 * 60 * 60 * 1000, n_intervals=0),
        dbc.Row([
            dbc.Col(
                html.Div([
                    html.H1("Stock Dashboard", className='text-center mb-2'),
                    html.P("Track prices, test allocations, and see how your portfolio could grow over time.", className="text-center text-muted"),
                ]),
                width=12
            )
        ]),

        dbc.Row(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H5('Enter a stock symbol'),
                                dcc.Dropdown(
                                    id='my_stock_picker',
                                    options=options,
                                    value=DEFAULT_STOCK_SYMBOLS,
                                    multi=True
                                ),
                                dbc.Badge("Supported tickers refresh daily", color="info", className="mt-2")
                            ], xs=12, sm=12, md=12, lg=5, xl=5, className='mt-2'),

                            dbc.Col([
                                html.H5('Select a start and end date'),
                                dmc.DatePickerInput(
                                    id='my_date_picker',
                                    type="range",
                                    minDate=date(1950, 1, 1),
                                    value=[datetime(2018, 1, 1), datetime.today()],
                                    style={"width": "100%"}
                                )
                            ], xs=12, sm=12, md=12, lg=5, xl=5, className='mt-2'),

                            dbc.Col(
                                html.Button(id='submit-button',
                                            n_clicks=0,
                                            children='Load prices',
                                            className='btn btn-primary btn-lg w-100'),
                                xs=12, sm=12, md=12, lg=2, xl=2, className='mt-4'
                            )
                        ], align='end'),
                    ]),
                    className="shadow-sm border-0"
                ),
                width=12,
                className="mt-3"
            )
        ),

        dbc.Row(
            dbc.Col(
                dcc.Loading(
                    dcc.Graph(
                        id='my_graph',
                        figure=empty_figure("Select stocks and dates, then click Load prices")
                    )
                ), width={'size': 12}, className='mt-4 g-0'
            )
        ),

        dbc.Row([dbc.Col(html.H2('Investment calculator', className='mt-4'))]),

        dbc.Row(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H5('Initial investment'),
                                dcc.Input(id='initial_investment', placeholder='USD', type='number', value=10000, className="w-100")
                            ], xs=12, sm=12, md=12, lg=3, xl=3, className='mt-2'),

                            dbc.Col([
                                html.H5('Monthly investment'),
                                dcc.Input(id='monthly_investment', placeholder='USD', type='number', value=200, className="w-100")
                            ], xs=12, sm=12, md=12, lg=3, xl=3, className='mt-2'),

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
                                        fill_width=False,
                                        style_table={"overflowX": "auto"},
                                    )
                                )
                            ], xs=12, sm=12, md=12, lg=4, xl=4, className='mt-2'),

                            dbc.Col(
                                dmc.HoverCard(
                                    withArrow=True,
                                    width=200,
                                    shadow="md",
                                    children=[
                                        dmc.HoverCardTarget(
                                            html.Button(id='calculate-button', n_clicks=0, children='Calculate',
                                                        className='btn btn-success btn-lg w-100')
                                        ),
                                        dmc.HoverCardDropdown(
                                            dmc.Text("Run the calculator to see portfolio value and stats", size="sm")
                                        ),
                                    ],
                                ), xs=12, sm=12, md=12, lg=2, xl=2, className='mt-4'
                            )
                        ]),
                        dbc.Row(dbc.Col(dbc.Alert(id='calc_alert', is_open=False, color='warning', className='mt-3'))),
                        dbc.Row([
                            dbc.Col([
                                html.H5('Investment value today'),
                                html.H3(id='investment_output', className='text-success'),
                                html.Div(id='portfolio_stats', className='mt-3')
                            ], width=12, className='mt-3'),
                        ]),
                        dbc.Row([
                            dbc.Col(html.H5('Portfolio value over time'), width=12),
                            dbc.Col(
                                dcc.Loading(dcc.Graph(id='portfolio_growth', figure=empty_figure("Run the calculator to see growth"))),
                                width=12
                            )
                        ])
                    ]),
                    className="shadow-sm border-0"
                ),
                width=12,
                className="mt-3"
            )
        ),
    ], fluid=True)
)


@app.callback(
    Output('my_stock_picker', 'options'),
    Input('ticker-refresh', 'n_intervals')
)
def refresh_ticker_options(n_intervals):
    ticker_df = load_supported_tickers(force_refresh=bool(n_intervals))
    return ticker_df['ticker'].tolist()


@app.callback(
    [
        Output('investment_output', 'children'),
        Output('portfolio_growth', 'figure'),
        Output('calc_alert', 'is_open'),
        Output('calc_alert', 'children'),
        Output('calc_alert', 'color'),
        Output('portfolio_stats', 'children'),
    ],
    [Input('calculate-button', 'n_clicks')],
    [State('initial_investment', 'value'), State('monthly_investment', 'value'), State('allocation_table', 'data')]
)
def display_growth(n_clicks, initial_investment, month_invest, allocation):
    alert_open = False
    alert_text = ""
    alert_color = "warning"

    if not allocation:
        return "—", empty_figure("Add tickers and allocations to calculate"), alert_open, alert_text, alert_color, []

    alloc_df = pd.DataFrame(allocation)
    alloc_df['Allocation'] = pd.to_numeric(alloc_df['Allocation'], errors='coerce').fillna(0)
    initial_investment = initial_investment or 0
    month_invest = month_invest or 0

    alloc_sum = alloc_df['Allocation'].sum()
    if not np.isclose(alloc_sum, 100, atol=0.5):
        alert_text = f"Allocations must sum to 100%. Current total: {alloc_sum:,.1f}%."
        return "—", empty_figure(alert_text), True, alert_text, "warning", []

    if isinstance(price_data, list):
        if not price_data:
            return "—", empty_figure("Load prices first, then run the calculator"), True, "Load prices first, then run the calculator.", "info", []
        working_price_data = pd.concat(price_data)
    else:
        if price_data.empty:
            return "—", empty_figure("Load prices first, then run the calculator"), True, "Load prices first, then run the calculator.", "info", []
        working_price_data = price_data

    if working_price_data.empty:
        return "—", empty_figure("No price data available for this selection"), True, "No price data available for this selection.", "info", []

    summary_df = pd.DataFrame(columns=['stock', 'start_date', 'initial_price', 'end_date', 'end_price', 'allocation(%)', 'Shares (initial)', 'Shares (monthly)', 'value ($)'])
    daily_value_df = pd.DataFrame(columns=['symbol', 'date', 'adjClose', 'daily_value'])
    traces = []
    
    i = 0
    for tic in alloc_df['Stock symbol'].unique():
        if tic not in working_price_data['symbol'].values:
            continue
            
        tic_data = working_price_data[working_price_data['symbol'] == tic].copy()
        price_col = None
        if 'adjClose' in tic_data.columns:
            price_col = 'adjClose'
        elif 'Adj Close' in tic_data.columns:
            price_col = 'Adj Close'
        elif 'Close' in tic_data.columns:
            price_col = 'Close'
        elif 'close' in tic_data.columns:
            price_col = 'close'
        else:
            continue

        tic_data['date'] = pd.to_datetime(tic_data['date'])
        tic_data = tic_data.sort_values('date')
        tic_data[price_col] = pd.to_numeric(tic_data[price_col], errors='coerce')
        tic_data = tic_data.dropna(subset=[price_col])
        if tic_data.empty:
            continue

        price_at_start = float(tic_data.iloc[0][price_col])
        if price_at_start == 0:
            continue

        price_at_end = float(tic_data.iloc[-1][price_col])
        start_date = tic_data.iloc[0]['date']
        end_date = tic_data.iloc[-1]['date']

        tic_allocation = alloc_df[alloc_df['Stock symbol'] == tic]['Allocation'].iloc[0]

        init_shares = (initial_investment * tic_allocation / 100) / price_at_start
        monthly_shares = init_shares

        tic_data['year'] = tic_data['date'].apply(lambda n: n.year)
        tic_data['month'] = tic_data['date'].apply(lambda n: n.month)
        
        # Calculate monthly updates
        for (year, month), data in tic_data.groupby(['year', 'month']):
            month_price = data[price_col].iloc[0]
            if pd.isna(month_price) or month_price == 0:
                continue
            monthly_shares += (month_invest * tic_allocation / 100) / month_price
            
            vals = data[price_col] * monthly_shares
            temp_df = data[['symbol', 'date', price_col]].copy()
            temp_df.rename(columns={price_col: 'adjClose'}, inplace=True)
            temp_df['daily_value'] = vals
            if temp_df.empty:
                continue
            if daily_value_df.empty:
                daily_value_df = temp_df
            else:
                daily_value_df = pd.concat([daily_value_df, temp_df], ignore_index=True)

        value = round(price_at_end * monthly_shares, 2)
        summary_df.loc[i] = [tic, start_date, price_at_start, end_date, price_at_end, tic_allocation, round(init_shares, 2), round(monthly_shares, 2), value]
        i += 1

    if summary_df.empty:
        alert_text = "No data available for the selected tickers and dates."
        return "—", empty_figure(alert_text), True, alert_text, "info", []

    if not daily_value_df.empty:
        portfolio_daily = daily_value_df.groupby('date')['daily_value'].sum().reset_index()
        traces.append({'x': portfolio_daily['date'], 'y': portfolio_daily['daily_value'], 'name': 'Portfolio value'})

    fig = {
        'data': traces,
        'layout': go.Layout(yaxis={'title': 'Value in USD'}, template='simple_white')
    }

    cash_out_value = float(summary_df['value ($)'].sum())
    earliest_date = pd.to_datetime(working_price_data['date']).min()
    latest_date = pd.to_datetime(working_price_data['date']).max()
    months_elapsed = max(1, (latest_date.year - earliest_date.year) * 12 + latest_date.month - earliest_date.month + 1)
    total_contributions = float(initial_investment) + float(month_invest) * months_elapsed
    gain = cash_out_value - total_contributions
    return_pct = (cash_out_value / total_contributions - 1) * 100 if total_contributions > 0 else 0

    stats = build_stats_cards(total_contributions, cash_out_value, gain, return_pct)
    return format_currency(cash_out_value), fig, alert_open, alert_text, alert_color, stats

@app.callback(
    [Output('my_graph', 'figure'), Output('allocation_table', 'data')],
    [Input('submit-button', 'n_clicks')],
    [State('my_date_picker', 'value'), State('my_stock_picker', 'value')]
)
def update_graph(n_clicks, date_range, stock_ticker):
    if not stock_ticker:
        stock_ticker = DEFAULT_STOCK_SYMBOLS
        
    start_date = date_range[0]
    end_date = date_range[1]
    
    # Ensure dates are datetime objects
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
            df = get_price_data(tic, start, end)
            df.reset_index(inplace=True)
            if df.empty:
                continue

            # Flatten possible MultiIndex columns (e.g., ('Close', 'AAPL') -> 'Close')
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] if isinstance(col, tuple) and len(col) > 0 else col for col in df.columns]
            
            # Standardize column names
            if 'Adj Close' in df.columns:
                df.rename(columns={'Adj Close': 'adjClose'}, inplace=True)
            if 'Adj_Close' in df.columns:
                df.rename(columns={'Adj_Close': 'adjClose'}, inplace=True)
            if 'Close' in df.columns and 'adjClose' not in df.columns:
                df['adjClose'] = df['Close']
            if 'close' in df.columns and 'adjClose' not in df.columns:
                df['adjClose'] = df['close']
            if 'Date' in df.columns:
                df.rename(columns={'Date': 'date'}, inplace=True)
            if 'date' not in df.columns and 'Datetime' in df.columns:
                df.rename(columns={'Datetime': 'date'}, inplace=True)

            if 'adjClose' not in df.columns or 'date' not in df.columns:
                continue
                
            df['symbol'] = tic
            
            price_data.append(df)
            traces.append({'x': df['date'], 'y': df['adjClose'], 'name': tic})
        except Exception as e:
            print(f"Error fetching {tic}: {e}")

    if price_data:
        price_data = pd.concat(price_data)
    else:
        fig = empty_figure("No price data for this selection")
        initial_alloc_table = pd.DataFrame({'Stock symbol': stock_ticker, 'Allocation': [0]*len(stock_ticker)})
        if len(stock_ticker) == 2:
            initial_alloc_table['Allocation'] = [50, 50]
        return fig, initial_alloc_table.to_dict('records')

    fig = empty_figure("No price data for this selection") if not traces else {
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
