""""""  		  	   		 	 	 			  		 			 	 	 		 		 	
"""MC2-P1: Market simulator.  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	 	 			  		 			 	 	 		 		 	
Atlanta, Georgia 30332  		  	   		 	 	 			  		 			 	 	 		 		 	
All Rights Reserved  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
Template code for CS 4646/7646  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	 	 			  		 			 	 	 		 		 	
works, including solutions to the projects assigned in this course. Students  		  	   		 	 	 			  		 			 	 	 		 		 	
and other users of this template code are advised not to share it with others  		  	   		 	 	 			  		 			 	 	 		 		 	
or to make it available on publicly viewable websites including repositories  		  	   		 	 	 			  		 			 	 	 		 		 	
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	 	 			  		 			 	 	 		 		 	
or edited.  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
We do grant permission to share solutions privately with non-students such  		  	   		 	 	 			  		 			 	 	 		 		 	
as potential employers. However, sharing with other current or future  		  	   		 	 	 			  		 			 	 	 		 		 	
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	 	 			  		 			 	 	 		 		 	
GT honor code violation.  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
-----do not edit anything above this line---  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
Student Name: Joshua Lutkemuller (replace with your name)  		  	   		 	 	 			  		 			 	 	 		 		 	
GT User ID: jlutkemuller3 (replace with your User ID)  		  	   		 	 	 			  		 			 	 	 		 		 	
GT ID: 904051695 (replace with your GT ID)   		  	   		 	 	 			  		 			 	 	 		 		 	
"""  		  	   		 	 	 			  		 			 	 	 		 		 	

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  		  	   		 	 	 			  		 			 	 	 		 		 	
from util import get_data, plot_data
import datetime

def compute_portvals(
    trades_df,
    symbol='JPM',
    start_val=1000000,
    commission=0.0,
    impact=0.0 		  	   		 	 	 			  		 			 	 	 		 		 	):
    """
    Computes the portfolio values.

    :param trades_df: Trades DataFrame
    :type orders_file: str or file object
    :param start_val: The starting value of the portfolio
    :type start_val: int
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)
    :type commission: float
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction
    :type impact: float
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.
    :rtype: pandas.DataFrame
    """
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your

    symbol = trades_df.columns[0]
    dates = trades_df.index
    prices = get_data([symbol], dates)
    prices = prices[[symbol]].copy()
    prices['CASH'] = 1.0

    # fill
    prices.fillna(method='ffill', inplace=True)
    prices.fillna(method='bfill', inplace=True)

    # build trades df
    df_trades = pd.DataFrame(0.0, index=dates, columns=[symbol, 'CASH'])
    df_trades[symbol] = trades_df[symbol]

    # cash flows
    df_trades['CASH'] = - trades_df[symbol] * prices[symbol] * (1 + impact) - commission * (trades_df[symbol] != 0)
    # inject cash
    df_trades.loc[dates[0], 'CASH'] += start_val
    holdings = df_trades.cumsum()
    portvals = (holdings * prices).sum(axis=1)

    return portvals.to_frame(name='Portvals')

  		  	   		 	 	 			  		 			 	 	 		 		 	
def gtid():
    """
    :return: The GT ID of the student
    :rtype: int
    """
    return 904051695  # replace with your GT ID number

def study_group():
    """
    Returns
        A comma separated string of GT_Name of each member of your study group
        # Example: "gburdell3, jdoe77, tbalch7" or "gburdell3" if a single individual working alone

    Return type
        str
    """
    return "jlutkemuller3"

def author():
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    :return: The GT username of the student  		  	   		 	 	 			  		 			 	 	 		 		 	
    :rtype: str  		  	   		 	 	 			  		 			 	 	 		 		 	
    """
    return "jlutkemuller3"  # Change this to your user ID

def order_reader(order_file):

    """
    Read in an order file and return an ordered dataframe (default csv).

    """
    return pd.read_csv(order_file,parse_dates=['Date'],index_col='Date').sort_index()

def compute_bollinger_pb(prices: pd.Series, window: int = 20) -> pd.Series:
    """
    Compute Bollinger %B: (P - SMA) / (2 * std).
    Returns a single real-valued pd.Series.
    """
    sma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    bbp = (prices - sma) / (2 * std)
    return bbp


def compute_golden_cross(prices: pd.Series,
                         short_window: int = 50,
                         long_window:  int = 200) -> pd.Series:
    """
    Golden Cross indicator: SMA_short_window – SMA_long_window,
    computed internally from two SMAs. Positive values (crossing above zero)
    signal a Golden Cross; negative (crossing below zero) a Death Cross.

    Parameters
    ----------
    prices : pd.Series
        Adjusted-close prices indexed by date.
    short_window : int
        Lookback for the “fast” SMA (e.g. 50 days).
    long_window : int
        Lookback for the “slow” SMA (e.g. 200 days).

    Returns
    -------
    pd.Series
        Single-valued vector: difference between the two SMAs,
        indexed by the same dates as `prices`.
    """
    sma_fast_series = compute_sma(prices,window = short_window)
    sma_slow_series = compute_sma(prices,window = long_window)

    return sma_fast_series - sma_slow_series

def compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """
    RSI method using Wilder's method.

    Parameters
    ----------
    prices : pd.Series
        Adjusted-close prices indexed by date.
    window : int
        Lookback period for RSI calculation.

    Returns
    -------
    pd.Series
        RSI values (0–100) as a single real-valued vector.
    """
    # 1) Price changes
    delta = prices.diff()

    # 2) Separate gains and losses
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # 3) Wilder's smoothing: alpha = 1/N
    alpha = 1.0 / window
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

    # 4) Relative Strength & RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

def compute_keltner_channels(prices: pd.Series,
                             ema_window: int = 20,
                             atr_window: int = 10,
                             multiplier: float = 2.0) -> pd.DataFrame:
    """
    Compute Keltner Channels:
      - Middle line: EMA of price over `ema_window`
      - ATR: average true range approximated by mean absolute daily change
      - Upper band: EMA + multiplier * ATR
      - Lower band: EMA - multiplier * ATR

    Parameters
    ----------


    Returns
    -------
        DataFrame with columns ['middle', 'upper', 'lower'], indexed by date.  Leverages an EMA with
        a user-defined window and multiplier effect for the lower and upper bands.
    """
    # Middle line: EMA of price
    ema_series = prices.ewm(span=ema_window, adjust=False).mean()

    # True range approximation: abs difference of close
    tr = prices.diff().abs()
    atr = tr.rolling(window=atr_window, min_periods=atr_window).mean()

    # Channels
    upper = ema_series + multiplier * atr
    lower = ema_series - multiplier * atr

    return pd.DataFrame({
        'middle': ema_series,
        'upper': upper,
        'lower': lower
    })

def compute_macd(prices: pd.Series,
                 short_win: int = 12,
                 long_win: int = 26,
                 signal_win: int = 9) -> pd.Series:
    """
    Compute MACD histogram as a single real-valued vector

    Parameters
    ----------

    Returns
    -------
    pd.Series
        MACD histogram values, indexed by date.
    """
    ema_short = prices.ewm(span=short_win, adjust=False).mean()
    ema_long  = prices.ewm(span=long_win,  adjust=False).mean()
    macd_line = ema_short - ema_long
    signal    = macd_line.ewm(span=signal_win, adjust=False).mean()
    hist      = macd_line - signal
    return hist

def plot_macd_components(prices: pd.Series,
                         short_win: int = 12,
                         long_win: int = 26,
                         signal_win: int = 9,
                         symbol: str = "JPM"):
    """
    Plot MACD components in a single figure with three subplots:
      1) Price with short and long EMAs
      2) MACD line and signal line
      3) MACD histogram

    Parameters
    ----------
    prices :
        Prices indexed by date.
    short_win :
        Span for the short-term EMA.
    long_win :
        Span for the long-term EMA.
    signal_win :
        Span for the signal EMA.
    """
    # Compute EMAs and MACD components
    ema_short = prices.ewm(span=short_win, adjust=False).mean()
    ema_long  = prices.ewm(span=long_win,  adjust=False).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=signal_win, adjust=False).mean()
    histogram = macd_line - signal_line

    # Plot all components
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Top subplot: Price with EMAs
    ax1.plot(prices.index, prices, label='Price')
    ax1.plot(prices.index, ema_short, label=f'{short_win}-Day EMA')
    ax1.plot(prices.index, ema_long, label=f'{long_win}-Day EMA')
    ax1.set_title('Price with Short and Long EMAs')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True)

    # Middle subplot: MACD and Signal lines
    ax2.plot(macd_line.index, macd_line, label='MACD Line')
    ax2.plot(signal_line.index, signal_line, label='Signal Line')
    ax2.set_title('MACD and Signal Lines')
    ax2.set_ylabel('Value')
    ax2.legend()
    ax2.grid(True)

    # Bottom subplot: MACD Histogram
    ax3.bar(histogram.index, histogram, label='Histogram')
    ax3.axhline(0, linestyle='--')
    ax3.set_title('MACD Histogram')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Histogram')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig(f'./MACD_plot_{symbol}.png')
    plt.close()

def plot_keltner_channels(symbol_prices: pd.Series,
                          ema_window: int = 20,
                          atr_window: int = 10,
                          multiplier: float = 2.0,
                          symbol: str = "JPM") -> pd.DataFrame:
    """
    Plot price with Keltner Channels and return the channel DataFrame.

    Parameters
    ----------
    prices : pd.Series
        Adjusted-close prices indexed by date.
    ema_window : int
        Lookback period for EMA.
    atr_window : int
        Lookback period for ATR.
    multiplier : float
        ATR multiplier for the width of the bands.

    Returns
    -------
    pd.DataFrame
        Keltner Channels DataFrame with ['middle','upper','lower'].
    """
    keltChannels = compute_keltner_channels(symbol_prices, ema_window, atr_window, multiplier)

    plt.figure(figsize=(12, 6))
    plt.plot(symbol_prices, label='Price', color='blue')
    plt.plot(keltChannels['middle'], label=f'EMA {ema_window}', color='orange', linewidth=1.5)
    plt.plot(keltChannels['upper'], label=f'Upper = EMA + {multiplier}×ATR', color='green', linestyle='--')
    plt.plot(keltChannels['lower'], label=f'Lower = EMA - {multiplier}×ATR', color='red',   linestyle='--')
    plt.title(f'Keltner Channels (EMA={ema_window}, ATR={atr_window}, Mult={multiplier})')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(f'./keltnerChannels_plot_{symbol}.png')
    plt.close()

    return keltChannels
def compute_sma(prices: pd.Series, window: int = 20) -> pd.Series:
    """
    Compute Simple Moving Average (SMA).

    Parameters
    ----------
    prices : pd.Series
        Prices indexed by date.
    window :
        Lookback period for the SMA.

    Returns
    -------
    pd.Series
        The N-day SMA as a single real-valued vector.
    """
    # Compute SMA
    return prices.rolling(window=window, min_periods=window).mean()

def plot_bollinger_indicator(prices: pd.Series, window: int = 20, symbol: str = "JPM"):
    """
    Plots Bollinger Bands and %B for the given symbol over the date range.
    Top subplot: Price, SMA, Upper & Lower Bands
    Bottom subplot: %B indicator with ±0.5 reference lines
    """

    # Calculate SMA and standard deviation
    sma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    bbp = compute_bollinger_pb(prices, window)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top: Price and Bands
    ax1.plot(prices.index, prices, label='Price', color='blue')
    ax1.plot(sma.index,    sma,    label=f'SMA ({window})', color='orange')
    ax1.plot(upper.index,  upper,  label='Upper Band', color='green', linestyle='--')
    ax1.plot(lower.index,  lower,  label='Lower Band', color='red',   linestyle='--')
    ax1.set_title(f'{symbol} Price with {window}-Day Bollinger Bands')
    ax1.set_ylabel('Price ($)')
    ax1.legend(loc='best')
    ax1.grid(True)

    # Bottom: %B
    ax2.plot(bbp.index, bbp, label='%B', color='purple')
    ax2.axhline(0.5,  color='gray', linestyle='--', label='Upper Threshold (+0.5)') # Means +1 SD above SMA
    ax2.axhline(-0.5, color='gray', linestyle='--', label='Lower Threshold (-0.5)') # Means -1 SD below SMA
    ax2.set_title(f'{symbol} Bollinger %B Indicator')
    ax2.set_ylabel('%B')
    ax2.set_xlabel('Date')
    ax2.legend(loc='best')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f'./bollinger_plot_{symbol}.png')
    plt.close()

    return bbp

def indictators_runner(symbol="JPM",
                      sd=datetime.datetime(2008, 1, 1),
                      ed=datetime.datetime(2009, 12, 31)):

    """
    Plots a symbol’s adjusted-close price alongside its N-day simple moving average and upper/lower Bollinger Bands,
     with a secondary panel showing the normalized %B indicator (price’s position within the bands).
    """

    dates_range = pd.date_range(sd, ed)
    prices = get_data([symbol], dates_range)[symbol].ffill().bfill()

    # Call the bollinger plotting function
    plot_bollinger_indicator(prices, window=20, symbol=symbol)

    # Call the SMA plotting function
    plot_golden_cross(prices,short_window=50,long_window=200,symbol=symbol)

    # Call the RSI plotting function
    plot_rsi(prices, window=14, symbol=symbol)
    plot_additional_rsi_subplots(prices,window=14,symbol=symbol)

    # Call the Keltner Channels plotting function
    plot_keltner_channels(prices,symbol=symbol)

    # Call the MACD plotting function
    plot_macd_components(prices,symbol=symbol)

def plot_additional_rsi_subplots(prices: pd.Series, window: int = 14, symbol: str = "JPM"):
    # 1) Compute RSI and next-day returns
    rsi = compute_rsi(prices, window=window)
    next_returns = prices.shift(-1) / prices - 1

    # 2) Align into one DataFrame and drop any rows with NaN
    df = pd.DataFrame({
        'RSI': rsi,
        'NextRet': next_returns
    }).dropna()

    # 3) Plot 1: Histogram of RSI values
    plt.figure(figsize=(8, 4))
    plt.hist(df['RSI'], bins=20)
    plt.title(f'Histogram of {window}-Day RSI for {symbol}')
    plt.xlabel('RSI Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(f'rsiHistogram_{window}_{symbol}.png')
    plt.close()

    # 4) Plot 2: Next-Day Return vs. RSI scatter
    plt.figure(figsize=(8, 4))
    plt.scatter(df['RSI'], df['NextRet'], alpha=0.6)
    plt.title(f'Next-Day Return vs. {window}-Day RSI for {symbol}')
    plt.xlabel('RSI Value at Day t')
    plt.ylabel('Return on Day t+1')
    plt.grid(True)
    plt.savefig(f'priceReturn_vs_rsiScatter_{window}_{symbol}.png')
    plt.close()

def plot_rsi(prices: pd.Series, window: int = 14, symbol: str = 'JPM') -> pd.Series:
    """
    Plot the RSI indicator.

    Parameters
    ----------
    prices : pd.Series
        Adjusted-close prices indexed by date.
    window : int
        Lookback period for RSI calculation.

    Returns
    -------
    pd.Series
        RSI values.
    """

    # compute the RSI series
    rsi_series = compute_rsi(prices, window)

    plt.figure(figsize=(10, 5))
    plt.plot(rsi_series.index, rsi_series, label=f'RSI ({window})', color='purple')
    plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
    plt.axhline(30, color='green', linestyle='--', label='Oversold (30)')
    plt.title(f'{symbol} Relative Strength Index (RSI) - {window}-Day')
    plt.xlabel('Date')
    plt.ylabel('RSI Value')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(f'./rsi_plot_window{window}_{symbol}.png')
    plt.close()

    return rsi_series

def plot_sma_indicator(prices: pd.Series, window: int = 20, symbol:str = "JPM") -> pd.Series:

    sma_series_20 = compute_sma(prices, window)
    sma_series_50 = compute_sma(prices, 50)

    # Plot Price vs. SMA
    plt.figure(figsize=(10, 5))
    plt.plot(prices.index, prices, label='Price', color='blue')
    plt.plot(sma_series_20.index, sma_series_20, label=f'{window}-Day SMA', color='orange', linewidth=2)
    plt.plot(sma_series_50.index, sma_series_50, label=f'{50}-Day SMA', color='purple', linewidth=2)
    plt.title(f'{symbol} Price vs. {window}-Day Simple Moving Average')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(f'./sma_plot_window{window}_{symbol}.png')
    plt.close()

    return sma_series_20

def plot_golden_cross(prices: pd.Series,
                      short_window: int = 50,
                      long_window:  int = 200,
                      symbol: str = "JPM") -> pd.Series:
    """
    Plot price plus the two SMAs and the Golden Cross indicator below.

    Returns the same single-vector indicator from compute_golden_cross.
    """
    golden_cross_series = compute_golden_cross(prices, short_window, long_window)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # Top: Price + SMAs
    sma_fast = compute_sma(prices,window=short_window)
    sma_slow = compute_sma(prices,window=long_window)

    # Compute slopes
    slope_fast = sma_fast.diff()
    slope_slow = sma_slow.diff()

    ax1.plot(prices,    label='Price',          color='blue')
    ax1.plot(sma_fast,  label=f'SMA {short_window}', color='green')
    ax1.plot(sma_slow,  label=f'SMA {long_window}',  color='orange')
    ax1.set_title(f'Price with {short_window}- and {long_window}-Day SMAs')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True)

    # Bottom: Golden Cross (SMA_diff)
    ax2.plot(golden_cross_series, label='Fast SMA - Slow SMA', color='purple')
    ax2.axhline(0, color='gray', linestyle='--')
    ax2.set_title('Golden Cross Indicator\n(>0 = Golden Cross, <0 = Death Cross)')
    ax2.set_ylabel('SMA Difference')
    ax2.set_xlabel('Date')
    ax2.legend()
    ax2.grid(True)

    # Bottom panel: SMA slopes
    ax3.plot(slope_fast, label=f'{short_window}-Day SMA Slope', color='green')
    ax3.plot(slope_slow, label=f'{long_window}-Day SMA Slope', color='orange')
    ax3.axhline(0, color='gray', linestyle='--')
    ax3.set_title('SMA Slope Filter')
    ax3.set_ylabel('Daily Change')
    ax3.set_xlabel('Date')
    ax3.legend(loc='best')
    ax3.grid(True)
    plt.tight_layout()
    plt.savefig(f'./goldenCross_plot_{symbol}.png')
    plt.close()
    return golden_cross_series

def exploratory_stats_print(portvals):
    """Accepts a dataframe of portfolio values from the `compute_portvals` function in the `marketsimcode.py` module and returns basic metrics of the portfolio, sharpe ratio, cumulative returns,
    volatility, average daily return, and final portfolio value"""

    # daily returns
    daily_rets = portvals.pct_change().dropna()

    # number of trading days
    trading_days = portvals.shape[0] - 1 # one less than total rows because of compounding effect

    ann_ret = (portvals.iloc[-1, 0] / portvals.iloc[0, 0]) ** (252.0 / trading_days) - 1

    # cumulative return
    cum_ret = portvals.iloc[-1, 0] / portvals.iloc[0, 0] - 1

    # average daily return
    avg_daily_ret = daily_rets.mean()[0]

    # std dev of daily returns
    std_daily_ret = daily_rets.std()[0]

    # Sharpe ratio (assume rf=0, 252 trading days)
    sr_annualized = (avg_daily_ret / std_daily_ret) * (252 ** 0.5)

    # Ending Portfolio Value
    ending_port_val = portvals.iloc[-1, 0]

    print("\nDate Range: {} to {}".format(portvals.index[0], portvals.index[-1], end='\n'))
    print("\nStarting Portfolio Value: {}\n".format(portvals.iloc[0, 0]))
    print("Sharpe Ratio of Fund: {}".format(sr_annualized))
    print("Cumulative Return of Fund: {}".format(cum_ret))
    print("Daily Standard Deviation of Fund: {}".format(std_daily_ret))
    print("Average Daily Return of Fund: {}".format(avg_daily_ret))
    print("Annualized Return of Fund: {}".format(ann_ret))
    print("\nFinal Portfolio Value: {}\n".format(portvals.iloc[-1,0]))
