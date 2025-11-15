""""""
"""MC2-P1: Test Code.  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
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
import datetime as dt
import pandas as pd
from StrategyLearner import StrategyLearner
from experiment1 import experiment1_runner
from experiment2 import experiment2_runner
from indicators import (
    compute_bollinger_pb,
    compute_keltner_channels,
    compute_golden_cross,
    compute_rsi,
    compute_macd
)
import matplotlib.pyplot as plt

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

class ManualLearner(object):
    """
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.

    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output.
    :type verbose: bool
    :param impact: The market impact of each transaction, defaults to 0.0
    :type impact: float
    :param commission: The commission amount charged, defaults to 0.0
    :type commission: float
    """

    # constructor
    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        """
        Constructor method
        """
        self.verbose = verbose
        self.impact = impact
        self.commission = commission

    def study_group(self):
        """
        Returns
            A comma separated string of GT_Name of each member of your study group
            # Example: "gburdell3, jdoe77, tbalch7" or "gburdell3" if a single individual working alone

        Return type
            str
        """
        return "jlutkemuller3"

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "jlutkemuller3"  # Change this to your user ID

    def gtid(self):
        """
        :return: The GT ID of the student
        :rtype: int
        """
        return 904051695  # replace with your GT ID number

    # manual_strategy.py

    def testPolicy(self, symbol='JPM',
                   sd=dt.datetime(2008, 1, 1),
                   ed=dt.datetime(2009, 12, 31),
                   sv=100000):
        # 1) Fetch prices
        dates = pd.date_range(sd, ed)
        prices = get_data([symbol], dates)[symbol].ffill().bfill()
        n = len(prices)

        # 2) Compute indicator vectors
        bbp = compute_bollinger_pb(prices, window=20)  # Bollinger %B

        gc = compute_golden_cross(prices, short_window=50,  # Golden Cross
                                  long_window=200)
        rsi14 = compute_rsi(prices, window=14)  # RSI
        macd_h = compute_macd(prices, short_win=12,  # MACD Hist
                              long_win=26, signal_win=9)

        # 3) Prepare trades DataFrame
        trades = pd.DataFrame(0.0, index=prices.index, columns=[symbol])
        position = 0

        # 3) Loop over days *except the last*, compute signal for day t,
        #    then place the trade on day t+1
        for t in range(n - 1):
            # 3a) Compute your vote-based signal on day t
            votes = 0
            if bbp.iloc[t] < -0.5:
                votes += 1
            elif bbp.iloc[t] > +0.5:
                votes -= 1

            if gc.iloc[t] > 0:
                votes += 1
            elif gc.iloc[t] < 0:
                votes -= 1

            if rsi14.iloc[t] < 30:
                votes += 1
            elif rsi14.iloc[t] > 70:
                votes -= 1

            if macd_h.iloc[t] > 0:
                votes += 1
            elif macd_h.iloc[t] < 0:
                votes -= 1

            signal = 0
            if votes >= 2 and position <= 0:
                signal = 1  # want to go long
            elif votes <= -2 and position >= 0:
                signal = -1  # want to go short
            else:
                signal = 0  # move to cash/flat if signals are weak

            #
            # # if we need to change our current position, schedule on t+1
            # next_day = prices.index[t + 1]
            # target = signal * 1000
            # if target != position:
            #     trades.loc[next_day, symbol] = target - position
            #     position = target

            # 3b) Place the trade on the *next* trading day (t+1)
            if signal != 0:
                next_day = prices.index[t + 1]
                target = signal * 1000
                trades.loc[next_day, symbol] = target - position
                position = target

        # 4) Finally, on the very last day ensure we flatten out:
        trades.iloc[-1, 0] = -position

        return trades

    def compute_daily_returns(self, df):
        """
        Compute daily returns from a DataFrame or Series of portfolio values.
        Returns a DataFrame of daily returns, with the first day dropped.
        """
        # pct_change automatically does (df[t] / df[t-1]) - 1
        daily_returns = df.pct_change().iloc[1:]
        return daily_returns

    def plot_performance_table(self, portvals_b, portvals_m, label_b, label_m, title="Performance Metrics",
                               in_out_sample=None):
        """
        Compute CR, ADR, and SDDR for two portfolios and display as a table.

        Parameters
        ----------
        portvals_b : pd.DataFrame or pd.Series
            First portfolio values (single column) which is benchmark.
        portvals_m : pd.DataFrame or pd.Series
            Second portfolio values.
        label_b : str
            Name for the first portfolio (e.g. "Benchmark").
        label_m : str
            Name for the second portfolio (e.g. "Manual Strategy").
        title : str
            Title for the figure.

        Returns
        -------
        None (displays a Matplotlib table).
        """
        title = title + '(' + in_out_sample + ')'

        def compute_metrics(portvals):
            # daily returns
            rets = portvals.pct_change().dropna()
            cr = portvals.iloc[-1] / portvals.iloc[0] - 1
            adr = rets.mean()
            sddr = rets.std()
            return cr.values[0], adr.values[0], sddr.values[0]

        cr_b, adr_b, sddr_b = compute_metrics(portvals_b)
        cr_m, adr_m, sddr_m = compute_metrics(portvals_m)

        # Build DataFrame
        df = pd.DataFrame({
            label_b: [cr_b, adr_b, sddr_b],
            label_m: [cr_m, adr_m, sddr_m]
        }, index=["Cumulative Return", "Avg Daily Return", "Std Dev Daily Return"])

        # Format as percentages/decimals
        df = df.applymap(lambda x: f"{x:.6f}")

        # Plot table
        fig, ax = plt.subplots(figsize=(8, 3))  # make it wider/taller
        ax.axis('off')

        tbl = ax.table(
            cellText=df.values,
            rowLabels=df.index,
            colLabels=df.columns,
            cellLoc='center',
            loc='center'
        )

        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 2)  # stretch cells taller if needed

        plt.title(title, pad=20)  # pad title so it doesn't overlap
        plt.tight_layout()  # make room for everything

        # and when you save:
        plt.savefig(f'./{title}.png', bbox_inches='tight', dpi=150)
        plt.close(fig)

    def report_metrics(self, port_vals, strategy):
        """
        Print cumulative return, average daily return, and std dev of daily returns.
        """
        daily_rets = self.compute_daily_returns(port_vals)

        cr = port_vals.iloc[-1, 0] / port_vals.iloc[0, 0] - 1
        adr = daily_rets.mean().values[0]
        sddr = daily_rets.std().values[0]

        if self.verbose:
            print(f"Cumulative Return of {strategy}: {cr:.6f}")
            print(f"Average Daily Return of {strategy}: {adr:.6f}")
            print(f"Std Dev of Daily Returns of {strategy}: {sddr:.6f}")

    def save_plot_old(self, df, sell_line_flags, buy_line_flags, plot_title, in_out_sample=None):

        if in_out_sample is None:
            if self.verbose == True:
                print("Pass an `In` or `Out` value to `in_out_sample` variable for plotting")
                return
            else:
                return
        # set axis
        ax = df.plot(title=f"Manual Strategy vs Benchmark ({in_out_sample})", fontsize=12, color=['purple', 'red'])
        ax.set_xlabel("Date")
        ax.set_ylabel("Normalized Portfolio Value")

        for sell_dates in sell_line_flags:
            plt.axvline(x=sell_dates, color='black', linestyle='--', label='SHORT Entry Points')

        for buy_dates in buy_line_flags:
            plt.axvline(x=buy_dates, color='blue', linestyle='--', label='LONG Entry Points')

        plt.savefig(f'./{plot_title}')
        plt.close()

    def save_plot(self, df, sell_dates, buy_dates, plot_title, in_out_sample=None):
        """
        df: DataFrame with two columns [Benchmark, Manual Strategy], indexed by date, already normalized.
        sell_dates: DatetimeIndex of short-entry dates
        buy_dates:  DatetimeIndex of long-entry dates
        plot_title: filename to save
        in_out_sample: "In Sample" or "Out Sample" for title
        """
        if in_out_sample not in ("In Sample", "Out Sample"):
            if self.verbose:
                print("Please pass 'In Sample' or 'Out Sample' as in_out_sample.")
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        # plot the two series
        ax.plot(df.index, df.iloc[:, 0], color='purple', lw=2, label='Benchmark')
        ax.plot(df.index, df.iloc[:, 1], color='red', lw=2, label='Manual Strategy')

        # draw entry lines on the same axes
        for d in buy_dates:
            ax.axvline(d, color='blue', linestyle='--', alpha=0.7, label='LONG Entry')
        for d in sell_dates:
            ax.axvline(d, color='black', linestyle='--', alpha=0.7, label='SHORT Entry')

        # dedupe legend entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=10)

        ax.set_title(f"Manual Strategy vs Benchmark ({in_out_sample})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Normalized Portfolio Value")
        ax.set_xlim(df.index[0], df.index[-1])
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(f"./{plot_title}")
        plt.close(fig)

    def add_evidence(self, symbol='IBM', sd=dt.datetime(2008, 1, 1, 0, 0), ed=dt.datetime(2009, 1, 1, 0, 0),
                     sv=100000):
        """ Not Used"""
        pass

    def mannual_run_sample_data(self, sd=dt.datetime(2008, 1, 1),
                                ed=dt.datetime(2009, 12, 31), plot_title="", in_out_sample=None, symbol="JPM",
                                start_val=100000):

        ####
        # Manual Strategy
        ####
        df_trades = self.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=start_val)

        manual_portvals = compute_portvals(df_trades, symbol, start_val=start_val, commission=9.95, impact=0.005)

        # Call report metrics on Manual Strategy
        self.report_metrics(manual_portvals, plot_title)
        normalized_manual_strategy = manual_portvals / manual_portvals.iloc[0, :]

        ####
        # Benchmark
        ####
        dates = df_trades.index
        bench_trades = pd.DataFrame(0, index=dates, columns=[symbol])

        # Buy +1000 shares and hold
        bench_trades.iloc[0, 0] = 1000  # buy on first trading day

        # compute benchmark portfolio values
        benchmark_portvals = compute_portvals(bench_trades,
                                              start_val=start_val,
                                              commission=9.95, impact=0.005)

        # Call Report Metrics on Benchmark
        self.report_metrics(benchmark_portvals, 'Benchmark')

        normalized_benchmark = benchmark_portvals / benchmark_portvals.iloc[0, :]

        ####
        # Plot
        ####
        # rename columns
        normalized_manual_strategy.columns = ['Manual Strategy']
        normalized_benchmark.columns = ['Benchmark Strategy']
        df_temp = normalized_benchmark.join(normalized_manual_strategy)

        sell_lines = df_trades[df_trades.values < 0].index
        buy_lines = df_trades[df_trades.values > 0].index

        # Plot Figure showing entry points over time
        self.save_plot(df_temp, sell_lines, buy_lines, plot_title, in_out_sample)

        # Plot figure for performance table
        self.plot_performance_table(normalized_benchmark, normalized_manual_strategy, 'Benchmark', 'Manual Strategy',
                                    'Performance Metrics', in_out_sample)


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

def compute_metrics(portfolio_values):
    """
    Compute three performance metrics for the Strategy Learner
    1) Cumulative Return
    2) Sharpe Ratio (annualized) ** assumptions are a 0% risk-free rate with 252 trading days in a year
    3) Average Daily Return
    """
    # sort index to be safe
    portfolio_values = portfolio_values.sort_index()

    # daily return stream
    daily_rets = portfolio_values.pct_change().dropna()

    # cumulative return metric
    cr_metric = portfolio_values.iloc[-1,0] / portfolio_values.iloc[0,0] - 1

    # total length of days for annualization factor
    day_count = len(daily_rets)

    # annualized return metric
    ann_ret_metric = ((1+cr_metric) ** (252/day_count))-1

    # average daily returns
    adr_metric = daily_rets.mean()[0]

    # daily standard deviation of returns
    sdr_metric = daily_rets.std()[0]

    # annualized standard deviation of daily returns
    asdr_metric = sdr_metric * (np.sqrt(252))

    # annualized sharpe ratio
    asharpe_metric = (adr_metric/sdr_metric) * (np.sqrt(252)) if sdr_metric != 0.0 else 0.0

    return cr_metric, ann_ret_metric, adr_metric, sdr_metric, asdr_metric, asharpe_metric

def evaluate_strategy_learner(symbol="JPM",
                              train_sd=dt.datetime(2008, 1, 1),
                              train_ed=dt.datetime(2009, 12, 31),
                              test_sd=dt.datetime(2010, 1, 1),
                              test_ed=dt.datetime(2011, 12, 31),
                              sv=100000, bb_windows = [10,20,30],verbose=False):

    """
    Function designed to evaluate the strategy learner
    """
    port_results = []
    bm_results = []
    for w in bb_windows:
        # 1) Train the learner
        learner = StrategyLearner(verbose=verbose, impact=0.005, commission=9.95)
        learner.add_evidence(symbol=symbol,
                             sd=train_sd, ed=train_ed,
                             sv=sv)

        # 2) Test out-of-sample
        trades = learner.testPolicy(symbol=symbol,
                                    sd=test_sd, ed=test_ed,
                                    sv=sv)

        # 3) Simulate portfolio
        portvals = compute_portvals(trades,
                                    start_val=sv,
                                    commission=0.0,
                                    impact=0.0)
        portvals.columns = ["StrategyLearner"]

        # Extract metrics from Strategy Learner
        cr_metric, ann_ret_metric, adr_metric, sdr_metric, asdr_metric, asharpe_metric = compute_metrics(portvals)

        port_results.append({'% B Window': w,
                        'Cumulative Return': cr_metric,
                        'Annualized Return': ann_ret_metric,
                        'Average Daily Return': sdr_metric,
                        'Daily Standard Deviation': sdr_metric,
                        'Annualized Standard Deviation': asdr_metric,
                        'Annualized Sharpe Ratio': asharpe_metric})


        # 4) Build benchmark (buy 1000 at first day, hold)
        dates = trades.index
        bench_trades = pd.DataFrame(0, index=dates, columns=[symbol])
        bench_trades.iloc[0, 0] = 1000
        bench_vals = compute_portvals(bench_trades,
                                      start_val=sv,
                                      commission=0.0,
                                      impact=0.0)
        bench_vals.columns = ["Benchmark"]

        # Extract metrics from Strategy Learner
        cr_metric, ann_ret_metric, adr_metric, sdr_metric, asdr_metric, asharpe_metric = compute_metrics(bench_vals)

        bm_results.append({'% B Window': w,
                        'Bench Cumulative Return': cr_metric,
                        'Bench Annualized Return': ann_ret_metric,
                        'Bench Average Daily Return': sdr_metric,
                        'Bench Daily Standard Deviation': sdr_metric,
                        'Bench Annualized Standard Deviation': asdr_metric,
                        'Bench Annualized Sharpe Ratio': asharpe_metric})
        # 5) Normalize both
        norm = pd.concat([bench_vals, portvals], axis=1)
        norm = norm / norm.iloc[0]


        # pick the first 10 and last 10 buy/sell dates
        buy_dates = trades[trades[symbol] > 0].index
        sell_dates = trades[trades[symbol] < 0].index

        # extract the buy dates at the beginning and end and do the same for end dates
        buy_plot_dates = buy_dates[:10].append(buy_dates[-10:])
        sell_plot_dates = sell_dates[:10].append(sell_dates[-10:])

        # 6) Plot
        StrategyLearner.plot_strategy_vs_benchmark(bench_vals, portvals,
                                   buy_dates=buy_plot_dates,
                                   sell_dates=sell_plot_dates,
                                   title=f"Strategy Learner vs Benchmark (OOS) (Boll. Band Window - {w})",transaction_lines=False)

    sl_port_results_df = pd.DataFrame(port_results).set_index('% B Window')
    sl_bm_results_df = pd.DataFrame(bm_results).set_index('% B Window')
    combined_results = pd.concat([sl_port_results_df, sl_bm_results_df], axis=1)

    combined_results = (combined_results.round(6)).to_csv('./strategyLearner_Results.csv')

    return combined_results


if __name__ == "__main__":

    ### Evaluate Manual Learner ###
    ml = ManualLearner(verbose=False, impact=0.005, commission=9.95)

    # In Sample
    ml.mannual_run_sample_data(sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),
                               plot_title="Manual Strategy - In Sample", in_out_sample="In Sample", symbol="JPM",
                               start_val=100000)  # Task 3

    # Out of Sample
    ml.mannual_run_sample_data(sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31),
                               plot_title="Manual Strategy - Out Sample", in_out_sample="Out Sample", symbol="JPM",
                               start_val=100000)  # Task 4

    ### Evaluate Strategy Learner ###
    strategy_learner_results_df = evaluate_strategy_learner(verbose=False)

    # Experiment 1 Runner
    results1_df_in, results1_df_out = experiment1_runner()

    # Experiment 2 Runner
    results2_df = experiment2_runner()
