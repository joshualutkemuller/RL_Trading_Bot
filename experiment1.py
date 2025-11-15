"""
experiment1.py

Student Name: Joshua Lutkemuller (replace with your name)
GT User ID: jlutkemuller (replace with your User ID)
GT ID: 904051695 (replace with your GT ID)

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from util import get_data, plot_data
import datetime as dt
import pandas as pd
import StrategyLearner as StratLearner
from StrategyLearner import StrategyLearner
from indicators import (
    compute_bollinger_pb,
    compute_keltner_channels,
    compute_golden_cross,
    compute_rsi,
    compute_macd
)

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
def normalize_dataframe(df):
    """Normalize DataFrame to 1.0 at the first index"""
    return df / df.iloc[0]
import matplotlib.pyplot as plt
import numpy as np

def save_experiment1_plot(norm_df, title,
                          xlabel='Date',
                          ylabel='Normalized Portfolio Value',
                          file_name: str = None):
    """
    norm_df : DataFrame with 3 columns: ['ManualStrategy','StrategyLearner','Benchmark'],
              already normalized to 1.0 at the start.
    """

    # 1) Main line plot
    fig, ax = plt.subplots(figsize=(10,6))
    colors = ['red','green','purple']
    for col, color in zip(norm_df.columns, colors):
        ax.plot(norm_df.index, norm_df[col], label=col, color=color, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper left')
    ax.grid(True)

    # 2) Compute Sharpe ratios on normalized series
    sharpes = {}
    daily = norm_df.pct_change().dropna()
    for col in norm_df.columns:
        adr  = daily[col].mean()
        sddr = daily[col].std()
        sharpes[col] = (adr / sddr) * np.sqrt(252) if sddr > 0 else 0.0

    # 3) Inset bar chart of Sharpe ratios
    #    You can tweak bbox to position it nicely.
    inset_ax = fig.add_axes([0.65, 0.15, 0.25, 0.25])
    inset_ax.bar(sharpes.keys(), sharpes.values(), color=colors)
    inset_ax.set_title(f"{title} - Sharpe Ratio")
    inset_ax.tick_params(axis='x', rotation=45)
    inset_ax.grid(axis='y', linestyle='--', alpha=0.5)

    # 4) Save and close
    if file_name is None:
        file_name = title.replace(" ", "_").lower()
    plt.tight_layout()
    plt.savefig(f'./{file_name}.png')
    plt.close(fig)

def save_experiment1_plot_perf_only(norm_df,title,xlabel = 'Date', ylabel = 'Normalized Portfolio Values',
                          file_name : str = None):
    """
    Function designed to save Experiment 1 plot to project directory
    in the form `experiment1_SAMPLETYPE_plot.png`
    """
    plt.figure(figsize=(10,6))
    for col, color in zip(norm_df.columns, ['red','green','purple']):
        plt.plot(norm_df.index, norm_df[col], label=col, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./{file_name}.png')
    plt.close()



def experiment1_runner(symbol='JPM',
                   in_sample=(dt.datetime(2008,1,1), dt.datetime(2009,12,31)),
                   out_sample=(dt.datetime(2010,1,1), dt.datetime(2011,12,31)),
                   start_val=100000,
                   impact=0.005,
                   commission=9.95):
    # Unpack dates
    insd, ined = in_sample
    osd, oed   = out_sample

    # Initialize learners
    ml = ManualLearner(verbose=False, impact=impact, commission=commission)
    sl = StrategyLearner(verbose=False, impact=impact, commission=commission)

    # Train strategy learner in-sample
    sl.add_evidence(symbol=symbol, sd=insd, ed=ined, sv=start_val)

    # In-sample trades
    trades_manual_in = ml.testPolicy(symbol=symbol, sd=insd, ed=ined, sv=start_val)
    trades_sl_in     = sl.testPolicy(symbol=symbol, sd=insd, ed=ined, sv=start_val)

    # Benchmark trades: buy 1000 shares on first day, hold
    dates_in = pd.date_range(insd, ined)
    bench_trades_in = pd.DataFrame(0, index=dates_in, columns=[symbol])
    bench_trades_in.iloc[0,0] = 1000

    # Portfolio values for `in sample`
    pv_manual_in = compute_portvals(trades_manual_in, start_val=start_val,
                                    commission=commission, impact=impact)
    pv_sl_in      = compute_portvals(trades_sl_in, start_val=start_val,
                                     commission=commission, impact=impact)
    pv_bench_in   = compute_portvals(bench_trades_in, start_val=start_val,
                                     commission=commission, impact=impact)

    # Normalize dataframe to 1.0
    df_in = pd.concat([pv_manual_in, pv_sl_in, pv_bench_in], axis=1)

    # drop rows where any portvals are NaN, so all three series start together
    df_in = df_in.dropna()

    df_in.columns = ['ManualStrategy', 'StrategyLearner', 'Benchmark']
    df_norm_in = normalize_dataframe(df_in)

    # Plot `In-Sample` for expierment 1
    save_experiment1_plot(df_norm_in, 'Experiment 1: In-Sample Performance',
                          xlabel='Date', ylabel='Normalized Portfolio Values',
                          file_name = 'experiment1_in_sample')
    # Out-of-sample trades
    trades_manual_out = ml.testPolicy(symbol=symbol, sd=osd, ed=oed, sv=start_val)
    trades_sl_out     = sl.testPolicy(symbol=symbol, sd=osd, ed=oed, sv=start_val)

    # Benchmark for out of sample
    dates_out = pd.date_range(osd, oed)
    bench_trades_out = pd.DataFrame(0, index=dates_out, columns=[symbol])
    bench_trades_out.iloc[0,0] = 1000

    # Portfolio values for `out of sample`
    pv_manual_out = compute_portvals(trades_manual_out, start_val=start_val,
                                     commission=commission, impact=impact)
    pv_sl_out     = compute_portvals(trades_sl_out, start_val=start_val,
                                     commission=commission, impact=impact)
    pv_bench_out  = compute_portvals(bench_trades_out, start_val=start_val,
                                     commission=commission, impact=impact)

    # Normalize
    df_out = pd.concat([pv_manual_out, pv_sl_out, pv_bench_out], axis=1)
    df_out = df_out.dropna()
    df_out.columns = ['ManualStrategy', 'StrategyLearner', 'Benchmark']
    df_norm_out = normalize_dataframe(df_out)

    # # Plot out-of-sample
    save_experiment1_plot(df_norm_out, 'Experiment 1: Out-Of-Sample Performance',
                          xlabel='Date', ylabel='Normalized Portfolio Values',
                          file_name = 'experiment1_out_of_sample')
    return df_norm_in, df_norm_out

if __name__ == '__main__':
    results1_df_in, results1_df_out = experiment1_runner()