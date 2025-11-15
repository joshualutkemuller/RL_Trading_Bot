"""
experiment2.py

Student Name: Joshua Lutkemuller (replace with your name)
GT User ID: jlutkemuller (replace with your User ID)
GT ID: 904051695 (replace with your GT ID)

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from util import get_data, plot_data
import datetime
import pandas as pd
import StrategyLearner as StratLearner
from StrategyLearner import StrategyLearner


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
def computer_experiment2_metrics(portfolio_values):
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

def experiment2_plot(df: pd.DataFrame = None, titlename : str = 'Experiment 2: Impact Sensitivity (In-Sample)',
                     file_name :str = 'experiment2_impact_sensitivity_insample'):

    # Set figures and subplots for the impact sensitivity analysis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Plots and settings for Subplot #1
    ax1.plot(df.index, df['Cumulative Return'], 'o-', label='Cumulative Return')
    ax1.plot(df.index, df['Average Daily Return'], '^-', label='Average Daily Return')
    ax1.legend();
    ax1.grid(True)
    ax1.set_ylabel('Return')

    # Plots and settings for Subplot #2
    # left y-axis for Sharpe
    ax2.set_xlabel('Impact')
    ax2.set_ylabel('Sharpe Ratio', color='green')
    ax2_s = ax2.plot(df.index, df['Annualized Sharpe Ratio'], marker='s', linestyle='--', color='green', label='Annualized Sharpe Ratio')
    ax2.tick_params(axis='y', labelcolor='green')

    # create secondary y-axis for Annualized Std Dev
    ax2b = ax2.twinx()
    ax2b.plot(df.index, df['Annualized Standard Deviation'], marker='d', linestyle='-.', color='purple', label='Annualized StdDev')
    ax2b.set_ylabel('Annualized StdDev', color='purple')
    ax2b.tick_params(axis='y', labelcolor='purple')

    # combine legends
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')
    ax2.grid(True)


    # Set Title of Entire Plot and Layout Settings & Save
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.suptitle('Experiment 2: Impact Sensitivity (In-Sample)')
    plt.title(titlename)
    plt.savefig(f'./{file_name}.png')
    plt.close(fig)

    # Plot results table
    # filter to impacts of interest and round values for table
    selected_impacts = [0.0, 0.0025, 0.005, 0.01]

    # ensure only those present
    df_table = df.copy()
    df_table = df_table.loc[df_table.index.intersection(selected_impacts)]
    df_table = df_table.round(2)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis('off')
    tbl = ax.table(cellText=df_table.reset_index().values,
                   colLabels=df_table.reset_index().columns,
                   loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2)
    plt.title('Experiment 2 Results Table (Metrics vs. Levels of Impact')
    plt.savefig('experiment2_results_table.png')
    plt.close(fig)

def plot_cumulative_returns_by_impact(portvals_dict,
                                      title='Cumulative Returns by Impact',
                                      file_name='impact_returns_over_time.png'):
    """
    portvals_dict : dict
        keys = impact level (float)
        values = pd.Series or pd.DataFrame of portfolio values indexed by date
    """
    plt.figure(figsize=(12, 6))
    for impact, pv in portvals_dict.items():
        # normalize to 1.0 at start
        norm = pv.iloc[:, 0] / pv.iloc[0, 0] if hasattr(pv, 'iloc') else pv / pv.iloc[0]
        plt.plot(norm.index, norm, label=f'impact={impact:.4f}')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Normalized Portfolio Value')
    plt.legend(loc='best', fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'experiment2_{file_name}')
    plt.close()


def experiment2_runner(symbol='JPM',
                       in_sample_date_tuple = (datetime.datetime(2008,1,1),
                                               datetime.datetime(2009,12,31)),
                       impact_list = [0.0,0.005,.02],
                       commission = 0.0,
                       start_val=100000):

    in_sample_sd , in_sample_ed = in_sample_date_tuple

    # initialize an empty array to store results
    experiment_results = []

    # add just before the for‐loop:
    portvals_by_impact = {}

    # iterate over impacts
    for impact in [.0005, .0025,.005]: #np.arange(0, 0.05 + 1e-8, 0.005):

        # Initiate the StrategyLearner class
        sl = StrategyLearner(verbose=False, impact=impact,commission=commission)

        # Add evidence
        sl.add_evidence(symbol=symbol, sd = in_sample_sd, ed=in_sample_ed, sv=start_val)

        # Run the in-sample test to get the trades
        trades_data = sl.testPolicy(symbol=symbol, sd = in_sample_sd, ed=in_sample_ed, sv=start_val)

        # Compute portfolio values
        portfolio_values = compute_portvals(trades_data, start_val = start_val,
                                            commission = commission,
                                            impact=impact)

        # Change column of dataframe
        portfolio_values.columns = ["Portfolio Values"]

        # Compute metrics
        cr_metric, ann_ret_metric, adr_metric, \
            sdr_metric, asdr_metric, asharpe_metric = computer_experiment2_metrics(portfolio_values)

        # Append data to list
        experiment_results.append({"Impact": impact,
                                   "Cumulative Return": cr_metric,
                                   "Annualized Return": ann_ret_metric,
                                   "Average Daily Return": adr_metric,
                                   "Daily Standard Deviation":sdr_metric,
                                   "Annualized Standard Deviation": asdr_metric,
                                   "Annualized Sharpe Ratio": asharpe_metric})

        # store a copy
        portvals_by_impact[impact] = portfolio_values.copy()

    # Create the Cumulative Return by Impact DataFrame
    #cumulative_port_values_by_impact = pd.DataFrame(portvals_by_impact)

    # After loop iterations are complete, append into dataframe
    results_df = pd.DataFrame(experiment_results).set_index('Impact')

    # Sorting index
    results_df.sort_index(ascending=True, inplace=True)

    # Archive to current project directory
    results_df.to_csv('./experiment2_results.csv')

    # Run plot function
    experiment2_plot(results_df)

    # Run cumulative plotting over time by impact
    plot_cumulative_returns_by_impact(portvals_by_impact)

    return results_df

if __name__ == "__main__":
    results2_df = experiment2_runner()
