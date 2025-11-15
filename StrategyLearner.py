# StrategyLearner.py

import datetime as dt
import pandas as pd
import numpy as np
from util import get_data
from indicators import (
    compute_bollinger_pb,
    compute_golden_cross,
    compute_rsi,
    compute_macd
)
from RTLearner import RTLearner
from BagLearner import BagLearner
import matplotlib.pyplot as plt

def compute_portvals(
    trades_df,
    symbol='JPM',
    start_val=1000000,
    commission=0.0,
    impact=0.0
):
    """
    Compute portfolio values from a trades DataFrame.
    """
    sym = trades_df.columns[0]
    dates = trades_df.index
    prices = get_data([sym], dates)[[sym]].copy()
    prices['CASH'] = 1.0
    prices.fillna(method='ffill', inplace=True)
    prices.fillna(method='bfill', inplace=True)

    trades = pd.DataFrame(0.0, index=dates, columns=[sym, 'CASH'])
    trades[sym] = trades_df[sym]
    trades['CASH'] = (
        - trades_df[sym] * prices[sym] * (1 + impact)
        - commission * (trades_df[sym] != 0)
    )
    trades.loc[dates[0], 'CASH'] += start_val

    holdings = trades.cumsum()
    portvals = (holdings * prices).sum(axis=1)
    return portvals.to_frame(name='Portvals')


class StrategyLearner(object):
    """
    A bagged RTLearner classification trader using technical indicators.
    """

    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        self.verbose    = verbose
        self.impact     = impact
        self.commission = commission
        self.bb_window  = 20
        self.bags       = 20    # per hint, ≥20 bags
        self.learner    = None

    def author(self):
        return "jlutkemuller3"

    def gtid(self):
        return 904051695

    def study_group(self):
        return "jlutkemuller3"

    def add_evidence(self, symbol="JPM",
                     sd=dt.datetime(2008,1,1),
                     ed=dt.datetime(2009,12,31),
                     sv=100000):
        # 1) Load prices
        dates  = pd.date_range(sd, ed)
        prices = get_data([symbol], dates)[symbol].ffill().bfill()

        # 2) Compute indicators
        bbp   = compute_bollinger_pb(prices, window=self.bb_window)
        gc    = compute_golden_cross(prices, 50, 200)
        rsi14 = compute_rsi(prices, window=14)
        macd  = compute_macd(prices, 12, 26, 9)
        X_all = pd.DataFrame({
            'BBP':  bbp,
            'GC':   gc,
            'RSI':  rsi14,
            'MACD': macd
        }).fillna(0.0)

        # 3) Grid‐search parameters by validation
        horizons   = [1, 3, 5]
        thresholds = [0.0, 0.001, 0.0025, 0.005]  # include zero‐threshold
        leaf_sizes = [5, 10, 15]                 # iterate through leaf sizes
        split_idx  = lambda m: int(0.7 * m)

        # Precompute raw returns for each horizon
        ret_full = {}
        for h in horizons:
            ret_full[h] = (prices.shift(-h) / prices - 1).iloc[:-h].values

        best_score  = -np.inf
        best_params = (1, 0.0, 5)

        n = len(prices)
        date_index = prices.index

        for h in horizons:

            # returns array
            rets = ret_full[h]

            # feature rows 0 .. n-h-1 correspond to dates[0..n-h-1]
            Xh = X_all.iloc[:-h]
            m  = len(rets)
            cut = split_idx(m)

            X_train = Xh.iloc[:cut].values
            X_val   = Xh.iloc[cut:].values

            dates_val = date_index[cut: cut + len(X_val)]

            for thresh in thresholds:
                # build labels
                y_full = np.where(rets >  thresh,  1,
                          np.where(rets < -thresh, -1, 0))
                y_train, y_val = y_full[:cut], y_full[cut:]

                for leaf in leaf_sizes:
                    # train
                    bag = BagLearner(RTLearner, {"leaf_size": leaf},
                                     bags=self.bags, boost=False)
                    bag.add_evidence(X_train, y_train)
                    raw_preds = bag.query(X_val)
                    preds = np.where(raw_preds >  0, 1,
                             np.where(raw_preds <  0, -1, 0))

                    # simulate validation trades
                    trades_val = pd.DataFrame(0, index=dates_val, columns=[symbol])
                    pos = 0
                    for j in range(len(preds)-1):
                        tgt   = preds[j] * 1000
                        delta = tgt - pos
                        trades_val.iat[j+1, 0] = delta
                        pos = tgt
                    # flatten at the end
                    if pos != 0:
                        trades_val.iat[-1, 0] -= pos

                    # compute validation cumulative return
                    pv = compute_portvals(
                        trades_val,
                        symbol=symbol,
                        start_val=sv,
                        commission=0.0,
                        impact=self.impact
                    )['Portvals']
                    cr = pv.iloc[-1] / pv.iloc[0] - 1.0

                    # update best
                    if cr > best_score:
                        best_score  = cr
                        best_params = (h, thresh, leaf)

        # save best hyperparameters
        self.best_horizon, self.best_thresh, self.best_leaf_size = best_params
        if self.verbose:
            horizon, return_threshold, leaf = best_params
            print(f"[best] h={horizon}, thresh={return_threshold:.4f}, leaf={leaf}, valCR={best_score:.3f}")

        # 4) Retrain on full in‐sample
        h, thresh, leaf = best_params
        rets = ret_full[h]
        Xf   = X_all.iloc[:-h].values
        y_final = np.where(rets >  thresh,  1,
                   np.where(rets < -thresh, -1, 0))

        self.learner = BagLearner(RTLearner, {"leaf_size": leaf},
                                  bags=self.bags, boost=False,
                                  verbose=self.verbose)
        self.learner.add_evidence(Xf, y_final)


    def testPolicy(self, symbol="JPM",
                   sd=dt.datetime(2010,1,1),
                   ed=dt.datetime(2011,12,31),
                   sv=100000):
        # 1) Load test data
        dates  = pd.date_range(sd, ed)
        prices = get_data([symbol], dates)[symbol].ffill().bfill()

        # 2) Indicators
        bbp   = compute_bollinger_pb(prices, window=self.bb_window)
        gc    = compute_golden_cross(prices, 50, 200)
        rsi14 = compute_rsi(prices, window=14)
        macd  = compute_macd(prices, 12, 26, 9)
        Xtest = pd.DataFrame({
            'BBP': bbp,
            'GC':  gc,
            'RSI': rsi14,
            'MACD':macd
        }).fillna(0.0).values

        # 3) Predict
        raw = self.learner.query(Xtest)
        preds = np.where(raw >  0, 1,
                np.where(raw <  0, -1, 0)).astype(int)

        # 4) Build trades DF
        trades = pd.DataFrame(0, index=prices.index, columns=[symbol])
        position_start   = 0
        max_position  = 1000

        # trade next day after signal
        for i in range(len(preds)-1):
            target   = preds[i] * max_position
            delta = target - position_start
            trades.iat[i+1, 0] = delta
            position_start = target

        # flatten at end
        if position_start != 0:
            trades.iat[-1, 0] -= position_start

        return trades.astype(int)


    @staticmethod
    def plot_strategy_vs_benchmark(benchmark_vals, strategy_vals,
                                   buy_dates=None, sell_dates=None,
                                   title="Strategy vs Benchmark",
                                   transaction_lines=False):
        bench_n = benchmark_vals / benchmark_vals.iloc[0,0]
        strat_n = strategy_vals  / strategy_vals.iloc[0,0]
        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(bench_n.index, bench_n.iloc[:,0], label='Benchmark')
        ax.plot(strat_n.index, strat_n.iloc[:,0], label='Strategy')
        if transaction_lines:
            if buy_dates:
                for d in buy_dates:  ax.axvline(d, linestyle='--', alpha=0.5)
            if sell_dates:
                for d in sell_dates: ax.axvline(d, linestyle='--', alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Normalized Price")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{title}.png")
        plt.close(fig)
