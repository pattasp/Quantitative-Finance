# Plot utilities module
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure the right working directory path

def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path)


def plot_equity_curve(equity, title, save_path):
    """
    Plot equity curve and save to PNG.

    equity: pd.Series indexed by date
    """
    equity = pd.Series(equity).dropna()
    if len(equity) == 0:
        print("plot_equity_curve: empty equity, skipping:", save_path)
        return

    ensure_dir(os.path.dirname(save_path))

    plt.figure()
    plt.plot(equity.index, equity.values)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _extract_trade_points_from_signal(price, raw_signal):
    """
    Fallback: infer entry/exit points from signal changes (approximate).
    Assumes you trade using shifted signal (like your backtest does).
    """
    price = pd.Series(price).dropna()
    raw_signal = pd.Series(raw_signal).reindex(price.index).fillna(0).astype(int)

    # Match your backtest: execute today using yesterday's signal
    sig = raw_signal.shift(1).fillna(0).astype(int)

    prev = sig.shift(1).fillna(0).astype(int)

    # Entry points:
    long_entry = (sig == 1) & (prev != 1)
    short_entry = (sig == -1) & (prev != -1)

    # Exit points (position -> flat or flip):
    long_exit = (prev == 1) & (sig != 1)
    short_exit = (prev == -1) & (sig != -1)

    return long_entry, long_exit, short_entry, short_exit


def plot_price_with_entries_exits(price, raw_signal, title, save_path):
    """
    Plot price series with entry/exit markers inferred from signal changes.
    This is an approximation (not trade-log perfect), but good enough for the deliverable
    until you add real trades_df logging.

    price: pd.Series
    raw_signal: pd.Series of -1,0,1
    """
    price = pd.Series(price).dropna()
    if len(price) == 0:
        print("plot_price_with_entries_exits: empty price, skipping:", save_path)
        return

    raw_signal = pd.Series(raw_signal).reindex(price.index).fillna(0).astype(int)

    ensure_dir(os.path.dirname(save_path))

    long_entry, long_exit, short_entry, short_exit = _extract_trade_points_from_signal(price, raw_signal)

    plt.figure()
    plt.plot(price.index, price.values, label="Price")

    # Markers
    plt.scatter(price.index[long_entry], price[long_entry], marker="^", label="Long Entry")
    plt.scatter(price.index[long_exit], price[long_exit], marker="v", label="Long Exit")

    plt.scatter(price.index[short_entry], price[short_entry], marker="v", label="Short Entry")
    plt.scatter(price.index[short_exit], price[short_exit], marker="^", label="Short Exit")

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_price_with_trades(price, trades_df, title, save_path):
    """
    Plot price with entry/exit markers from real executed trades.
    Use this once you add trade logging to your backtest.

    trades_df must have columns: ["date", "action", "price"] at minimum.
    action examples: OPEN_LONG, CLOSE_LONG, OPEN_SHORT, CLOSE_SHORT, etc.
    """
    price = pd.Series(price).dropna()
    if len(price) == 0:
        print("plot_price_with_trades: empty price, skipping:", save_path)
        return

    if trades_df is None or len(trades_df) == 0:
        print("plot_price_with_trades: empty trades_df, skipping:", save_path)
        return

    df = trades_df.copy()
    if "date" not in df.columns or "action" not in df.columns:
        print("plot_price_with_trades: trades_df missing required columns, skipping:", save_path)
        return

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    ensure_dir(os.path.dirname(save_path))

    plt.figure()
    plt.plot(price.index, price.values, label="Price")

    # Separate actions
    def _scatter(action_name, marker, label):
        m = df["action"] == action_name
        if m.any():
            plt.scatter(df.loc[m, "date"], df.loc[m, "price"], marker=marker, label=label)

    _scatter("OPEN_LONG", "^", "Open Long")
    _scatter("CLOSE_LONG", "v", "Close Long")
    _scatter("OPEN_SHORT", "v", "Open Short")
    _scatter("CLOSE_SHORT", "^", "Close Short")
    _scatter("FORCE_CLOSE_LONG", "x", "Force Close Long")
    _scatter("FORCE_CLOSE_SHORT", "x", "Force Close Short")

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
