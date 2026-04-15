#=================== PART 1 -- Greeedy Optimization ==========================

import os
import pickle
import numpy as np
import pandas as pd
import itertools

import techindicators as ti
import fundamentalindicators as fi
from utilspositions import LongPosition, ShortPosition
from metric_utils import compute_equity_metrics


#=============================================================
#    Sharpe Ratio Metric
#=============================================================

def sharpe_ratio(equity, annualizing_factor=252):
    equity = equity.dropna()
    if len(equity) < 3:
        return -np.inf

    rets = equity.pct_change().dropna()
    std = rets.std(ddof=1)

    if std == 0 or np.isnan(std):
        return -np.inf

    return float(np.sqrt(annualizing_factor) * rets.mean() / std)


# ==================================================================
# Load collected data
# ==================================================================

def load_collected_data(pickle_path):
    if not os.path.exists(pickle_path):
        raise FileNotFoundError("Pickle file not found at " + str(pickle_path))
    with open(pickle_path, "rb") as f:
        return pickle.load(f)


def extract_price_series(collected_data, symbol, prefer_adj_close=True):
    for group_data in collected_data.values():
        if symbol in group_data:
            df = group_data[symbol].copy()

            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            df = df.sort_index()

            if prefer_adj_close and "Adj Close" in df.columns:
                return df["Adj Close"].astype(float)
            if "Close" in df.columns:
                return df["Close"].astype(float)
            if "close" in df.columns:
                return df["close"].astype(float)

            raise ValueError("No Close/Adj Close column found.")
    raise KeyError(f"Symbol not found for : {symbol}")


def extract_volume_series(collected_data, symbol):
    for group_data in collected_data.values():
        if symbol in group_data:
            df = group_data[symbol].copy()

            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            df = df.sort_index()

            if "Volume" in df.columns:
                return df["Volume"].astype(float)
            if "volume" in df.columns:
                return df["volume"].astype(float)

            return pd.Series(0.0, index=df.index, name="Volume")

    raise KeyError(f"Symbol not found for: {symbol}")


#=================================================================================
# NEW: TRAIN / VAL / TEST split by YEAR ranges
#=================================================================================

def split_by_years_3way(
    series: pd.Series,
    train_years=(2016, 2017, 2018, 2019, 2020, 2021),
    val_years=(2022, 2023),
    test_years=(2024, 2025, 2026),
):
    s = series.dropna().copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
    years = s.index.year

    train = s[years.isin(train_years)]
    val   = s[years.isin(val_years)]
    test  = s[years.isin(test_years)]

    return train, val, test


# =========================================================================================
# SIGNALS
# =========================================================================================

def signal_from_sma(price: pd.Series, L: int, theta: float = 0.0) -> pd.Series:
    ma = ti.sma(price, L)
    long_cond = price > ma * (1 + theta)
    short_cond = price < ma * (1 - theta)

    sig = pd.Series(0, index=price.index, dtype=int)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig


def signal_from_ema(price: pd.Series, L: int, theta: float = 0.0) -> pd.Series:
    ma = ti.ema(price, L)
    long_cond = price > ma * (1 + theta)
    short_cond = price < ma * (1 - theta)

    sig = pd.Series(0, index=price.index, dtype=int)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig


def signal_from_rsi(price: pd.Series, L: int, low, high) -> pd.Series:
    r = ti.rsi(price, L)
    sig = pd.Series(0, index=price.index, dtype=int)
    sig[r < low] = 1
    sig[r > high] = -1
    return sig


def signal_from_macd(price, fast, slow, signal_window) -> pd.Series:
    macd_df = ti.macd(price, short=fast, long=slow, signal=signal_window)
    trigger = ti.macd_signals(macd_df)

    sig = pd.Series(0, index=price.index, dtype=int)

    if "Bullish" in trigger.columns:
        sig[trigger["Bullish"].fillna(False)] = 1
    if "Bearish" in trigger.columns:
        sig[trigger["Bearish"].fillna(False)] = -1

    sig = sig.replace(0, np.nan).ffill().fillna(0).astype(int)
    return sig


def signal_from_bbands(price, L: int, k: float, theta: float = 0.0) -> pd.Series:
    bb = ti.boolinger_bands(price, n=L, k=k)
    mid = bb.iloc[:, 0]
    upper = bb.iloc[:, 1]
    lower = bb.iloc[:, 2]

    sig = pd.Series(0, index=price.index, dtype=int)
    sig[price < lower * (1.0 - theta)] = 1
    sig[price > upper * (1.0 + theta)] = -1

    hold = sig.replace(0, np.nan).ffill()
    exit_long = (hold == 1) & (price >= mid)
    exit_short = (hold == -1) & (price <= mid)
    hold[exit_long | exit_short] = 0
    hold = hold.fillna(0).astype(int)
    return hold


def signal_from_obv(price, volume, L, theta=0.0):
    obv_series = ti.obv(price, volume)
    obv_ma = ti.sma(obv_series, L)

    base = obv_ma.abs().replace(0, np.nan)

    sig = pd.Series(0, index=price.index, dtype=int)
    sig[obv_series > obv_ma + theta * base] = 1
    sig[obv_series < obv_ma - theta * base] = -1
    return sig


def signal_from_pe_ratio(pe, theta):
    sig = pd.Series(0, index=pe.index, dtype=int)
    sig[pe < theta] = 1
    sig[pe > theta] = -1
    return sig


def signal_from_earnings_surprise(es, theta):
    sig = pd.Series(0, index=es.index, dtype=int)
    sig[es > theta] = 1
    sig[es < -theta] = -1
    return sig


# ==================== BACKTESTING ==============================

def backtest(price, raw_signal, initial_capital, fee, allow_short=True, lambda_short=1.5):
    price = price.dropna()
    raw_signal = raw_signal.reindex(price.index).fillna(0).astype(int)

    signal = raw_signal.shift(1).fillna(0).astype(int)

    cash = float(initial_capital)
    shares = 0
    position_type = 0

    equity_curve = []

    for t, p in price.items():
        desired = int(signal.loc[t])

        if desired != position_type:
            # Close existing
            if position_type == 1 and shares > 0:
                cash += shares * p
                cash -= fee
                shares = 0
                position_type = 0

            elif position_type == -1 and shares < 0:
                cash += shares * p
                cash -= fee
                shares = 0
                position_type = 0

            # Open desired
            if desired == 1:
                longp = LongPosition(cash, p, fee)
                qmax = longp.q_maxShares()
                if qmax > 0:
                    cash -= (qmax * p + fee)
                    shares = qmax
                    position_type = 1

            elif desired == -1 and allow_short:
                shortp = ShortPosition(cash, p, fee, lambda_short)
                qmax = shortp.q_short_maxShares()
                if qmax > 0:
                    cash += (qmax * p - fee)
                    shares = -qmax
                    position_type = -1

        equity_curve.append(cash + shares * p)

    return pd.Series(equity_curve, index=price.index, name="Equity")


# =============================================================================
# GRID PARAMS
# =============================================================================

def default_grid_params(
    sma_L=(10, 20, 50),
    sma_theta=(0.0, 0.002, 0.005),
    ema_L=(10, 20, 50),
    ema_theta=(0.0, 0.002, 0.005),
    rsi_L=(7, 14, 21),
    rsi_levels=((30, 70), (35, 65), (25, 75)),
    macd_triplets=((12, 26, 9), (8, 21, 9), (5, 35, 5)),
    bb_L=(10, 20, 50),
    bb_k=(1.5, 2.0, 2.5),
    bb_theta=(0.0, 0.002),
    obv_L=(5, 10, 20),
    obv_theta=(0.0, 0.01),
    pe_theta=(10, 15, 20, 25),
    es_theta=(0.02, 0.05, 0.10),
):
    grids = {}
    grids["SMA"] = [{"L": L, "theta": th} for L in sma_L for th in sma_theta]
    grids["EMA"] = [{"L": L, "theta": th} for L in ema_L for th in ema_theta]
    grids["RSI"] = [{"L": L, "low": low, "high": high} for L in rsi_L for (low, high) in rsi_levels]
    grids["MACD"] = [{"fast": f, "slow": s, "signal": sig} for (f, s, sig) in macd_triplets]
    grids["BBANDS"] = [{"L": L, "k": k, "theta": th} for L in bb_L for k in bb_k for th in bb_theta]
    grids["OBV"] = [{"L": L, "theta": th} for L in obv_L for th in obv_theta]
    grids["PE"] = [{"theta": th} for th in pe_theta]
    grids["EARN_SURPRISE"] = [{"theta": th} for th in es_theta]
    return grids


def build_signal(indicator, price, volume, params, extra_series=None):
    if indicator == "SMA":
        return signal_from_sma(price, params["L"], params.get("theta", 0.0))
    if indicator == "EMA":
        return signal_from_ema(price, params["L"], params.get("theta", 0.0))
    if indicator == "RSI":
        return signal_from_rsi(price, params["L"], params["low"], params["high"])
    if indicator == "MACD":
        return signal_from_macd(price, params["fast"], params["slow"], params["signal"])
    if indicator == "BBANDS":
        return signal_from_bbands(price, params["L"], params["k"], params.get("theta", 0.0))
    if indicator == "OBV":
        return signal_from_obv(price, volume, params["L"], params.get("theta", 0.0))

    if indicator == "PE":
        if extra_series is None or "pe" not in extra_series:
            return None
        pe = extra_series["pe"].reindex(price.index)
        return signal_from_pe_ratio(pe, params["theta"])

    if indicator == "EARN_SURPRISE":
        if extra_series is None or "es" not in extra_series:
            return None
        es = extra_series["es"].reindex(price.index)
        return signal_from_earnings_surprise(es, params["theta"])

    raise ValueError("Unknown indicator: " + str(indicator))


def combine_signals(signals):
    df = pd.concat(signals, axis=1).fillna(0).astype(int)
    D = df.sum(axis=1)
    out = pd.Series(0, index=df.index, dtype=int)
    out[D > 0] = 1
    out[D < 0] = -1
    return out


# =============================================================================
# GREEDY OPTIMIZATION (train on TRAIN only)
# =============================================================================

def greedy_optimize_single_indicator(
    price_train, vol_train, name, grid, initial_capital, fee,
    allow_short=True, lambda_short=1.5, annualization=252
):
    best_params = None
    best_score = -np.inf

    for params in grid:
        sig = build_signal(name, price_train, vol_train, params)
        if sig is None:
            continue

        equity = backtest(price_train, sig, initial_capital, fee, allow_short, lambda_short)
        score = sharpe_ratio(equity, annualization)

        if score > best_score:
            best_score = score
            best_params = params

    return best_params, float(best_score)


def greedy_forward_select(
    price_train, vol_train, grids, initial_capital, fee,
    allow_short=True, lambda_short=1.5, annualization=252,
    max_indicators=3
):
    chosen = []
    chosen_params = {}
    best_score = -np.inf
    best_combined_signal = None

    remaining = list(grids.keys())

    for _ in range(max_indicators):
        best_add = None
        best_add_params = None
        best_add_score = best_score
        best_add_signal = None

        for indi in remaining:
            params, _ = greedy_optimize_single_indicator(
                price_train, vol_train, indi, grids[indi], initial_capital, fee,
                allow_short=allow_short, lambda_short=lambda_short, annualization=annualization
            )
            if params is None:
                continue

            sig_new = build_signal(indi, price_train, vol_train, params)
            if sig_new is None:
                continue

            combined = sig_new if best_combined_signal is None else combine_signals([best_combined_signal, sig_new])

            equity = backtest(price_train, combined, initial_capital, fee, allow_short, lambda_short)
            score = sharpe_ratio(equity, annualization)

            if score > best_add_score:
                best_add_score = score
                best_add = indi
                best_add_params = params
                best_add_signal = combined

        if best_add is None:
            break

        chosen.append(best_add)
        chosen_params[best_add] = best_add_params
        best_score = best_add_score
        best_combined_signal = best_add_signal
        remaining.remove(best_add)

    return chosen, chosen_params, float(best_score)


# =============================================================================
#  Wrapping returns under TRAIN + VAL + TEST evaluation
# =============================================================================

def run_greedy_for_symbol(
    symbol,
    pickle_path="./data/collected_data.pkl",
    initial_capital=10000.0,
    fee=1.0,
    allow_short=True,
    lambda_short=1.5,
    annualization=252,
    max_indicators=3,
    grids=None,
    allowed_indicators=None,
    forced_indicators=None,
    # NEW:
    train_years=(2016, 2017, 2018, 2019, 2020, 2021),
    val_years=(2022, 2023),
    test_years=(2024, 2025, 2026),
):
    collected = load_collected_data(pickle_path)

    price_all = extract_price_series(collected_data=collected, symbol=symbol)
    vol_all = extract_volume_series(collected_data=collected, symbol=symbol)

    # 3-way time split
    price_train, price_val, price_test = split_by_years_3way(
        price_all, train_years=train_years, val_years=val_years, test_years=test_years
    )

    if len(price_train) < 50:
        raise ValueError("Train set too small after year split. Check available years in your data.")
    if len(price_val) < 10:
        print("[WARN] Validation set is very small after year split.")
    if len(price_test) < 10:
        print("[WARN] Test set is very small after year split.")

    vol_train = vol_all.reindex(price_train.index).fillna(0.0)
    vol_val   = vol_all.reindex(price_val.index).fillna(0.0)
    vol_test  = vol_all.reindex(price_test.index).fillna(0.0)

    # Grid setup + optional filtering
    if grids is None:
        grids = default_grid_params()

    if allowed_indicators is not None:
        allowed_set = set(allowed_indicators)
        grids = {k: v for k, v in grids.items() if k in allowed_set}

    if len(grids) == 0:
        raise ValueError("No indicators left in grids after applying allowed_indicators filtering.")

    def build_final_signal(price_series, vol_series, chosen_inds, chosen_params_dict):
        sigs = []
        for ind in chosen_inds:
            params = chosen_params_dict[ind]
            s = build_signal(ind, price_series, vol_series, params)
            if s is not None:
                sigs.append(s)

        if len(sigs) == 0:
            return pd.Series(0, index=price_series.index, dtype=int)
        if len(sigs) == 1:
            return sigs[0]
        return combine_signals(sigs)

    # Mode 1: Forced indicator set (tune combo on TRAIN only)
    if forced_indicators is not None:
        forced = list(forced_indicators)

        missing = [x for x in forced if x not in grids]
        if missing:
            raise ValueError(
                f"forced_indicators contains indicators not available in grids: {missing}. "
                f"Available keys: {list(grids.keys())}"
            )

        grid_lists = [grids[name] for name in forced]

        best_score = -np.inf
        best_params_combo = None

        for combo in itertools.product(*grid_lists):
            params_dict = {forced[i]: combo[i] for i in range(len(forced))}

            sig_train_combo = build_final_signal(price_train, vol_train, forced, params_dict)
            equity_train_combo = backtest(price_train, sig_train_combo, initial_capital, fee, allow_short, lambda_short)
            score = sharpe_ratio(equity_train_combo, annualization)

            if score > best_score:
                best_score = score
                best_params_combo = params_dict

        chosen = forced
        chosen_params = best_params_combo if best_params_combo is not None else {}
        train_sharpe_greedy = float(best_score)

    # Mode 2: Normal greedy forward selection on TRAIN only
    else:
        chosen, chosen_params, train_sharpe_greedy = greedy_forward_select(
            price_train, vol_train, grids, initial_capital, fee,
            allow_short, lambda_short, annualization, max_indicators=max_indicators
        )

    # Evaluate on TRAIN / VAL / TEST (same chosen strategy)
    sig_train = build_final_signal(price_train, vol_train, chosen, chosen_params)
    equity_train = backtest(price_train, sig_train, initial_capital, fee, allow_short, lambda_short)

    sig_val = build_final_signal(price_val, vol_val, chosen, chosen_params) if len(price_val) else pd.Series(dtype=int)
    equity_val = backtest(price_val, sig_val, initial_capital, fee, allow_short, lambda_short) if len(price_val) else pd.Series(dtype=float)

    sig_test = build_final_signal(price_test, vol_test, chosen, chosen_params) if len(price_test) else pd.Series(dtype=int)
    equity_test = backtest(price_test, sig_test, initial_capital, fee, allow_short, lambda_short) if len(price_test) else pd.Series(dtype=float)

    train_sharpe = sharpe_ratio(equity_train, annualization)
    val_sharpe   = sharpe_ratio(equity_val, annualization) if len(equity_val) else float("nan")
    test_sharpe  = sharpe_ratio(equity_test, annualization) if len(equity_test) else float("nan")

    train_metrics = compute_equity_metrics(equity_train, annualization=annualization, rf=0.0)
    val_metrics   = compute_equity_metrics(equity_val, annualization=annualization, rf=0.0) if len(equity_val) else {}
    test_metrics  = compute_equity_metrics(equity_test, annualization=annualization, rf=0.0) if len(equity_test) else {}

    return {
        "symbol": symbol,
        "chosen_indicators": chosen,
        "chosen_params": chosen_params,

        "train_sharpe_greedy": train_sharpe_greedy,
        "train_sharpe": train_sharpe,
        "val_sharpe": val_sharpe,
        "test_sharpe": test_sharpe,

        "equity_train": equity_train,
        "equity_val": equity_val,
        "equity_test": equity_test,

        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,

        "price_train": price_train,
        "price_val": price_val,
        "price_test": price_test,

        "sig_train": sig_train,
        "sig_val": sig_val,
        "sig_test": sig_test,
    }


if __name__ == "__main__":
    pass
