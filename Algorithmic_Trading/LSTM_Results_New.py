# LSTM_Results_New.py
# Readjusted to match the newest LSTM_Base.py API:
# - train_one_epoch_hybrid / evaluate_epoch return dict metrics
# - get_series_on_loader / collect_diagnostics require `device`
# - discrete capital simulation aligned to test mask (no n_train heuristic)

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

# ---- Core logic from Base ----
from LSTM_Base import (
    device,
    TradingDataset,
    LSTMTradingAgent,
    build_tensors_from_combined_df,
    train_one_epoch_hybrid,
    evaluate_epoch,
    get_series_on_loader,
    collect_diagnostics,
    simulate_portfolio_discrete,
)

# ---- Your pipeline modules ----
from DataCollector import load_collected_data
from DataPreparator import build_combined_price_df


# =============================================================================
# Plot saving
# =============================================================================
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def _save_or_show(fig, filename: str, show: bool = False):
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"[saved] {path}")


# =============================================================================
# Plot helpers
# =============================================================================
def plot_history(results_df: pd.DataFrame, show: bool = False):
    fig = plt.figure()
    plt.plot(results_df["epoch"], results_df["train_sharpe"], label="Train Sharpe")
    plt.plot(results_df["epoch"], results_df["val_sharpe"], label="Val Sharpe")
    plt.plot(results_df["epoch"], results_df["test_sharpe"], label="Test Sharpe")
    plt.xlabel("Epoch")
    plt.ylabel("Sharpe")
    plt.title("Sharpe Over Epochs")
    plt.legend()
    _save_or_show(fig, "sharpe_over_epochs.png", show=show)

    fig = plt.figure()
    plt.plot(results_df["epoch"], results_df["train_mean"], label="Train Mean")
    plt.plot(results_df["epoch"], results_df["val_mean"], label="Val Mean")
    plt.plot(results_df["epoch"], results_df["test_mean"], label="Test Mean")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Strategy Return")
    plt.title("Mean Strategy Return Over Epochs")
    plt.legend()
    _save_or_show(fig, "mean_return_over_epochs.png", show=show)


def plot_equity_from_returns(strat_rets: np.ndarray, asset_rets: np.ndarray, name: str, show: bool = False):
    strat_rets = np.asarray(strat_rets).reshape(-1)
    asset_rets = np.asarray(asset_rets).reshape(-1)

    equity = np.cumprod(1.0 + strat_rets)
    buyhold = np.cumprod(1.0 + asset_rets)

    fig = plt.figure()
    plt.plot(equity, label=f"Strategy ({name})")
    plt.plot(buyhold, label="Buy & Hold (target)")
    plt.xlabel("Test Steps")
    plt.ylabel("Growth of $1")
    plt.title(f"Equity Curve from Returns (Test) [{name}]")
    plt.legend()
    _save_or_show(fig, f"equity_curve_returns_{name}.png", show=show)


def plot_capital_curve(C: np.ndarray, show: bool = False, filename: str = "capital_curve_discrete.png"):
    C = np.asarray(C).reshape(-1)
    fig = plt.figure()
    plt.plot(C / C[0])
    plt.xlabel("Test Steps")
    plt.ylabel("Growth of $1")
    plt.title("Capital Curve C(t) (Discrete Execution)")
    _save_or_show(fig, filename, show=show)


def plot_distribution(strat_rets: np.ndarray, name: str, show: bool = False):
    strat_rets = np.asarray(strat_rets).reshape(-1)
    fig = plt.figure()
    plt.hist(strat_rets, bins=50)
    plt.title(f"Strategy Returns Distribution (Test) [{name}]")
    plt.xlabel("Return")
    plt.ylabel("Frequency")
    _save_or_show(fig, f"returns_hist_{name}.png", show=show)


def plot_series(series: np.ndarray, title: str, filename: str, ylabel: str, show: bool = False):
    series = np.asarray(series).reshape(-1)
    fig = plt.figure()
    plt.ticklabel_format(style="plain", axis="y", useOffset=False)
    plt.plot(series)
    plt.title(title)
    plt.xlabel("Test Steps")
    plt.ylabel(ylabel)
    _save_or_show(fig, filename, show=show)


def plot_lambdas(lambda_series: list[np.ndarray], max_indicators_to_plot: int = 4, show: bool = False):
    k = min(len(lambda_series), max_indicators_to_plot)
    for i in range(k):
        lam = np.asarray(lambda_series[i]).reshape(-1)
        fig = plt.figure()
        plt.plot(lam)
        plt.title(f"Learned Window Size λ_{i}(t) on Test")
        plt.xlabel("Test Steps")
        plt.ylabel("Window Size")
        _save_or_show(fig, f"lambda_{i}.png", show=show)


def plot_thresholds(th_plus: list[np.ndarray], th_minus: list[np.ndarray], max_indicators_to_plot: int = 4, show: bool = False):
    k = min(len(th_plus), len(th_minus), max_indicators_to_plot)
    for i in range(k):
        fig = plt.figure()
        plt.plot(np.asarray(th_plus[i]).reshape(-1), label="theta_plus")
        plt.plot(np.asarray(th_minus[i]).reshape(-1), label="theta_minus")
        plt.title(f"Thresholds θ_{i}(t) on Test")
        plt.xlabel("Test Steps")
        plt.ylabel("Threshold (scaled)")
        plt.legend()
        _save_or_show(fig, f"thresholds_{i}.png", show=show)


# =============================================================================
# Simple metric summary table
# =============================================================================
def summarize_returns(r: np.ndarray):
    r = np.asarray(r).reshape(-1)
    mean = float(np.mean(r))
    std = float(np.std(r) + 1e-12)
    sharpe = mean / std
    total = float(np.prod(1.0 + r) - 1.0)
    return mean, std, sharpe, total


def equity_metrics_from_curve(
    equity: np.ndarray,
    returns: np.ndarray | None = None,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> dict:
    equity = np.asarray(equity, dtype=float)

    start_eq = float(equity[0])
    final_eq = float(equity[-1])
    cum_return = final_eq / start_eq - 1.0

    running_max = np.maximum.accumulate(equity)
    drawdowns = equity / running_max - 1.0
    max_dd = float(drawdowns.min())

    if returns is None:
        rets = np.diff(equity) / equity[:-1]
    else:
        rets = np.asarray(returns, dtype=float)

    if rets.size >= 2:
        vol = float(np.std(rets, ddof=1))
    else:
        vol = float("nan")

    if annualize and np.isfinite(vol):
        vol *= np.sqrt(periods_per_year)

    return {
        "StartEquity": start_eq,
        "FinalEquity": final_eq,
        "CumReturn": float(cum_return),
        "Volatility": vol,
        "MaxDD": max_dd,
    }


# =============================================================================
# Experiment runner
# =============================================================================
def run_experiment(
    combined_df: pd.DataFrame,
    target_col: str = "AMD",
    seq_len: int = 30,
    window_sizes: tuple[int, ...] = (5, 10, 15, 20, 30),
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    delta_discrete: float = 0.2,
    show_plots: bool = False,
    max_indicators_to_plot: int = 4,
    # trading cost parameters
    cost: float = 5e-4,
    kappa: float = 5.0,
    fee: float = 0.5,
    C0: float = 10000.0,
    lambda_short: float = 1.5,
):
    torch.manual_seed(42)

    # Build tensors (also returns aligned prices/index for discrete simulation & plotting)
    X, ind_bank, y, d_input, num_indicators, price_filtered, sample_dates = build_tensors_from_combined_df(
        combined_df=combined_df,
        target_col=target_col,
        seq_len=seq_len,
        window_sizes=window_sizes,
        scale_X=True,
        scale_indicators=True,
        train_split=0.8,  # scaling split inside Base (kept for compatibility)
    )

    time_index = pd.DatetimeIndex(sample_dates)
    years = time_index.year.values

    # ---- YEAR SPLIT (Same benchmarks as Greedy) ----
    train_years = np.array([2016, 2017, 2018, 2019, 2020, 2021])
    val_years   = np.array([2022, 2023])
    test_years  = np.array([2024, 2025, 2026])

    train_mask = np.isin(years, train_years)
    val_mask   = np.isin(years, val_years)
    test_mask  = np.isin(years, test_years)

    # Safety checks (no overlap + non-empty)
    if np.any(train_mask & val_mask) or np.any(train_mask & test_mask) or np.any(val_mask & test_mask):
        raise RuntimeError("Year masks overlap. Fix train/val/test years.")
    if not (train_mask.any() and val_mask.any() and test_mask.any()):
        raise RuntimeError(
            f"Empty split detected. Counts: train={train_mask.sum()}, val={val_mask.sum()}, test={test_mask.sum()}. "
            "Check available years in your dataset."
        )

    train_ds = TradingDataset(X[train_mask], [b[train_mask] for b in ind_bank], y[train_mask])
    val_ds   = TradingDataset(X[val_mask],   [b[val_mask]   for b in ind_bank], y[val_mask])
    test_ds  = TradingDataset(X[test_mask],  [b[test_mask]  for b in ind_bank], y[test_mask])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    print(
        f"Split sizes | train={len(train_ds)} (years {train_years.min()}-{train_years.max()}) | "
        f"val={len(val_ds)} (years {val_years.min()}-{val_years.max()}) | "
        f"test={len(test_ds)} (years {test_years.min()}-{test_years.max()})"
    )

    # Model + optimizer
    model = LSTMTradingAgent(
        d_input=d_input,
        d_hidden=64,
        num_indicators=num_indicators,
        window_sizes=list(window_sizes),
        tau=1.0,
        eps_sigmoid=0.1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train history
    history = []
    best_val_sharpe = -np.inf
    best_epoch = None
    best_state = None

    for ep in range(1, epochs + 1):
        tr = train_one_epoch_hybrid(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            cost=cost,
            kappa=kappa,
        )
        va = evaluate_epoch(model=model, loader=val_loader, device=device, cost=cost, kappa=kappa)
        te = evaluate_epoch(model=model, loader=test_loader, device=device, cost=cost, kappa=kappa)

        history.append({
            "epoch": ep,
            "train_mean": tr["mean"], "train_std": tr["std"], "train_sharpe": tr["sharpe"], "train_total": tr["total_return"],
            "val_mean":   va["mean"], "val_std":   va["std"], "val_sharpe":   va["sharpe"], "val_total":   va["total_return"],
            "test_mean":  te["mean"], "test_std":  te["std"], "test_sharpe":  te["sharpe"], "test_total":  te["total_return"],
        })

        print(
            f"Epoch {ep:03d} | "
            f"Train: mean={tr['mean']:.6f} std={tr['std']:.6f} sharpe={tr['sharpe']:.6f} | "
            f"Val:   mean={va['mean']:.6f} std={va['std']:.6f} sharpe={va['sharpe']:.6f} | "
            f"Test:  mean={te['mean']:.6f} std={te['std']:.6f} sharpe={te['sharpe']:.6f}"
        )

        if va["sharpe"] > best_val_sharpe:
            best_val_sharpe = va["sharpe"]
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    results_df = pd.DataFrame(history)

    print("\n=== Epoch Results (last 10) ===")
    print(results_df.tail(10).to_string(index=False))

    print("\n=== Best Epoch by VAL Sharpe ===")
    best_row = results_df.loc[results_df["val_sharpe"].idxmax()]
    print(best_row.to_string())

    # Restore best model (by val sharpe)
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nRestored best model weights from epoch {best_epoch} (val_sharpe={best_val_sharpe:.6f}).")

    # Optional plots: training curves
    # plot_history(results_df, show=show_plots)

    # ---- Series on test set ----
    decisions_cont, strat_rets_cont, asset_rets = get_series_on_loader(
        model=model, loader=test_loader, device=device, delta=None, cost=cost, kappa=kappa,
        return_actions=True, return_asset=True
    )
    actions_disc, strat_rets_disc, _ = get_series_on_loader(
        model=model, loader=test_loader, device=device, delta=delta_discrete, cost=cost, kappa=kappa,
        return_actions=True, return_asset=False
    )

    # Optional plots (disabled by default to keep Spyder stable)
    # plot_equity_from_returns(strat_rets_cont, asset_rets, name="continuous", show=show_plots)
    # plot_distribution(strat_rets_cont, name="continuous", show=show_plots)
    # plot_series(decisions_cont, "Decision D(t) on Test (Continuous)", "decision_continuous.png", "D(t)", show=show_plots)
    # plot_equity_from_returns(strat_rets_disc, asset_rets, name=f"discrete_delta_{delta_discrete}", show=show_plots)
    # plot_distribution(strat_rets_disc, name=f"discrete_delta_{delta_discrete}", show=show_plots)
    # plot_series(actions_disc, f"Action a(t) on Test (Discrete δ={delta_discrete})",
    #             "action_discrete.png", "a(t)", show=show_plots)

    # Summary table (test, discrete)
    s_mean, s_std, s_sh, s_tot = summarize_returns(strat_rets_disc)
    b_mean, b_std, b_sh, b_tot = summarize_returns(asset_rets)

    summary = pd.DataFrame([
        {"Series": f"Strategy (δ={delta_discrete})", "Mean": s_mean, "Std": s_std, "Sharpe": s_sh, "TotalReturn": s_tot},
        {"Series": "BuyHold", "Mean": b_mean, "Std": b_std, "Sharpe": b_sh, "TotalReturn": b_tot},
    ])

    print("\n=== Test Summary (Discrete Trading Returns) ===")
    print(summary.to_string(index=False))

    # ---- Discrete capital simulation (aligned to test mask) ----
    # price_filtered is aligned to sample_dates; so just take the test slice.
    price_test = np.asarray(price_filtered, dtype=float).reshape(-1)[test_mask]

    T_sim = min(len(price_test), len(actions_disc))
    if T_sim >= 2:
        C = simulate_portfolio_discrete(
            prices=price_test[:T_sim],
            decisions=actions_disc[:T_sim],  # already {-1,0,+1}
            delta=delta_discrete,
            C0=C0,
            fee=fee,
            lambda_short=lambda_short,
        )
        plot_capital_curve(C, show=show_plots, filename="capital_curve_discrete_test.png")

        eq_metrics = equity_metrics_from_curve(
            equity=C,
            returns=strat_rets_disc[: min(len(strat_rets_disc), len(C))],
            annualize=True,
            periods_per_year=252,
        )
        eq_table = pd.DataFrame([{"Series": f"Strategy (Discrete δ={delta_discrete})", **eq_metrics}])

        print("\n=== Test Equity-Curve Metrics (Discrete Strategy) ===")
        print(eq_table.to_string(index=False))
    else:
        print("\n[warn] Not enough test prices for discrete capital simulation.")

    # ---- Deliverables: λ_i(t) and θ_i(t) on test ----
    # D_series, lambda_series, th_plus, th_minus = collect_diagnostics(model, test_loader, device=device, window_sizes=list(window_sizes))
    # plot_lambdas(lambda_series, max_indicators_to_plot=max_indicators_to_plot, show=show_plots)
    # plot_thresholds(th_plus, th_minus, max_indicators_to_plot=max_indicators_to_plot, show=show_plots)

    return model, results_df, summary


if __name__ == "__main__":
    collected = load_collected_data("./data")
    combined_df = build_combined_price_df(collected)

    run_experiment(
        combined_df=combined_df,
        target_col="AMD",
        seq_len=30,
        window_sizes=(10, 15, 20, 25, 30),
        epochs=20,
        batch_size=32,
        lr=1e-3,
        delta_discrete=0.2,
        show_plots=False,
        max_indicators_to_plot=4,
        cost=5e-4,
        kappa=5.0,
        fee=0.5,
        C0=10000.0,
        lambda_short=1.5,
    )
