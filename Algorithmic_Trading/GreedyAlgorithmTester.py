# Greedy Optimization Test Module

from GreedyOptimizationAlgorithm import run_greedy_for_symbol, default_grid_params
from metric_utils import compute_equity_metrics
import pandas as pd

import os
from plot_utils import plot_equity_curve, plot_price_with_entries_exits


def main():
    # -------- Choose a symbol you want to test either INTC/AMD----------------
    symbol = "INTC"
    pickle_path = "./data/collected_data.pkl"

    # ----------Create custom grid parameters------------------------
    custom_grids = default_grid_params(
    ema_L=(5, 10, 20, 50, 100),
    ema_theta=(0.0, 0.0005, 0.001, 0.002, 0.005),
    rsi_L=(7, 14, 21),
    rsi_levels=((20, 80), (30, 70), (40, 60)),
    bb_L=(10, 20, 50),
    bb_k=(1.5, 2.0, 2.5),
    bb_theta=(0.0, 0.001, 0.002),
    )
    
    # Second Grid : grid2 > assingned 
    custom_grids2 = default_grid_params(
    ema_L=(3, 5, 8, 13, 21),
    ema_theta=(0.0, 0.00025, 0.0005, 0.001),
    rsi_L=(5, 7, 10, 14),
    rsi_levels=((15, 85), (20, 80), (25, 75), (30, 70)),
    bb_L=(5, 10, 15, 20),
    bb_k=(1.2, 1.5, 1.8, 2.0),
    bb_theta=(0.0, 0.0005, 0.001, 0.002),
    )
    
    # Third Grid: grid3 
    custom_grids3 = default_grid_params(
    ema_L=(8, 10, 15, 20, 30, 50),
    ema_theta=(0.0, 0.0005, 0.001, 0.0015, 0.0025),
    rsi_L=(10, 14, 21),
    rsi_levels=((25, 75), (30, 70), (35, 65)),
    bb_L=(15, 20, 30, 50),
    bb_k=(1.5, 2.0, 2.25),
    bb_theta=(0.0, 0.0005, 0.001, 0.0015),
    )
    

    # -------- Run the greedy optimization ---------------------
    result = run_greedy_for_symbol(
        symbol=symbol,
        pickle_path=pickle_path,
        initial_capital=10000.0,
        fee=0.5,
        allow_short=True,
        lambda_short=1.5,
        annualization=252,
        max_indicators=5,
        grids=custom_grids2,
        allowed_indicators=["MACD", "BBANDS", "EMA", "RSI"],
        forced_indicators=["EMA", "RSI"],
        # NEW SPLIT:
        train_years=(2016, 2017, 2018, 2019, 2020, 2021),
        val_years=(2022, 2023),
        test_years=(2024, 2025, 2026),
    )

    # -------- Compute extra metrics from equity curves --------
    train_metrics = compute_equity_metrics(result["equity_train"], annualization=252, rf=0.0)
    val_metrics = compute_equity_metrics(result["equity_val"], annualization=252, rf=0.0) if len(result["equity_val"]) else {}
    test_metrics = compute_equity_metrics(result["equity_test"], annualization=252, rf=0.0) if len(result["equity_test"]) else {}

    # ----- Print results -----
    print("\n============================")
    print("GREEDY OPTIMIZATION RESULTS")
    print("============================")
    print("Symbol:", result["symbol"])
    print("Chosen indicators:", result["chosen_indicators"])
    print("Chosen params:")
    for k in result["chosen_indicators"]:
        print(" ", k, "->", result["chosen_params"][k])

    print("\nTrain Sharpe (greedy internal):", result["train_sharpe_greedy"])
    print("Train Sharpe (recomputed):", result["train_sharpe"])
    print("Val Sharpe:", result["val_sharpe"])
    print("Test Sharpe:", result["test_sharpe"])

    print("\n============================")
    print("METRICS FROM EQUITY CURVES")
    print("============================")

    print("\nTRAIN METRICS:")
    for k, v in train_metrics.items():
        print(f"  {k}: {v}")

    if val_metrics:
        print("\nVAL METRICS:")
        for k, v in val_metrics.items():
            print(f"  {k}: {v}")
    else:
        print("\nVAL METRICS: (empty / not available)")

    if test_metrics:
        print("\nTEST METRICS:")
        for k, v in test_metrics.items():
            print(f"  {k}: {v}")
    else:
        print("\nTEST METRICS: (empty / not available)")

    turnover_train = (result["sig_train"].diff().abs().fillna(0) > 0).sum()
    turnover_val = (result["sig_val"].diff().abs().fillna(0) > 0).sum() if len(result["sig_val"]) else 0
    turnover_test = (result["sig_test"].diff().abs().fillna(0) > 0).sum() if len(result["sig_test"]) else 0

    print("\nTurnover train (position changes):", int(turnover_train))
    print("Turnover val   (position changes):", int(turnover_val))
    print("Turnover test  (position changes):", int(turnover_test))

    # =============================================================================
    # PLOTTING
    # =============================================================================
    out_dir = "./results"
    os.makedirs(out_dir, exist_ok=True)

    # -------- Table row ---------------------------------------
    row = {
        "symbol": result["symbol"],
        "chosen_indicators": ",".join(result["chosen_indicators"]),
        "chosen_params": str(result["chosen_params"]),

        "train_sharpe": train_metrics.get("sharpe", None),
        "train_cum_return": train_metrics.get("cum_return", None),
        "train_volatility": train_metrics.get("volatility", None),
        "train_max_drawdown": train_metrics.get("max_drawdown", None),

        "val_sharpe": val_metrics.get("sharpe", None),
        "val_cum_return": val_metrics.get("cum_return", None),
        "val_volatility": val_metrics.get("volatility", None),
        "val_max_drawdown": val_metrics.get("max_drawdown", None),

        "test_sharpe": test_metrics.get("sharpe", None),
        "test_cum_return": test_metrics.get("cum_return", None),
        "test_volatility": test_metrics.get("volatility", None),
        "test_max_drawdown": test_metrics.get("max_drawdown", None),
    }

    df_out = pd.DataFrame([row])

    print("\n============================")
    print("RESULTS TABLE")
    print("============================")
    print(df_out.to_string(index=False))

    csv_path = os.path.join(out_dir, "greedy_results.csv")
    df_out.to_csv(csv_path, index=False)
    print("Saved results table:", csv_path)

    # -------- Equity plots ------------------------------------
    plot_equity_curve(
        result["equity_train"],
        f"{symbol} Equity (Train)",
        os.path.join(out_dir, f"{symbol}_equity_train.png")
    )

    if len(result["equity_val"]):
        plot_equity_curve(
            result["equity_val"],
            f"{symbol} Equity (Val)",
            os.path.join(out_dir, f"{symbol}_equity_val.png")
        )

    if len(result["equity_test"]):
        plot_equity_curve(
            result["equity_test"],
            f"{symbol} Equity (Test)",
            os.path.join(out_dir, f"{symbol}_equity_test.png")
        )

    # -------- Price + entry/exit plots ------------------------
    plot_price_with_entries_exits(
        result["price_train"],
        result["sig_train"],
        f"{symbol} Train Price with Entries/Exits",
        os.path.join(out_dir, f"{symbol}_train_entries_exits.png")
    )

    if len(result["price_val"]):
        plot_price_with_entries_exits(
            result["price_val"],
            result["sig_val"],
            f"{symbol} Val Price with Entries/Exits",
            os.path.join(out_dir, f"{symbol}_val_entries_exits.png")
        )

    if len(result["price_test"]):
        plot_price_with_entries_exits(
            result["price_test"],
            result["sig_test"],
            f"{symbol} Test Price with Entries/Exits",
            os.path.join(out_dir, f"{symbol}_test_entries_exits.png")
        )

    print("Saved plots to:", out_dir)


if __name__ == "__main__":
    main()