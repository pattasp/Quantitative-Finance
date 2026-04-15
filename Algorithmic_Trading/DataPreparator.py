# ============= TIME SERIES DATA PREPARATION ========================
# This module loads the collected OHLCV dataset (pickle), synchronizes symbols on common dates,
# builds the combined "price" DataFrame (one column per asset), converts it to supervised-learning
# format (lags + future target), and saves CSVs.
#
# YEAR SPLITS (as requested):
#   Train: 2016–2021
#   Val:   2022–2023
#   Test:  2024–2026
#
# Key properties:
# - No side effects on import (no auto-loading / auto-running unless run as script).
# - Robust price-column selection via priority list ("Adj Close" then "Close", etc.).
# - Saves separate TRAIN/VAL/TEST CSVs for single-series and multi-series supervised datasets.

import os
import pickle
from typing import Iterable, Tuple, List, Optional, Dict

import pandas as pd


# =============================================================================
# Load collected_data dictionary from the DataCollector pickle file
# =============================================================================

def load_collected_data(directory: str = "./data", filename: str = "collected_data.pkl"):
    path = os.path.join(directory, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No cached dataset found at: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


# =============================================================================
# Synchronize all DataFrames in collected_data so they share the same DateTimeIndex
# and keep only intersected (common) dates.
# =============================================================================

def synchronize_dataframes(collected_data):
    all_dataframes = []

    for group_data in collected_data.values():
        for df in group_data.values():
            all_dataframes.append(df)

    if not all_dataframes:
        return collected_data

    # Ensure DateTimeIndex + sorted index
    for i in range(len(all_dataframes)):
        if not isinstance(all_dataframes[i].index, pd.DatetimeIndex):
            all_dataframes[i].index = pd.to_datetime(all_dataframes[i].index)
        all_dataframes[i] = all_dataframes[i].sort_index()

    # Intersect indices
    common_index = all_dataframes[0].index
    for df in all_dataframes[1:]:
        common_index = common_index.intersection(df.index)

    # Reindex each DF to common index
    for group, group_data in collected_data.items():
        for symbol, df in group_data.items():
            group_data[symbol] = df.reindex(common_index)

    return collected_data


# =============================================================================
# Build one DataFrame that will contain a chosen "price" series for all assets
# =============================================================================

def _pick_price_column(df: pd.DataFrame, priority: Iterable[str]) -> str:
    for col in priority:
        if col in df.columns:
            return col
    raise ValueError(
        f"None of the requested price columns exist. Tried: {list(priority)}. "
        f"Available columns: {list(df.columns)}"
    )


def build_combined_price_df(
    collected_data,
    price_col_priority: Tuple[str, ...] = ("Adj Close", "Close", "close"),
    rename_map: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """Build a combined DataFrame with DateTimeIndex and 1 column per asset.

    For each symbol, picks the first available column from `price_col_priority`.
    """
    price_data = {}

    for group, group_data in collected_data.items():
        for symbol, df in group_data.items():
            col = _pick_price_column(df, price_col_priority)
            price_data[symbol] = df[col]

    combined_df = pd.DataFrame(price_data)

    if rename_map is not None:
        combined_df = combined_df.rename(columns=rename_map)

    return combined_df


# =============================================================================
# Create supervised learning dataset (target shifted to the future + lagged features)
# =============================================================================

def create_supervised_dataset(
    df: pd.DataFrame,
    target_col: str,
    future_dt: int,
    past_dt: int,
    multiseries: bool = False,
    date_col: str = "Date"
) -> pd.DataFrame:
    """Transform a time-series DataFrame into a supervised dataset.

    Output format: [Date, target(t+future_dt), lag features ...]
    """
    if future_dt <= 0:
        raise ValueError("future_dt must be positive.")
    if past_dt < 0:
        raise ValueError("past_dt must be >= 0.")
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found in df columns.")

    # Keep Date column unlagged (if it exists)
    date_series = df[date_col].copy() if date_col in df.columns else None

    # Future target
    y = df[target_col].shift(-future_dt)
    y.name = f"{target_col}_t+{future_dt}"

    # Which columns to lag
    if multiseries:
        columns_to_lag = [col for col in df.columns if col != date_col]
    else:
        columns_to_lag = [target_col]

    lagged_features = []
    for col in columns_to_lag:
        for lag in range(past_dt + 1):
            col_lag_name = f"{col}_t-{lag}"
            lagged_features.append(df[col].shift(lag).rename(col_lag_name))

    supervised_df = pd.concat([y] + lagged_features, axis=1)
    supervised_df.dropna(inplace=True)

    if date_series is not None:
        supervised_df[date_col] = date_series

    if date_col in supervised_df.columns:
        col_order = [date_col, y.name] + [c for c in supervised_df.columns if c not in (date_col, y.name)]
        supervised_df = supervised_df[col_order]

    supervised_df = supervised_df.reset_index(drop=True)
    return supervised_df


# =============================================================================
# Save helper
# =============================================================================

def save_dataframe(df: pd.DataFrame, data_directory: str, datafile: str) -> str:
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
        print(f"Directory {data_directory} created.")
    data_path = os.path.join(data_directory, datafile)
    df.to_csv(data_path, index=False)
    print(f"DataFrame saved to {data_path}.")
    return data_path


# =============================================================================
# Minor helper: convert DateTimeIndex to a column named 'Date'
# =============================================================================

def combined_to_case_study_format(combined_df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    out = combined_df.reset_index()
    out = out.rename(columns={out.columns[0]: date_col})
    return out


# =============================================================================
# Split helpers: by years (train/val/test)
# =============================================================================

def split_by_years_3way(
    df: pd.DataFrame,
    date_col: str = "Date",
    train_years: Tuple[int, ...] = (2016, 2017, 2018, 2019, 2020, 2021),
    val_years: Tuple[int, ...] = (2022, 2023),
    test_years: Tuple[int, ...] = (2024, 2025, 2026),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    if date_col not in out.columns:
        raise ValueError(f"split_by_years_3way requires a '{date_col}' column.")
    out[date_col] = pd.to_datetime(out[date_col])
    years = out[date_col].dt.year

    train_df = out[years.isin(train_years)].reset_index(drop=True)
    val_df   = out[years.isin(val_years)].reset_index(drop=True)
    test_df  = out[years.isin(test_years)].reset_index(drop=True)

    return train_df, val_df, test_df


def _years_label(years: Tuple[int, ...]) -> str:
    if not years:
        return "NONE"
    ys = sorted(set(int(y) for y in years))
    return f"{ys[0]}_{ys[-1]}" if len(ys) > 1 else f"{ys[0]}"


# =============================================================================
# MAIN PIPELINE SECTION
# =============================================================================

def run_preparation_pipeline(
    data_directory: str = "./data",
    pickle_filename: str = "collected_data.pkl",
    future_dt: int = 1,
    past_dt: int = 200,
    price_col_priority: Tuple[str, ...] = ("Adj Close", "Close", "close"),
    split_years: bool = True,
    train_years: Tuple[int, ...] = (2016, 2017, 2018, 2019, 2020, 2021),
    val_years: Tuple[int, ...] = (2022, 2023),
    test_years: Tuple[int, ...] = (2024, 2025, 2026),
) -> Tuple[pd.DataFrame, List[str]]:
    """End-to-end preparation pipeline.

    Steps:
      1) load pickle
      2) synchronize
      3) build combined price df
      4) DateTimeIndex -> 'Date' column
      5) create supervised datasets per series (single & multi)
      6) save CSVs (either full, or train/val/test split by years)

    Returns:
      combined_df (Date + one column per asset)
      saved_files (list of saved CSV paths)
    """
    collected_data = load_collected_data(directory=data_directory, filename=pickle_filename)
    synchronize_dataframes(collected_data)

    combined_df = build_combined_price_df(
        collected_data,
        price_col_priority=price_col_priority,
        rename_map=None
    )
    combined_df = combined_to_case_study_format(combined_df, date_col="Date")

    columns_to_process = [c for c in combined_df.columns if c != "Date"]
    saved_files: List[str] = []

    train_lbl = _years_label(train_years)
    val_lbl   = _years_label(val_years)
    test_lbl  = _years_label(test_years)

    for target_col in columns_to_process:
        dataset = create_supervised_dataset(
            combined_df, target_col, future_dt, past_dt, multiseries=False, date_col="Date"
        )
        multi_dataset = create_supervised_dataset(
            combined_df, target_col, future_dt, past_dt, multiseries=True, date_col="Date"
        )

        safe = target_col.replace("/", "_").replace("\\", "_")

        if split_years:
            tr, va, te = split_by_years_3way(
                dataset, date_col="Date", train_years=train_years, val_years=val_years, test_years=test_years
            )
            trm, vam, tem = split_by_years_3way(
                multi_dataset, date_col="Date", train_years=train_years, val_years=val_years, test_years=test_years
            )

            saved_files.append(save_dataframe(tr,  data_directory, f"{safe}_TRAIN_{train_lbl}.csv"))
            saved_files.append(save_dataframe(va,  data_directory, f"{safe}_VAL_{val_lbl}.csv"))
            saved_files.append(save_dataframe(te,  data_directory, f"{safe}_TEST_{test_lbl}.csv"))

            saved_files.append(save_dataframe(trm, data_directory, f"{safe}_MULTI_TRAIN_{train_lbl}.csv"))
            saved_files.append(save_dataframe(vam, data_directory, f"{safe}_MULTI_VAL_{val_lbl}.csv"))
            saved_files.append(save_dataframe(tem, data_directory, f"{safe}_MULTI_TEST_{test_lbl}.csv"))
        else:
            saved_files.append(save_dataframe(dataset, data_directory, f"{safe}_time_series_data.csv"))
            saved_files.append(save_dataframe(multi_dataset, data_directory, f"{safe}_multi_time_series_data.csv"))

    return combined_df, saved_files


if __name__ == "__main__":
    # Default split:
    #   Train: 2016-2021
    #   Val:   2022-2023
    #   Test:  2024-2026
    combined_df, saved_files = run_preparation_pipeline(
        data_directory="./data",
        pickle_filename="collected_data.pkl",
        future_dt=1,
        past_dt=200,
        price_col_priority=("Adj Close", "Close", "close"),
        split_years=True,
        train_years=(2016, 2017, 2018, 2019, 2020, 2021),
        val_years=(2022, 2023),
        test_years=(2024, 2025, 2026),
    )
    print(f"Saved {len(saved_files)} files.")
# EXTRA NOTES : =========

# IF YOU GET THE ERROR MESSAGE : Pickle file not found at ./data/collected_data.pkl'

# THEN WRITE ON YOUR CONSOLE : 
    # > import os 
    # > print(os.getcwd()) 
    # to see on which current directory you are applying the Pipeline
    
    # Then check if the data exist in a file with the command :
    # print(os.listdir("./data") if os.path.exists("./data") else "No ./data folder here")
    

