import os
import sys
import logging
from typing import Dict, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import RangeSlider, Button
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

###############################################################################
# Logging Setup
###############################################################################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)


###############################################################################
# Utility Functions
###############################################################################
def resource_path(relative_path: str) -> str:
    """
    Get absolute path to resource, works for dev and for PyInstaller.

    Args:
        relative_path (str): Relative path to the resource.

    Returns:
        str: Absolute path to the resource.
    """
    try:
        base_path = sys._MEIPASS  # type: ignore
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def load_and_filter_csv(filepath: str,
                        start_date: str,
                        end_date: str) -> pd.DataFrame:
    """
    Load CSV data, filter by date, and select the 'close' column.
    Assumes the CSV has a 'time' column to parse dates.

    Args:
        filepath (str): Path to the CSV file.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: Filtered DataFrame with a DatetimeIndex and 'close' column.
    """
    # Ensure path is correct for PyInstaller bundling
    filepath = resource_path(filepath)

    if not os.path.exists(filepath):
        logging.error(f"File not found: {filepath}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(filepath, parse_dates=['time'])
        df = df.sort_values('time').set_index('time')
        df = df.loc[start_date:end_date, ['close']]
        return df
    except Exception as e:
        logging.exception(f"Error loading {filepath}: {e}")
        return pd.DataFrame()


def compute_lag_correlations(series_m2: pd.Series,
                             series_asset: pd.Series,
                             max_lag: int,
                             method: str = 'pearson') -> Dict[int, float]:
    """
    For each lag in [1..max_lag], shift the asset by -lag (so we compare M2(t)
    with Asset(t+lag)) and compute a single correlation across the entire dataset.

    Args:
        series_m2 (pd.Series): M2 time series.
        series_asset (pd.Series): Asset time series.
        max_lag (int): Maximum lag in days.
        method (str): Correlation method ('pearson', 'spearman', etc.).

    Returns:
        Dict[int, float]: Dictionary of {lag -> correlation_value}.
    """
    correlations = {}
    for lag in range(1, max_lag + 1):
        # Shift the asset by -lag
        shifted_asset = series_asset.shift(-lag)
        # Align
        df_aligned = pd.concat([
            series_m2.rename("m2"),
            shifted_asset.rename("asset")
        ], axis=1).dropna()

        if not df_aligned.empty:
            correlations[lag] = df_aligned["m2"].corr(df_aligned["asset"], method=method)
        else:
            correlations[lag] = np.nan  # or 0
    return correlations


def linear_regression_inset(ax_parent: plt.Axes,
                            series_m2: pd.Series,
                            series_asset: pd.Series,
                            best_lag: int) -> None:
    """
    Create a small inset inside ax_parent that shows a scatter of M2 vs. Asset
    (using the best_lag shift) plus a linear regression line.

    Args:
        ax_parent (plt.Axes): The main Axes object where the inset will be placed.
        series_m2 (pd.Series): M2 time series.
        series_asset (pd.Series): Asset time series.
        best_lag (int): The lag (days) that yields the highest correlation.
    """
    # Build an inset axis
    ax_inset = inset_axes(
        ax_parent,
        width="30%",  # 30% of parent axis
        height="30%",
        loc="upper right",  # top-right corner
        borderpad=1.5
    )

    # Shift asset by -best_lag so we compare M2(t) with Asset(t+lag)
    shifted_asset = series_asset.shift(-best_lag)
    df_aligned = pd.concat([
        series_m2.rename("m2"),
        shifted_asset.rename("asset")
    ], axis=1).dropna()

    if df_aligned.empty:
        ax_inset.text(0.5, 0.5, "No Data", ha='center', va='center',
                      transform=ax_inset.transAxes, fontsize=8)
        return

    x_vals = df_aligned["m2"].values
    y_vals = df_aligned["asset"].values

    # Scatter
    ax_inset.scatter(x_vals, y_vals, s=10, c="blue", alpha=0.5)

    # Linear regression
    slope, intercept = np.polyfit(x_vals, y_vals, deg=1)
    x_line = np.array([x_vals.min(), x_vals.max()])
    y_line = slope * x_line + intercept
    ax_inset.plot(x_line, y_line, color="red", linewidth=2)

    # Minimal ticks / labels
    ax_inset.set_title("LinReg (M2 vs Asset)", fontsize=8)
    ax_inset.tick_params(axis='both', which='major', labelsize=7)
    ax_inset.spines["right"].set_visible(False)
    ax_inset.spines["top"].set_visible(False)


###############################################################################
# Configuration
###############################################################################
START_DATE = "2018-01-01"
END_DATE = "2025-12-31"
MAX_LAG = 180  # Maximum lag in days
CORR_METHOD = 'pearson'
# CSV Paths (relative or absolute)
CSV_PATHS = {
    "M2": "data/Global Liquity.csv",
    "BTC": "data/Bitcoin Price.csv",
    "NASDAQ": "data/NASDAQ Price.csv",
    "US10Y": "data/US10Y.csv",
    "DXY": "data/DXY.csv",
    "DJI": "data/DJI Price.csv",
    "Chainlink": "data/Chainlink Price.csv"
}

###############################################################################
# Main Data Loading
###############################################################################
logging.info("Loading data from CSV files...")

df_m2 = load_and_filter_csv(CSV_PATHS["M2"], START_DATE, END_DATE).rename(columns={'close': 'close_m2'})
df_btc = load_and_filter_csv(CSV_PATHS["BTC"], START_DATE, END_DATE).rename(columns={'close': 'close_btc'})
df_nasdaq = load_and_filter_csv(CSV_PATHS["NASDAQ"], START_DATE, END_DATE).rename(columns={'close': 'close_nasdaq'})
df_us10y = load_and_filter_csv(CSV_PATHS["US10Y"], START_DATE, END_DATE).rename(columns={'close': 'close_us10y'})
df_dxy = load_and_filter_csv(CSV_PATHS["DXY"], START_DATE, END_DATE).rename(columns={'close': 'close_dxy'})
df_dji = load_and_filter_csv(CSV_PATHS["DJI"], START_DATE, END_DATE).rename(columns={'close': 'close_dji'})
df_chainlink = load_and_filter_csv(CSV_PATHS["Chainlink"], START_DATE, END_DATE).rename(
    columns={'close': 'close_chainlink'})

assets = {
    "BTC": df_btc["close_btc"],
    "NASDAQ": df_nasdaq["close_nasdaq"],
    "US10Y": df_us10y["close_us10y"],
    "DXY": df_dxy["close_dxy"],
    "DJI": df_dji["close_dji"],
    "Chainlink": df_chainlink["close_chainlink"]
}

# Sort all data by index just to be consistent
df_m2.sort_index(inplace=True)
for k, v in assets.items():
    assets[k] = v.sort_index()

# Combine all date indices for slider range
all_dates = df_m2.index
for series in assets.values():
    all_dates = all_dates.union(series.index)

min_date = all_dates.min()
max_date = all_dates.max()
proj_max_date = max_date + pd.Timedelta(days=MAX_LAG)

start_date = min_date
min_day = 0
max_day = (max_date - start_date).days
proj_max_day = max_day + MAX_LAG

###############################################################################
# Figure & Subplots
###############################################################################
fig, (ax_corr, ax_main) = plt.subplots(1, 2, figsize=(16, 8))
fig.subplots_adjust(
    left=0.07, right=0.97,
    top=0.88, bottom=0.12,
    wspace=0.30
)

ax_main_r = ax_main.twinx()

###############################################################################
# Slider Setup
###############################################################################
slider_ax = fig.add_axes([0.15, 0.05, 0.70, 0.03])
range_slider = RangeSlider(
    slider_ax, "Days from Start",
    min_day, proj_max_day,
    valinit=(min_day, proj_max_day)
)

###############################################################################
# Button Setup
###############################################################################
button_specs = [
    ("BTC", [0.05, 0.92, 0.09, 0.05]),
    ("NASDAQ", [0.16, 0.92, 0.09, 0.05]),
    ("US10Y", [0.27, 0.92, 0.09, 0.05]),
    ("DXY", [0.38, 0.92, 0.09, 0.05]),
    ("DJI", [0.49, 0.92, 0.09, 0.05]),
    ("Chainlink", [0.60, 0.92, 0.09, 0.05]),
]

buttons = {}
for label, rect in button_specs:
    ax_btn = fig.add_axes(rect)
    buttons[label] = Button(ax_btn, label)

###############################################################################
# Caching for Correlations
###############################################################################
correlation_cache: Dict[str, Dict[int, float]] = {}


def get_correlations(asset_label: str) -> Tuple[int, float, Dict[int, float]]:
    """
    Retrieve or compute correlations for a given asset vs. M2, and return
    the optimal lag, the optimal correlation, and the dictionary of all correlations.

    Args:
        asset_label (str): The name of the asset (e.g. "BTC").

    Returns:
        (int, float, Dict[int, float]): (best_lag, best_corr, all_correlations)
    """
    if asset_label in correlation_cache:
        # Already computed
        correlations = correlation_cache[asset_label]
    else:
        # Compute and cache
        correlations = compute_lag_correlations(
            df_m2["close_m2"], assets[asset_label],
            max_lag=MAX_LAG, method=CORR_METHOD
        )
        correlation_cache[asset_label] = correlations

    if correlations:
        best_lag = max(correlations, key=correlations.get)
        best_corr = correlations[best_lag]
    else:
        best_lag = 0
        best_corr = float('nan')

    return best_lag, best_corr, correlations


###############################################################################
# Plotting Logic
###############################################################################
def plot_asset_vs_m2(asset_label: str) -> None:
    """
    1) Clear old lines on both subplots, but keep the axes themselves.
    2) Retrieve or compute correlation vs. lag => best lag.
    3) Plot correlation vs. lag on ax_corr.
    4) Insert a small linear regression chart (M2 vs. Asset at best lag).
    5) Shift M2 by best lag, build ratio overlay on ax_main/ax_main_r.
    6) Update x-limits from slider.

    Args:
        asset_label (str): The name of the asset to plot (e.g. "BTC").
    """
    ax_corr.cla()
    ax_main.cla()
    ax_main_r.cla()

    series_asset = assets[asset_label].dropna()
    series_m2_data = df_m2["close_m2"].dropna()

    # 2) Get correlations (cached), find best lag
    best_lag, best_corr, correlations = get_correlations(asset_label)
    logging.info(f"Asset: {asset_label}, Best Lag: {best_lag}, Corr: {best_corr:.4f}")

    # 3) Plot correlation vs. lag
    lags = sorted(correlations.keys())
    corr_vals = [correlations[lag] for lag in lags]

    ax_corr.plot(lags, corr_vals, marker='o', color='tab:blue')
    ax_corr.axvline(best_lag, color='red', linestyle='--',
                    label=f'Optimal Lag: {best_lag} days')
    ax_corr.set_xlabel("Lag (days)", fontsize=11)
    ax_corr.set_ylabel("Correlation", fontsize=11)
    ax_corr.set_title(f"Correlation vs. Lag\n({asset_label} vs. M2)", fontsize=13, pad=12)
    ax_corr.legend(fontsize=10)
    ax_corr.grid(True)

    # Annotate correlation in the top-left
    ax_corr.annotate(
        f"Best Lag: {best_lag}\nCorr: {best_corr:.2f}",
        xy=(0.03, 0.97),
        xycoords='axes fraction',
        ha='left', va='top',
        fontsize=12,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', alpha=0.8)
    )

    # 4) Inset linear regression chart
    linear_regression_inset(ax_corr, series_m2_data, series_asset, best_lag)

    # 5) Plot ratio overlay
    asset_ratio = series_asset / series_asset.iloc[0]

    df_m2_shifted = series_m2_data.copy()
    # Shift index by best_lag days
    df_m2_shifted.index = df_m2_shifted.index + pd.Timedelta(days=best_lag)

    first_valid_idx = df_m2_shifted.first_valid_index()
    if first_valid_idx is not None:
        m2_ratio = df_m2_shifted / df_m2_shifted.loc[first_valid_idx]
    else:
        m2_ratio = df_m2_shifted

    ax_main.plot(asset_ratio.index, asset_ratio,
                 label=f"{asset_label} (Ratio)", color="tab:blue")
    ax_main.set_yscale('log')
    ax_main.set_xlabel("Date", fontsize=11, labelpad=8)
    ax_main.set_ylabel(f"{asset_label} (Log Ratio)", color="tab:blue", fontsize=11, labelpad=8)
    ax_main.tick_params(axis='y', labelcolor="tab:blue")
    ax_main.set_title(f"{asset_label} & Global M2 Leading Indicator (Ratio)", fontsize=13, pad=12)

    ax_main_r.plot(m2_ratio.index, m2_ratio,
                   label=f"M2 (Shifted {best_lag} days)", color="tab:orange")
    ax_main_r.set_ylabel("Global M2 (Ratio)", color="tab:orange", fontsize=11, labelpad=8)
    ax_main_r.tick_params(axis='y', labelcolor="tab:orange")

    # 6) Update x-limits from the slider
    vmin, vmax = range_slider.val
    x_start = start_date + pd.Timedelta(days=vmin)
    x_end = start_date + pd.Timedelta(days=vmax)
    ax_main.set_xlim(x_start, x_end)
    ax_main_r.set_xlim(x_start, x_end)

    fig.canvas.draw_idle()


def on_slider_change(val: float) -> None:
    """
    Callback for the slider that updates the x-limits of ax_main and ax_main_r
    based on the selected date range.
    """
    vmin, vmax = range_slider.val
    x_start = start_date + pd.Timedelta(days=vmin)
    x_end = start_date + pd.Timedelta(days=vmax)
    ax_main.set_xlim(x_start, x_end)
    ax_main_r.set_xlim(x_start, x_end)
    fig.canvas.draw_idle()


range_slider.on_changed(on_slider_change)


###############################################################################
# Button Callbacks
###############################################################################
def make_asset_callback(asset_label: str):
    """
    Return a callback function that plots the given asset vs. M2.
    """

    def callback(event):
        plot_asset_vs_m2(asset_label)

    return callback


# Assign callbacks to buttons
for label, btn in buttons.items():
    btn.on_clicked(make_asset_callback(label))

###############################################################################
# Initial Plot (BTC by default)
###############################################################################
plot_asset_vs_m2("BTC")

plt.show()
