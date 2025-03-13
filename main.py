import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import RangeSlider, Button
import sys
import os


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller."""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


# ---------------------------
# Data processing functions
# ---------------------------
def load_and_filter_csv(filepath, start_date, end_date):
    """
    Load CSV data, filter by date, and select the 'close' column.
    Assumes the CSV has a 'time' column to parse dates.
    """
    filepath = resource_path(filepath)
    df = pd.read_csv(filepath, parse_dates=['time'])
    df = df.sort_values('time').set_index('time')
    return df.loc[start_date:end_date, ['close']]


def compute_lag_correlations(series_m2, series_asset, max_lag, method='pearson', window_size=60):
    """
    Compute the correlation between M2(t) and Asset(t+lag) for lags in [1, max_lag].

    Instead of a single global correlation, a rolling window correlation is computed
    over 'window_size' days (if enough data points exist), then averaged. This rolling
    average correlation is more robust to nonstationarity in the data.

    Parameters:
      series_m2  : Pandas Series for Global M2 (should be aligned by date)
      series_asset: Pandas Series for the asset (e.g., BTC, NASDAQ, etc.)
      max_lag    : Maximum lag (in days) to search. 180 days is chosen to capture
                   medium-term lead/lag effects.
      method     : Correlation method ('pearson', 'spearman', or 'kendall')
      window_size: Size of the rolling window (in days) used to compute average correlation.

    Returns:
      A dictionary mapping each lag (1 to max_lag) to its average rolling correlation.
    """
    correlations = {}
    for lag in range(1, max_lag + 1):
        # Shift the asset series by 'lag' days (negative shift means asset(t+lag) aligns with M2(t))
        shifted_asset = series_asset.shift(-lag)
        aligned = pd.concat([series_m2.rename("m2"), shifted_asset.rename("asset")], axis=1).dropna()

        if len(aligned) >= window_size:
            # Compute rolling correlation over the defined window.
            # Using min_periods=window_size ensures only complete windows are considered.
            rolling_corr = aligned["m2"].rolling(window=window_size, min_periods=window_size).corr(aligned["asset"])
            # If the rolling correlation produces valid numbers, take the mean; else fallback to simple correlation.
            if rolling_corr.notna().sum() > 0:
                correlation = rolling_corr.mean()
            else:
                correlation = aligned["m2"].corr(aligned["asset"], method=method)
        else:
            # Fallback to global correlation if there aren't enough data points.
            correlation = aligned["m2"].corr(aligned["asset"], method=method)
        correlations[lag] = correlation
    return correlations


# ---------------------------
# Main Script Parameters
# ---------------------------
START_DATE = "2018-01-01"
END_DATE = "2025-12-31"
MAX_LAG = 180  # Maximum lag in days; chosen to capture medium-term effects.
CORR_METHOD = 'pearson'  # Correlation method; consider 'spearman' if non-linear relationships are suspected.
WINDOW_SIZE = 60  # Rolling window size in days for computing average correlations.

# ---------------------------
# Load Data Files
# ---------------------------
# Global M2 Supply
df_m2 = load_and_filter_csv('data/Global Liquity.csv', START_DATE, END_DATE).sort_index()
df_m2 = df_m2.rename(columns={'close': 'close_m2'})

# Assets
df_btc = load_and_filter_csv('data/Bitcoin Price.csv', START_DATE, END_DATE).sort_index()
df_btc = df_btc.rename(columns={'close': 'close_btc'})

df_nasdaq = load_and_filter_csv('data/NASDAQ Price.csv', START_DATE, END_DATE).sort_index()
df_nasdaq = df_nasdaq.rename(columns={'close': 'close_nasdaq'})

df_us10y = load_and_filter_csv('data/US10Y.csv', START_DATE, END_DATE).sort_index()
df_us10y = df_us10y.rename(columns={'close': 'close_us10y'})

df_dxy = load_and_filter_csv('data/DXY.csv', START_DATE, END_DATE).sort_index()
df_dxy = df_dxy.rename(columns={'close': 'close_dxy'})

df_dji = load_and_filter_csv('data/DJI Price.csv', START_DATE, END_DATE).sort_index()
df_dji = df_dji.rename(columns={'close': 'close_dji'})

df_chainlink = load_and_filter_csv('data/Chainlink Price.csv', START_DATE, END_DATE).sort_index()
df_chainlink = df_chainlink.rename(columns={'close': 'close_chainlink'})

# Build a dictionary of assets for easy switching.
assets = {
    "BTC": df_btc["close_btc"],
    "NASDAQ": df_nasdaq["close_nasdaq"],
    "US10Y": df_us10y["close_us10y"],
    "DXY": df_dxy["close_dxy"],
    "DJI": df_dji["close_dji"],
    "Chainlink": df_chainlink["close_chainlink"]
}

# -------------------------------------------------------
# Figure, Subplots, Slider, and Button Setup
# -------------------------------------------------------
fig = plt.figure(figsize=(18, 7))

# Left subplot: Correlation vs. Lag
ax_corr = fig.add_subplot(1, 2, 1)
# Right subplot: Ratio overlay (with twin y-axis for M2)
ax_main = fig.add_subplot(1, 2, 2)
ax_main_r = None  # Will be created/recreated in our plotting function

# Union of all dates (M2 plus all asset dates) for slider range
all_dates = (df_m2.index
             .union(df_btc.index)
             .union(df_nasdaq.index)
             .union(df_us10y.index)
             .union(df_dxy.index)
             .union(df_dji.index)
             .union(df_chainlink.index))
min_date = all_dates.min()
max_date = all_dates.max()
proj_max_date = max_date + pd.Timedelta(days=MAX_LAG)

# For the slider, use "days from start" (start_date = min_date)
start_date = min_date
min_day = 0
max_day = (max_date - start_date).days
proj_max_day = max_day + MAX_LAG

# Slider axis
slider_ax = fig.add_axes([0.125, 0.05, 0.775, 0.03])
range_slider = RangeSlider(slider_ax, "Days from Start", min_day, proj_max_day, valinit=(min_day, proj_max_day))

# Button axes for each asset
btc_button = Button(fig.add_axes([0.05, 0.92, 0.12, 0.05]), "BTC")
nasdaq_button = Button(fig.add_axes([0.18, 0.92, 0.12, 0.05]), "NASDAQ")
us10y_button = Button(fig.add_axes([0.31, 0.92, 0.12, 0.05]), "US10Y")
dxy_button = Button(fig.add_axes([0.44, 0.92, 0.12, 0.05]), "DXY")
dji_button = Button(fig.add_axes([0.57, 0.92, 0.12, 0.05]), "DJI")
chainlink_button = Button(fig.add_axes([0.70, 0.92, 0.12, 0.05]), "Chainlink")


# -------------------------------------------------------
# Plotting Function: Asset vs. M2 Analysis
# -------------------------------------------------------
def plot_asset_vs_m2(asset_label):
    """
    Perform the following steps:
      1. Clear previous plots.
      2. Compute rolling-window correlation vs. lag between M2 and the selected asset.
         This yields an 'optimal' lag based on the average correlation over a window.
      3. Plot the correlation vs. lag on the left subplot.
      4. Shift the M2 series by the optimal lag and compute ratio series for both the asset
         (plotted on a logarithmic scale) and M2 (plotted on a linear scale) on the right subplot.
      5. Update x-axis limits based on the current slider selection.
    """
    global ax_main_r

    # Clear previous plots
    ax_corr.clear()
    ax_main.clear()
    if ax_main_r:
        ax_main_r.remove()
    ax_main_r = ax_main.twinx()

    # 1) Select series (dropping missing data)
    series_asset = assets[asset_label].dropna()
    series_m2 = df_m2["close_m2"].dropna()

    # 2) Compute correlation vs. lag using a rolling window average
    correlations = compute_lag_correlations(series_m2, series_asset, MAX_LAG, method=CORR_METHOD,
                                            window_size=WINDOW_SIZE)
    if correlations:
        optimal_lag = max(correlations, key=correlations.get)
        optimal_corr = correlations[optimal_lag]
    else:
        optimal_lag = 0
        optimal_corr = 0

    print(f"Asset: {asset_label}")
    print(f"Optimal lag (days): {optimal_lag}")
    print(f"Correlation coefficient: {optimal_corr}")

    # 3) Plot correlation vs. lag on the left subplot with an annotation for optimal lag
    ax_corr.plot(list(correlations.keys()), list(correlations.values()), marker='o')
    ax_corr.axvline(optimal_lag, color='red', linestyle='--', label=f'Optimal Lag: {optimal_lag} days')
    ax_corr.set_xlabel("Lag (days)")
    ax_corr.set_ylabel("Correlation")
    ax_corr.set_title(f"Correlation vs. Lag ({asset_label} vs. M2)")
    ax_corr.legend()
    ax_corr.grid()

    # 4) Prepare ratio series for the overlay chart
    # (A) Compute asset ratio relative to its first available value.
    df_asset_sorted = series_asset.sort_index()
    asset_ratio = df_asset_sorted / df_asset_sorted.iloc[0]

    # (B) Shift M2 by adding the optimal lag (in days) to its index.
    df_m2_shifted = series_m2.sort_index().copy()
    df_m2_shifted.index = df_m2_shifted.index + pd.Timedelta(days=optimal_lag)

    # (C) Compute M2 ratio relative to its first value after shifting.
    first_valid_idx = df_m2_shifted.first_valid_index()
    if first_valid_idx is not None:
        m2_ratio = df_m2_shifted / df_m2_shifted.loc[first_valid_idx]
    else:
        m2_ratio = df_m2_shifted

    # 5) Plot the overlay on the right subplot:
    #    - Left y-axis: Asset ratio on a logarithmic scale.
    #    - Right y-axis: Global M2 ratio on a linear scale.
    ax_main.plot(asset_ratio.index, asset_ratio, label=f"{asset_label} (Ratio)", color="blue")
    ax_main.set_yscale('log')
    ax_main.set_xlabel("Date")
    ax_main.set_ylabel(f"{asset_label} (Log Ratio)", color="blue")
    ax_main.tick_params(axis='y', labelcolor="blue")

    ax_main_r.plot(m2_ratio.index, m2_ratio, label=f"M2 (Shifted {optimal_lag} days)", color="orange")
    ax_main_r.set_ylabel("Global M2 (Ratio)", color="orange")
    ax_main_r.tick_params(axis='y', labelcolor="orange")

    ax_main.set_title(f"{asset_label} & Global M2 Leading Indicator (Ratio)")

    # 6) Update x-axis limits based on the current slider selection ("Days from Start")
    vmin, vmax = range_slider.val
    x_start = start_date + pd.Timedelta(days=vmin)
    x_end = start_date + pd.Timedelta(days=vmax)
    ax_main.set_xlim(x_start, x_end)
    ax_main_r.set_xlim(x_start, x_end)
    fig.canvas.draw_idle()


# -------------------------------------------------------
# Slider Callback
# -------------------------------------------------------
def on_slider_change(val):
    vmin, vmax = range_slider.val
    x_start = start_date + pd.Timedelta(days=vmin)
    x_end = start_date + pd.Timedelta(days=vmax)
    ax_main.set_xlim(x_start, x_end)
    ax_main_r.set_xlim(x_start, x_end)
    fig.canvas.draw_idle()


range_slider.on_changed(on_slider_change)


# -------------------------------------------------------
# Button Callbacks for Each Asset
# -------------------------------------------------------
def on_btc_click(event):
    plot_asset_vs_m2("BTC")


def on_nasdaq_click(event):
    plot_asset_vs_m2("NASDAQ")


def on_us10y_click(event):
    plot_asset_vs_m2("US10Y")


def on_dxy_click(event):
    plot_asset_vs_m2("DXY")


def on_dji_click(event):
    plot_asset_vs_m2("DJI")


def on_chainlink_click(event):
    plot_asset_vs_m2("Chainlink")


btc_button.on_clicked(on_btc_click)
nasdaq_button.on_clicked(on_nasdaq_click)
us10y_button.on_clicked(on_us10y_click)
dxy_button.on_clicked(on_dxy_click)
dji_button.on_clicked(on_dji_click)
chainlink_button.on_clicked(on_chainlink_click)

# -------------------------------------------------------
# Initial Plot (BTC by default)
# -------------------------------------------------------
plot_asset_vs_m2("BTC")

plt.tight_layout(rect=[0, 0.1, 1, 0.9])  # leave space at top for buttons
plt.show()
