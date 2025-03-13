import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    """Load CSV, filter by date, and select the 'close' price."""
    filepath = resource_path(filepath)
    df = pd.read_csv(filepath, parse_dates=['time'])
    df = df.sort_values('time').set_index('time')
    return df.loc[start_date:end_date, ['close']]

def compute_lag_correlations(df_merged, max_lag):
    """Compute correlation for different lag values (only positive lags)."""
    correlations = {}
    for lag in range(1, max_lag + 1):
        # Compare M2(t) with BTC(t+lag)
        shifted_btc = df_merged['close_btc'].shift(-lag)
        aligned = pd.concat([df_merged['close_m2'], shifted_btc], axis=1).dropna()
        if not aligned.empty:
            correlations[lag] = aligned['close_m2'].corr(aligned['close_btc'])
    return correlations

# ---------------------------
# Main Script
# ---------------------------
START_DATE = "2018-01-01"
END_DATE   = "2025-12-31"
MAX_LAG    = 180
FILE_M2    = 'data/Global Liquity.csv'
FILE_BTC   = 'data/Bitcoin Price.csv'

# Load datasets
df_m2 = load_and_filter_csv(FILE_M2, START_DATE, END_DATE)
df_btc = load_and_filter_csv(FILE_BTC, START_DATE, END_DATE)

# Rename columns for merging
df_m2_renamed = df_m2.rename(columns={'close': 'close_m2'})
df_btc_renamed = df_btc.rename(columns={'close': 'close_btc'})
df_merged = pd.merge(df_m2_renamed, df_btc_renamed, left_index=True, right_index=True, how='inner')

# Compute correlations and determine the optimal lag
correlations = compute_lag_correlations(df_merged, MAX_LAG)
optimal_lag = max(correlations, key=correlations.get, default=57)
optimal_corr = correlations.get(optimal_lag, 0)
print("Optimal lag (days):", optimal_lag)
print("Correlation coefficient:", optimal_corr)

# Create a figure with two subplots:
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Subplot 1: Correlation vs. Lag
ax1.plot(list(correlations.keys()), list(correlations.values()), marker='o')
ax1.axvline(optimal_lag, color='red', linestyle='--', label=f'Optimal Lag: {optimal_lag} days')
ax1.set_xlabel("Lag (days)")
ax1.set_ylabel("Correlation")
ax1.set_title("Correlation vs. Lag")
ax1.legend()
ax1.grid()

# Subplot 2: Overlay BTC Price and shifted Global M2
# Shift Global M2 by the optimal lag days
df_m2_shifted = df_m2.shift(periods=optimal_lag)

# Choose a scaling factor if needed (adjust as necessary)
scale_factor = 1.0

ax2.plot(df_btc.index, df_btc['close'], label="BTC Price", color="blue")
ax2.plot(df_m2_shifted.index, df_m2_shifted['close'] * scale_factor,
         label=f"Global M2 (Shifted by {optimal_lag} days)", color="orange")
ax2.set_xlabel("Date")
ax2.set_ylabel("Price / Global M2")
ax2.set_title("BTC Price with Global M2 Leading Indicator")
ax2.legend()
ax2.grid()
# Set y-axis to logarithmic scale for smoothing the long-term data
ax2.set_yscale('log')

plt.tight_layout()
plt.show()
