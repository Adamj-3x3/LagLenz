import os
import sys

# Adjust for PyInstaller's extracted location
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS  # PyInstaller temp extraction folder
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")

# Example: Load CSV
csv_file = os.path.join(DATA_DIR, "DJI Price.csv")

if os.path.exists(csv_file):
    print("CSV file found:", csv_file)
else:
    print("ERROR: CSV file missing:", csv_file)


import logging
from typing import Dict, Tuple

import pandas as pd
import numpy as np
import matplotlib

# Use a Qt-based backend that works with PySide6 (Qt6)
matplotlib.use("QtAgg")  # or "Qt6Agg" in newer Matplotlib

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import pyplot as plt

# PySide6 imports (instead of PyQt5)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox
)
from PySide6.QtCore import Qt, QThread, Signal


###############################################################################
# Logging Setup
###############################################################################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


###############################################################################
# Global Parameters
###############################################################################
START_DATE = "2018-01-01"
END_DATE = "2025-12-31"
MAX_LAG = 180          # Maximum lag in days
CORR_METHOD = 'pearson'

# Example CSV Paths. Adjust to your actual file paths or resource paths.
CSV_PATHS = {
    "M2":        "data/Global Liquity.csv",
    "BTC":       "data/Bitcoin Price.csv",
    "NASDAQ":    "data/NASDAQ Price.csv",
    "US10Y":     "data/US10Y.csv",
    "DXY":       "data/DXY.csv",
    "DJI":       "data/DJI Price.csv",
    "Chainlink": "data/Chainlink Price.csv"
}


###############################################################################
# Data Loading and Computation Functions
###############################################################################
def load_csv(filepath: str) -> pd.DataFrame:
    """
    Load a CSV file and return a DataFrame with a DatetimeIndex.
    Expects columns: time, close.
    """
    if not os.path.exists(filepath):
        logging.error(f"File not found: {filepath}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(filepath, parse_dates=['time'])
        df = df.sort_values('time').set_index('time')
        return df
    except Exception as e:
        logging.exception(f"Error loading CSV {filepath}: {e}")
        return pd.DataFrame()


def compute_lag_correlations(series_m2: pd.Series,
                             series_asset: pd.Series,
                             max_lag: int,
                             method: str = 'pearson') -> Dict[int, float]:
    """
    For each lag in [1..max_lag], shift the asset by -lag (so we compare M2(t)
    with Asset(t+lag)) and compute a single correlation across the entire dataset.

    Returns {lag -> correlation_value}
    """
    correlations = {}
    for lag in range(1, max_lag + 1):
        shifted_asset = series_asset.shift(-lag)
        df_aligned = pd.concat([
            series_m2.rename("m2"),
            shifted_asset.rename("asset")
        ], axis=1).dropna()

        if not df_aligned.empty:
            correlations[lag] = df_aligned["m2"].corr(df_aligned["asset"], method=method)
        else:
            correlations[lag] = np.nan
    return correlations


###############################################################################
# Matplotlib Canvas
###############################################################################
class MplCanvas(FigureCanvas):
    """
    A custom Matplotlib canvas that we'll embed in our PySide6 MainWindow.
    """
    def __init__(self, parent=None, width=10, height=6, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.axes_left = self.figure.add_subplot(1, 2, 1)
        self.axes_right = self.figure.add_subplot(1, 2, 2)

        super().__init__(self.figure)
        self.setParent(parent)

        # Some Matplotlib defaults
        self.figure.tight_layout()


###############################################################################
# Worker Thread
###############################################################################
class CorrelationWorker(QThread):
    """
    A QThread to handle correlation computations without blocking the UI.
    """
    # PySide6 uses 'Signal' instead of 'pyqtSignal'
    finished = Signal(str, int, float, object)
    # asset_label, best_lag, best_corr, correlations_dict

    def __init__(self, asset_label: str, m2_series: pd.Series, asset_series: pd.Series):
        super().__init__()
        self.asset_label = asset_label
        self.m2_series = m2_series
        self.asset_series = asset_series

    def run(self):
        correlations = compute_lag_correlations(self.m2_series, self.asset_series, MAX_LAG, CORR_METHOD)
        if correlations:
            best_lag = max(correlations, key=correlations.get)
            best_corr = correlations[best_lag]
        else:
            best_lag = 0
            best_corr = float('nan')

        self.finished.emit(self.asset_label, best_lag, best_corr, correlations)


###############################################################################
# Main Window
###############################################################################
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Investment Correlation App - PySide6 Edition")
        self.resize(1200, 700)

        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        # Top panel: asset selection + plot button
        top_panel = QHBoxLayout()
        main_layout.addLayout(top_panel)

        self.asset_combo = QComboBox()
        self.asset_combo.addItems(["BTC", "NASDAQ", "US10Y", "DXY", "DJI", "Chainlink"])
        top_panel.addWidget(QLabel("Select Asset:"))
        top_panel.addWidget(self.asset_combo)

        self.plot_button = QPushButton("Plot Correlation")
        top_panel.addWidget(self.plot_button)

        # Connect the button
        self.plot_button.clicked.connect(self.on_plot_button_clicked)

        # The Matplotlib canvas
        self.canvas = MplCanvas(self, width=10, height=4, dpi=100)
        main_layout.addWidget(self.canvas)

        # We'll track any twin axis so we can remove it before re-plotting
        self.ax_m2 = None

        # Load data at startup
        self.df_m2 = load_csv(CSV_PATHS["M2"]).rename(columns={'close': 'close_m2'})
        self.assets = {
            "BTC":       load_csv(CSV_PATHS["BTC"]).rename(columns={'close': 'close_btc'}),
            "NASDAQ":    load_csv(CSV_PATHS["NASDAQ"]).rename(columns={'close': 'close_nasdaq'}),
            "US10Y":     load_csv(CSV_PATHS["US10Y"]).rename(columns={'close': 'close_us10y'}),
            "DXY":       load_csv(CSV_PATHS["DXY"]).rename(columns={'close': 'close_dxy'}),
            "DJI":       load_csv(CSV_PATHS["DJI"]).rename(columns={'close': 'close_dji'}),
            "Chainlink": load_csv(CSV_PATHS["Chainlink"]).rename(columns={'close': 'close_chainlink'})
        }

        # Sort indexes
        self.df_m2.sort_index(inplace=True)
        for key, df_asset in self.assets.items():
            df_asset.sort_index(inplace=True)

        # Optional: correlation cache if you want to skip re-computing
        self.correlation_cache: Dict[str, Dict[int, float]] = {}

    def on_plot_button_clicked(self):
        """
        Triggered when the user clicks "Plot Correlation".
        We'll compute correlation in a background thread,
        then update the chart upon completion.
        """
        asset_label = self.asset_combo.currentText()
        # Retrieve data
        m2_series = self.df_m2["close_m2"].dropna()
        asset_series = self.assets[asset_label][f"close_{asset_label.lower()}"].dropna()

        # Start a worker thread to do correlation
        self.worker = CorrelationWorker(asset_label, m2_series, asset_series)
        self.worker.finished.connect(self.on_correlation_finished)
        self.worker.start()

    def on_correlation_finished(self,
                                asset_label: str,
                                best_lag: int,
                                best_corr: float,
                                correlations: object):
        """
        Slot called when CorrelationWorker finishes. We update the plots here.

        'correlations' is actually a dict[int, float], but declared as 'object'
        in the Signal to avoid issues with generic aliases.
        """
        correlations_dict = correlations  # type: Dict[int, float]

        # Clear the main left and right axes
        self.canvas.axes_left.cla()
        self.canvas.axes_right.cla()

        # If there's an existing twin axis from a previous plot, remove it
        if self.ax_m2 is not None:
            self.canvas.figure.delaxes(self.ax_m2)
            self.ax_m2 = None

        # --- LEFT AXES: Correlation vs Lag ---
        if correlations_dict:
            lags = sorted(correlations_dict.keys())
            corr_vals = [correlations_dict[lag] for lag in lags]
            self.canvas.axes_left.plot(lags, corr_vals, marker='o', color='tab:blue')
            self.canvas.axes_left.axvline(best_lag, color='red', linestyle='--',
                                          label=f'Optimal Lag: {best_lag} days')
            self.canvas.axes_left.set_xlabel("Lag (days)")
            self.canvas.axes_left.set_ylabel("Correlation")
            self.canvas.axes_left.set_title(f"Correlation vs. Lag\n({asset_label} vs. M2)")
            self.canvas.axes_left.legend(fontsize=10)
            self.canvas.axes_left.grid(True)

            # Annotate best correlation
            self.canvas.axes_left.annotate(
                f"Best Lag: {best_lag}\nCorr: {best_corr:.2f}",
                xy=(0.03, 0.97),
                xycoords='axes fraction',
                ha='left', va='top',
                fontsize=11,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', alpha=0.8)
            )
        else:
            self.canvas.axes_left.text(0.5, 0.5, "No Correlation Data",
                                       ha='center', va='center',
                                       transform=self.canvas.axes_left.transAxes,
                                       fontsize=12)

        # --- RIGHT AXES: Plot ratio overlays ---
        # shift M2 by best_lag, compute ratio
        m2_series = self.df_m2["close_m2"].dropna().copy()
        m2_series.index = m2_series.index + pd.Timedelta(days=best_lag)

        asset_series = self.assets[asset_label][f"close_{asset_label.lower()}"].dropna()
        if not asset_series.empty:
            asset_ratio = asset_series / asset_series.iloc[0]
        else:
            asset_ratio = asset_series

        first_valid_idx = m2_series.first_valid_index()
        if first_valid_idx is not None and not m2_series.empty:
            m2_ratio = m2_series / m2_series.loc[first_valid_idx]
        else:
            m2_ratio = m2_series

        # Plot asset ratio (log scale)
        self.canvas.axes_right.plot(asset_ratio.index, asset_ratio,
                                    label=f"{asset_label} (Ratio)", color="tab:blue")
        self.canvas.axes_right.set_yscale('log')
        self.canvas.axes_right.set_ylabel(f"{asset_label} (Log Ratio)", color="tab:blue")
        self.canvas.axes_right.set_xlabel("Date")
        self.canvas.axes_right.tick_params(axis='y', labelcolor="tab:blue")
        self.canvas.axes_right.set_title(f"{asset_label} & Global M2 Leading Indicator (Ratio)")

        # Create a brand-new twin axis for the M2 ratio
        self.ax_m2 = self.canvas.axes_right.twinx()
        self.ax_m2.plot(m2_ratio.index, m2_ratio, label="M2 Ratio", color="tab:orange")
        self.ax_m2.set_ylabel("Global M2 (Ratio)", color="tab:orange")
        self.ax_m2.tick_params(axis='y', labelcolor="tab:orange")

        # Redraw
        self.canvas.draw_idle()


###############################################################################
# Application Entry Point
###############################################################################
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    # PySide6 uses app.exec() instead of app.exec_()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
