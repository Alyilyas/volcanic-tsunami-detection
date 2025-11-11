import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.sandbox.stats.runs import runstest_1samp
from pathlib import Path

# --- 1. CONFIGURATION ---

CONFIG = {
    # Analysis parameters
    "window_size": 200,
    "lags_range": list(range(1, 21)),  # Must be a list or range
    "runs_test_p_value": 0.05,  # p-value threshold for randomness

    # File and data parameters
    # You can specify a single string OR a list
    "stations": ['CGJI', 'LWLI', 'MDSI', 'SBJI'], #or "SBJI", for a single station
    "components": ['BHE', 'BHN', 'BHZ'], #or "BHE", for a single component
    "events": {
        # for a single event
        #"Flank collapse": ""
        # or multiple events at once using:
        "Flank collapse": "",
        "Volcanic 1": "1",
        "Volcanic 2": "2",
        "Volcanic 3": "3"
    },
    "motion_type": "d",  # 'd' for displacement
    "data_dir": Path("data"),
    "save_dir": Path("output/figures/AR_Lag_Supplementary"),

    # Plotting parameters
    "figsize_base": (6, 5),  # base size per subplot (width, height)
    "title_fontsize": 24,
    "label_fontsize": 20,
    "axis_fontsize": 16,
    "legend_fontsize": 12,
}


# --- 2. HELPER FUNCTIONS ---

def ensure_list(x):
    """Ensure the variable is a list."""
    return x if isinstance(x, (list, tuple)) else [x]


def calculate_bic_ar(residuals, k):
    """Calculates the Bayesian Information Criterion for AR model residuals."""
    n = len(residuals)
    if n == 0:
        return np.nan
    resid_var = np.var(residuals)
    if resid_var == 0:  # Avoid log(0)
        return np.nan
    # BIC formula for AR model
    bic = np.log(n) * k + n * (np.log(2 * np.pi * resid_var) + 1)
    return bic


def plot_lag_dist_for_case(ax, station, component, event_suffix, config):
    """
    Loads data for a single case, calculates optimal AR lags, and plots the
    distribution on a given Matplotlib axis object.
    """
    # --- Load parameters from config ---
    window_size = config["window_size"]
    lags_range = config["lags_range"]
    p_value_threshold = config["runs_test_p_value"]

    # --- Load Data ---
    data_file = f"{config['motion_type']}{station}{component}{event_suffix}.txt"
    file_path = config["data_dir"] / data_file

    try:
        data = pd.read_csv(file_path, header=None)
        seismic_data = data[0].values
        print(f"  Processing: {file_path}")
    except FileNotFoundError:
        print(f"  ❌ ERROR: File not found for: '{file_path}'")
        ax.text(0.5, 0.5, f'File not found:\n{data_file}', ha='center',
                va='center', color='red', fontsize=10)
        return

    # --- Per-window storage ---
    num_windows = (len(seismic_data) - max(lags_range) - window_size) // window_size + 1
    if num_windows <= 0:
        print(f"  ⚠️ Not enough data to create any windows for {data_file}.")
        ax.text(0.5, 0.5, 'Not enough data for analysis', ha='center', va='center')
        return

    window_bic = np.full((num_windows, len(lags_range)), np.nan)
    window_pval = np.full((num_windows, len(lags_range)), np.nan)

    # --- Loop over lags and windows ---
    for lag_idx, lag in enumerate(lags_range):
        for w in range(num_windows):
            start_idx = w * window_size
            end_idx = start_idx + window_size + lag
            train = seismic_data[start_idx:end_idx]

            try:
                # Fit AR model
                model = AutoReg(train, lags=lag, old_names=False).fit()
                residuals = model.resid[-window_size:]
                if len(residuals) < window_size:
                    continue

                # Validate residuals for randomness
                _, p_val = runstest_1samp(residuals)
                bic = calculate_bic_ar(residuals, k=lag)

                window_bic[w, lag_idx] = bic
                window_pval[w, lag_idx] = p_val
            except Exception:
                continue  # Skip windows that fail to fit

    # --- Determine optimal lag for each window ---
    optimal_lags = []
    for w in range(num_windows):
        # Find lags that pass the randomness test
        valid_mask = window_pval[w, :] > p_value_threshold
        if np.any(valid_mask):
            # Of the valid lags, find the one that minimizes BIC
            valid_bics = window_bic[w, :][valid_mask]
            valid_lags_range = np.array(list(lags_range))[valid_mask]

            best_lag = valid_lags_range[np.nanargmin(valid_bics)]
            optimal_lags.append(best_lag)
        else:
            # No lag passed the randomness test for this window
            optimal_lags.append(np.nan)

    # --- Plotting ---
    valid_lags = [lag for lag in optimal_lags if not np.isnan(lag)]
    if not valid_lags:
        print(f"  ⚠️ No optimal lags found for {data_file}.")
        ax.text(0.5, 0.5, 'No optimal lags found', ha='center', va='center')
        return

    mean_lag = np.mean(valid_lags)
    median_lag = np.median(valid_lags)

    # Plot histogram
    bins = np.arange(min(lags_range) - 0.5, max(lags_range) + 1.5, 1)
    ax.hist(valid_lags, bins=bins, edgecolor='black', color='skyblue')

    # Plot mean/median lines
    ax.axvline(mean_lag, color='red', linestyle='--', linewidth=2,
               label=f'Mean = {mean_lag:.2f}')
    ax.axvline(median_lag, color='green', linestyle=':', linewidth=2.5,
               label=f'Median = {median_lag:.0f}')

    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', labelsize=10)
    ax.legend(fontsize=config["legend_fontsize"])
    ax.set_xticks(range(min(lags_range), max(lags_range) + 1, 2))


# --- 3. MAIN WORKFLOW ---

def main():
    """Main function to perform AR Lag analysis for all stations, events, and components."""
    stations = ensure_list(CONFIG["stations"])
    components = ensure_list(CONFIG["components"])
    events = CONFIG["events"]

    for station in stations:
        print(f"\n--- Generating figure for station: {station} ---")

        nrows = len(events)
        ncols = len(components)
        figsize = (CONFIG["figsize_base"][0] * ncols,
                   CONFIG["figsize_base"][1] * nrows)

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize,
            sharex=True,
            sharey=True
        )

        # Handle single subplot gracefully (like in your Fig. 6 script)
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = np.array([axes])
        elif ncols == 1:
            axes = np.array([[ax] for ax in axes])

        for row_idx, (event_name, event_suffix) in enumerate(events.items()):
            for col_idx, component in enumerate(components):
                ax = axes[row_idx, col_idx]
                plot_lag_dist_for_case(ax, station, component, event_suffix, CONFIG)

                # Set column titles (Component)
                if row_idx == 0:
                    ax.set_title(component, fontsize=CONFIG["axis_fontsize"], pad=15)

                # Set row labels (Event Name)
                # We put this on the *right* side to avoid crowding the supylabel
                if col_idx == (ncols - 1):
                    ax_twin = ax.twinx()
                    ax_twin.set_ylabel(event_name, fontsize=CONFIG["axis_fontsize"],
                                       rotation=270, labelpad=30)
                    ax_twin.set_yticks([])

        fig.supxlabel('Optimal AR lag ($p$)', fontsize=CONFIG["label_fontsize"], y=0.06)
        fig.supylabel('Count (number of windows)', fontsize=CONFIG["label_fontsize"], x=0.07)
        fig.suptitle(f'Distribution of optimal AR lags - Station {station}',
                     fontsize=CONFIG["title_fontsize"], y=0.98)

        plt.tight_layout(rect=[0.08, 0.08, 0.95, 0.93])  # Adjust layout

        CONFIG["save_dir"].mkdir(parents=True, exist_ok=True)

        filename = f'SI_Figure_AR_Lag_Station_{station}.png'
        full_path = CONFIG["save_dir"] / filename
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"✅ Figure for {station} saved successfully to: {full_path}")
        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()