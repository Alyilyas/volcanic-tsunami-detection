import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from scipy.stats import norm
from pathlib import Path

# --- 1. CONFIGURATION ---

CONFIG = {
    # Analysis parameters
    "window_size": 200,
    "step": 200,  # Step size for rolling window
    "alpha": 0.05,  # Significance level for Bartlett's C.I.
    "max_lag_acf": 5,  # Max ACF lag to check for violations
    "lags_range": list(range(1, 21)),  # AR lags to test
    "allowed_violations": 0,  # Max allowed ACF violations per window

    # File and data parameters
    # You can specify a single string OR a list
    "stations": ['CGJI', 'LWLI', 'MDSI', 'SBJI'], #or "SBJI", for a single station
    "components": ['BHE', 'BHN', 'BHZ'], #or "BHE", for a single component
    "events": {
        # for a single event:
        #"Flank collapse": ""
        # or multiple events at once using:
        "Flank collapse": "",
        "Volcanic 1": "1",
        "Volcanic 2": "2",
        "Volcanic 3": "3"
    },
    "motion_type": "d",  # 'd' for displacement
    "data_dir": Path("data"),
    "save_dir": Path("output/figures/ACF_Validation_Supplementary"),

    # Plotting parameters
    "figsize_base": (7, 5),  # base size per subplot (width, height)
    "title_fontsize": 24,
    "label_fontsize": 20,
    "axis_fontsize": 16,
    "legend_fontsize": 12,
}


# --- 2. HELPER FUNCTIONS ---

def ensure_list(x):
    """Ensure the variable is a list."""
    return x if isinstance(x, (list, tuple)) else [x]


def manual_acf(x, nlags):
    """Calculates the autocorrelation function manually."""
    x = x - np.mean(x)
    n = len(x)
    if n == 0:
        return np.zeros(nlags + 1)

    # Use variance for the denominator for an unbiased estimate at lag 0
    denom = np.var(x) * n
    if denom == 0:
        return np.zeros(nlags + 1)

    acf_vals = [1.0]
    for lag in range(1, nlags + 1):
        num = np.sum(x[:n - lag] * x[lag:])
        acf_vals.append(num / denom)
    return np.array(acf_vals)


def bartlett_confint(acf_vals, n, alpha=0.05):
    """Calculates Bartlett's formula for confidence intervals."""
    z = norm.ppf(1 - alpha / 2)  # Z-score for two-tailed C.I.
    confint = []
    for k in range(len(acf_vals)):
        if k == 0:
            se = 0.0  # No variance at lag 0
        else:
            # Bartlett's formula
            se = np.sqrt((1 + 2 * np.sum(acf_vals[1:k] ** 2)) / n)
        bound = z * se
        confint.append((-bound, bound))
    return np.array(confint)


def plot_acf_validation_for_case(ax, station, component, event_suffix, config):
    """
    Loads data for a single case, calculates % of non-independent windows
    for a range of AR lags, and plots the result on a given axis.
    """
    # --- Load parameters from config ---
    window_size = config["window_size"]
    step = config["step"]
    alpha = config["alpha"]
    max_lag_acf = config["max_lag_acf"]
    lags_range = config["lags_range"]
    allowed_violations = config["allowed_violations"]

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

    # --- Analysis Loop ---
    percent_not_independent = []
    for lag in lags_range:
        train_size = lag + window_size
        num_windows = (len(seismic_data) - train_size) // step + 1

        if num_windows <= 0:
            percent_not_independent.append(np.nan)  # No data for this lag
            continue

        not_independent_count = 0
        valid_window_count = 0
        for w in range(num_windows):
            start_idx = w * step
            end_idx = start_idx + train_size
            if end_idx > len(seismic_data):
                continue

            train = seismic_data[start_idx:end_idx]

            try:
                model = AutoReg(train, lags=lag, old_names=False).fit()
            except Exception:
                continue  # Skip windows that fail to fit

            valid_window_count += 1
            residuals = model.resid
            if len(residuals) < max_lag_acf:
                continue  # Not enough residuals to check

            acf_vals = manual_acf(residuals, nlags=max_lag_acf)
            confint = bartlett_confint(acf_vals, len(residuals), alpha=alpha)

            violation_count = 0
            for lag_i in range(1, max_lag_acf + 1):
                lower, upper = confint[lag_i]
                if not (lower <= acf_vals[lag_i] <= upper):
                    violation_count += 1

            if violation_count > allowed_violations:
                not_independent_count += 1

        if valid_window_count == 0:
            percent_not_independent.append(np.nan)
        else:
            perc = (not_independent_count / valid_window_count) * 100
            percent_not_independent.append(perc)

    # --- Plotting on the provided axis ---
    ax.plot(lags_range, percent_not_independent, marker='o', color='blue', markersize=5)
    ax.axhline(5, color='red', linestyle='--', label='5% Threshold')

    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', labelsize=10)
    ax.legend(fontsize=config["legend_fontsize"])
    ax.set_xticks(range(min(lags_range), max(lags_range) + 1, 2))


# --- 3. MAIN WORKFLOW ---

def main():
    """Main function to perform ACF validation for all stations, events, and components."""
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

        # Handle single subplot gracefully
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = np.array([axes])
        elif ncols == 1:
            axes = np.array([[ax] for ax in axes])

        for row_idx, (event_name, event_suffix) in enumerate(events.items()):
            for col_idx, component in enumerate(components):
                ax = axes[row_idx, col_idx]
                plot_acf_validation_for_case(ax, station, component, event_suffix, CONFIG)

                # Set column titles (Component)
                if row_idx == 0:
                    ax.set_title(component, fontsize=CONFIG["axis_fontsize"], pad=15)

                # Set row labels (Event Name) on the right
                if col_idx == (ncols - 1):
                    ax_twin = ax.twinx()
                    ax_twin.set_ylabel(event_name, fontsize=CONFIG["axis_fontsize"],
                                       rotation=270, labelpad=30)
                    ax_twin.set_yticks([])

        fig.supxlabel('AutoReg (AR) Lag ($p$)', fontsize=CONFIG["label_fontsize"], y=0.06)
        fig.supylabel('% of Non-independent windows', fontsize=CONFIG["label_fontsize"], x=0.07)
        fig.suptitle(f'Residual independence test - Station {station}',
                     fontsize=CONFIG["title_fontsize"], y=0.98)

        plt.tight_layout(rect=[0.08, 0.08, 0.95, 0.93])  # Adjust layout

        CONFIG["save_dir"].mkdir(parents=True, exist_ok=True)

        filename = f'SI_Figure_ACF_Validation_Station_{station}.png'
        full_path = CONFIG["save_dir"] / filename
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"✅ Figure for {station} saved successfully to: {full_path}")

        plt.close(fig)  # Close figure to save memory


if __name__ == "__main__":
    main()