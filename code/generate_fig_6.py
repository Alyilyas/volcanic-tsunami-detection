import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --- 1. CONFIGURATION ---

CONFIG = {
    # Analysis parameters
    "window_min": 10,
    "window_max": 1000,
    "window_step": 1,

    # File and data parameters
    # You can specify a single string OR a list
    "stations": "CGJI",#['CGJI', 'LWLI', 'MDSI', 'SBJI'], #or "SBJI", for a single station
    "components": ['BHE', 'BHN', 'BHZ'], #or "BHE", for a single component
    "events": {
        # for a single event
        "Flank collapse": ""
        # or multiple events at once using:
        #"Flank collapse": "",
        #"Volcanic 1": "1",
        #"Volcanic 2": "2",
        #"Volcanic 3": "3"

    },
    "motion_type": "d",        # 'd' for displacement
    "data_dir": Path("data"),     # folder where .txt files are stored
    "save_dir": Path("output/figures"),

    # Plotting parameters
    "figsize_base": (6, 5),    # base size per subplot (width, height)
    "title_fontsize": 24,
    "label_fontsize": 20,
    "axis_fontsize": 16,
}


# --- 2. HELPER FUNCTIONS ---

def ensure_list(x):
    """Ensure the variable is a list."""
    return x if isinstance(x, (list, tuple)) else [x]


def calculate_bic(seismic_data, window_size):
    """Calculates the Bayesian Information Criterion (BIC) for a given window size."""
    n = len(seismic_data)
    if window_size <= 0 or n < window_size:
        return np.nan
    m = n // window_size
    if m == 0:
        return np.nan
    total_variance = 0
    total_samples = 0
    for j in range(m):
        window = seismic_data[j * window_size:(j + 1) * window_size]
        if np.any(np.isnan(window)):
            continue
        var = np.var(window, ddof=1)
        if var > 0:
            total_variance += var * (window_size - 1)
            total_samples += (window_size - 1)
    if total_samples == 0:
        return np.nan
    total_variance /= total_samples
    bic = m * np.log(n) + window_size * (np.log(2 * np.pi * total_variance) + 1)
    return bic


def plot_bic_for_case(ax, station, component, event_suffix, config):
    """Loads data for a single case, calculates BIC, and plots on a given axis."""
    data_file = f"{config['motion_type']}{station}{component}{event_suffix}.txt"
    file_path = config["data_dir"] / data_file

    try:
        seismic_data = np.loadtxt(file_path)
        seismic_data = seismic_data / np.std(seismic_data)
        print(f"  Processing: {file_path}")
    except (FileNotFoundError, IOError):
        print(f"  ❌ ERROR: File not found for: '{file_path}'")
        ax.text(0.5, 0.5, 'Data not found', ha='center', va='center',
                color='red', fontsize=10)
        return

    window_sizes = range(config["window_min"], config["window_max"], config["window_step"])
    bic_values = []
    for window_size in window_sizes:
        bic = calculate_bic(seismic_data, window_size)
        if not np.isnan(bic):
            bic_values.append((window_size, bic))

    if not bic_values:
        print(f"  ⚠️ No valid BIC values computed for {file_path}.")
        ax.text(0.5, 0.5, 'No valid BIC values', ha='center', va='center')
        return

    optimal_window_size, _ = min(bic_values, key=lambda x: x[1])
    window_sizes_list, bic_list = zip(*bic_values)

    ax.plot(window_sizes_list, bic_list, marker='.', markersize=1, linestyle='-')
    ax.axvline(x=optimal_window_size, color='r', linestyle='--')

    ax.text(0.95, 0.95, f'optimal window size = {optimal_window_size}',
            ha='right', va='top', transform=ax.transAxes, fontsize=10,
            color='black', bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.2'))
    ax.grid(True)
    ax.tick_params(axis='both', which='major', labelsize=10)


# --- 3. MAIN WORKFLOW ---

def main():
    """Main function to perform BIC analysis for all stations, events, and components."""
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
                plot_bic_for_case(ax, station, component, event_suffix, CONFIG)

                if col_idx == 0:
                    ax.set_ylabel(event_name, fontsize=CONFIG["axis_fontsize"], labelpad=15)
                if row_idx == 0:
                    ax.set_title(component, fontsize=CONFIG["axis_fontsize"], pad=15)

        fig.supxlabel('Window size', fontsize=CONFIG["label_fontsize"], y=0.06)
        fig.supylabel(f'BIC score of {station}', fontsize=CONFIG["label_fontsize"], x=0.07)

        plt.tight_layout(rect=[0.08, 0.08, 1, 0.93])

        CONFIG["save_dir"].mkdir(parents=True, exist_ok=True)

        filename = f'SI_Figure_BIC_Station_{station}.png'
        full_path = CONFIG["save_dir"] / filename
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"✅ Figure for {station} saved successfully to: {full_path}")
        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
