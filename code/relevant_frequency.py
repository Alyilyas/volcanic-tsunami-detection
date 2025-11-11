import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- 1. CONFIGURATION ---

CONFIG = {
    # Analysis parameters
    "fs": 20,  # Sampling frequency in Hz
    "freq_min": 0.02,
    "freq_max": 0.5,

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
    "file_prefix": "d",
    "data_dir": Path("data"),
    "save_dir": Path("output/figures/FFT_Supplementary"),

    # Plotting parameters
    "figsize_base": (8, 5),  # base size per subplot (width, height)
    "plot_colors": ['blue', 'red', 'green', 'purple'],
    "title_fontsize": 24,
    "label_fontsize": 20,
    "axis_fontsize": 16,
    "legend_fontsize": 12,
}


# --- 2. HELPER FUNCTIONS ---

def ensure_list(x):
    """Ensure the variable is a list."""
    return x if isinstance(x, (list, tuple)) else [x]


def plot_fft_for_case(ax, station, component, config):
    """
    Loads data for all events for a single station/component
    and plots their FFTs on the given axis.
    """
    event_names = list(config["events"].keys())
    event_suffixes = list(config["events"].values())
    plot_colors = config["plot_colors"]

    has_data = False
    for i, event_name in enumerate(event_names):
        event_suffix = event_suffixes[i]
        color = plot_colors[i % len(plot_colors)]  # Cycle through colors

        file_name = f"{config['file_prefix']}{station}{component}{event_suffix}.txt"
        file_path = config["data_dir"] / file_name

        try:
            # Load data
            data = np.loadtxt(file_path)

            # FFT Calculation
            n = len(data)
            data_detrended = data - np.mean(data)  # Detrend
            freqs = np.fft.rfftfreq(n, d=1 / config["fs"])
            fft_vals = np.abs(np.fft.rfft(data_detrended)) * (2 / n)  # Normalize

            # Frequency filtering
            mask = (freqs >= config["freq_min"]) & (freqs <= config["freq_max"])

            # Plot
            ax.plot(freqs[mask], fft_vals[mask],
                    color=color,
                    label=event_name)
            has_data = True

        except (FileNotFoundError, IOError):
            print(f"  File not found: {file_path}")
            continue
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
            continue

    if has_data:
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(axis='both', labelsize=10)
        ax.legend(fontsize=config["legend_fontsize"])
        ax.set_yscale('log')  # Use log scale for amplitude
    else:
        print(f"  ❌ No data plotted for {station}-{component}")
        ax.text(0.5, 0.5, 'No data found', ha='center', va='center',
                color='red', fontsize=10)


# --- 3. MAIN WORKFLOW ---

def main():
    """Main function to perform FFT analysis for all stations and components."""
    stations = ensure_list(CONFIG["stations"])
    components = ensure_list(CONFIG["components"])

    for station in stations:
        print(f"\n--- Generating figure for station: {station} ---")

        nrows = 1
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
        if ncols == 1:
            axes = np.array([axes])

        for col_idx, component in enumerate(components):
            ax = axes[col_idx]
            plot_fft_for_case(ax, station, component, CONFIG)

            # Set column titles (Component)
            ax.set_title(component, fontsize=CONFIG["axis_fontsize"], pad=15)

        fig.supxlabel(f"Frequency (Hz) [{CONFIG['freq_min']} - {CONFIG['freq_max']}]",
                      fontsize=CONFIG["label_fontsize"], y=0.06)
        fig.supylabel('Normalized FFT amplitude (log scale)',
                      fontsize=CONFIG["label_fontsize"], x=0.07)
        fig.suptitle(f'FFT analysis - Station {station}',
                     fontsize=CONFIG["title_fontsize"], y=0.98)

        plt.tight_layout(rect=[0.08, 0.08, 1, 0.93])

        CONFIG["save_dir"].mkdir(parents=True, exist_ok=True)

        filename = f'SI_Figure_FFT_Station_{station}.png'
        full_path = CONFIG["save_dir"] / filename
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"✅ Figure for {station} saved successfully to: {full_path}")

        plt.close(fig)  # Close figure to save memory


if __name__ == "__main__":
    main()