import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from scipy.fft import fft
from scipy.interpolate import interp1d
import os
from pathlib import Path
from datetime import datetime, timedelta

# --- 1. CONFIGURATION ---
CONFIG = {
    # Analysis parameters
    "threshold_percentile": 99.9,  # Anomaly detection threshold (for 99.9% CI)
    "fs": 20,  # Sampling rate in Hz
    # "ar_lag" is now set from the table below
    "train_points": 200,  # Points for AR model training window
    "pred_points": 200,  # Points for AR model prediction window
    "analysis_window": 200,  # Points for the final analysis window (bootstrap/FFT)
    "step": 200,  # Step size (20 for 1s overlapping step)
    "n_bootstrap": 100,  # Number of bootstrap iterations
    "target_frequency": 0.1,  # Target frequency in Hz for FFT analysis

    # --- Data and run parameters ---
    "stations_list": ['CGJI', 'LWLI', 'MDSI', 'SBJI'],
    "components_list": ['BHE', 'BHN', 'BHZ'],
    "motion_type": 'd',  # 'd' for displacement
    "data_dir": Path("data"),  # Assumes data files are in the data/ folder
    "save_dir": Path("output/figures/quick_test"),

    # --- AR Lag Lookup Table (from Manuscript Table 1) ---
    "AR_LAG_TABLE": {
        'CGJI': {'BHE': 10, 'BHN': 10, 'BHZ': 11},
        'SBJI': {'BHE': 10, 'BHN': 11, 'BHZ': 10},
        'LWLI': {'BHE': 9, 'BHN': 8, 'BHZ': 9},
        'MDSI': {'BHE': 12, 'BHN': 11, 'BHZ': 12}
    },

    # Plotting parameters
    "data_start_time_utc": "13:54:00",  # The UTC start time of the data files
    "event_start_time": 108,  # In seconds from data start (13:55:48 UTC)
    "seismic_arrival_time": None,  # This will be set automatically
    "plot_colors": {
        "Flank collapse": "blue",
        "Volcanic 1": "red",
        "Volcanic 2": "green",
        "Volcanic 3": "purple"
    }
}

# --- Arrival time mapping ---
STATION_ARRIVAL_TIMES = {
    "CGJI": 121,
    "SBJI": 125,
    "LWLI": 158,
    "MDSI": 150
}


# --- 2. HELPER FUNCTIONS ---

def load_seismic_data(file_path):
    """Loads seismic data from a single-column text file."""
    try:
        data = pd.read_csv(file_path, header=None)
        return data[0].astype(float).values
    except FileNotFoundError:
        print(f"    Error: Data file not found at {file_path}")
        return None


def get_fft_magnitude(data, freq, sampling_rate):
    """Calculates the FFT magnitude at a specific target frequency using interpolation."""
    data = np.asarray(data, dtype=float)
    n = len(data)
    if n == 0:
        return 0.0
    data = data - np.mean(data)
    fft_vals = np.abs(fft(data))[:n // 2]
    freqs = np.fft.fftfreq(n, d=1 / sampling_rate)[:n // 2]
    if freqs.size == 0:
        return 0.0
    interpolator = interp1d(freqs, fft_vals, kind='linear', bounds_error=False, fill_value=0.0)
    return float(interpolator(freq))


def perform_rolling_ar_forecast(data, config):
    """Fits an AR model in a rolling window to generate predictions and residuals."""
    train_size, pred_size, lag = config["train_points"], config["pred_points"], config["ar_lag"]
    window_size = train_size + pred_size
    predicted_data = np.full(len(data), np.nan, dtype=float)
    residuals = np.full(len(data), np.nan, dtype=float)
    if len(data) < window_size:
        return predicted_data, residuals

    for j in range(0, len(data) - window_size + 1, pred_size):
        train_data = data[j: j + train_size]
        true_values = data[j + train_size: j + window_size]
        try:
            model = AutoReg(train_data, lags=lag).fit()
            predictions = model.predict(start=len(train_data), end=len(train_data) + pred_size - 1, dynamic=True)
            predicted_data[j + train_size: j + window_size] = predictions
            residuals[j + train_size: j + window_size] = true_values - predictions
        except Exception as e:
            continue
    return predicted_data, residuals


def calculate_bootstrap_spectra(original_data, predicted_data, residuals, config, lower_p, upper_p):
    """Performs sliding window bootstrap analysis to get spectral confidence intervals."""
    window_size, step = config["analysis_window"], config["step"]
    time_stamps, magnitudes, lower_bounds, upper_bounds = [], [], [], []
    all_bootstrap_magnitudes = []

    if len(original_data) < window_size:
        return np.array([]), np.array([]), np.array([]), np.array([]), []

    for start_idx in range(0, len(original_data) - window_size + 1, step):
        end_idx = start_idx + window_size
        window_pred = predicted_data[start_idx:end_idx]
        window_resid = residuals[start_idx:end_idx]
        window_orig = original_data[start_idx:end_idx]

        if not (np.all(np.isfinite(window_pred)) and np.all(np.isfinite(window_resid))):
            continue

        bootstrap_magnitudes_for_window = []
        for _ in range(config["n_bootstrap"]):
            resampled_residuals = np.random.choice(window_resid, size=len(window_resid), replace=True)
            synthetic_signal = window_orig - resampled_residuals
            magnitude = get_fft_magnitude(synthetic_signal, config["target_frequency"], config["fs"])
            bootstrap_magnitudes_for_window.append(magnitude)

        if not bootstrap_magnitudes_for_window:
            continue

        all_bootstrap_magnitudes.append(bootstrap_magnitudes_for_window)
        magnitudes.append(get_fft_magnitude(window_orig, config["target_frequency"], config["fs"]))
        lower_bounds.append(np.percentile(bootstrap_magnitudes_for_window, lower_p))
        upper_bounds.append(np.percentile(bootstrap_magnitudes_for_window, upper_p))
        time_stamps.append((start_idx + window_size) / config["fs"])

    return (np.array(time_stamps), np.array(magnitudes), np.array(lower_bounds),
            np.array(upper_bounds), all_bootstrap_magnitudes)


def plot_results(all_results, threshold, exceeding_time, config, station, component):
    """Generates and saves the final plot with UTC time labels and window markers."""
    plt.figure(figsize=(12, 6))

    # Use the specific arrival time from the config
    seismic_arrival_time = config["seismic_arrival_time"]

    max_time = 0
    for event_name, result in all_results.items():
        if result['time'].size > 0:
            max_time = max(max_time, result['time'][-1])
            color = config["plot_colors"].get(event_name, 'gray')
            plt.plot(result['time'], result['magnitude'], label=event_name, color=color, linewidth=2)
            plt.fill_between(result['time'], result['lower'], result['upper'], color=color, alpha=0.3)

    window_duration_seconds = config["analysis_window"] / config["fs"]
    step_duration_seconds = config["step"] / config["fs"]

    if window_duration_seconds > 0:
        for i in np.arange(0, max_time + 1, window_duration_seconds):
            plt.axvline(x=i, color='gray', linestyle='--', linewidth=0.2)

    if step_duration_seconds > 0:
        for i in np.arange(0, max_time + 1, step_duration_seconds):
            plt.axvline(x=i, color='lightgray', linestyle='--', linewidth=0.1)

    start_dt = datetime.strptime(config["data_start_time_utc"], "%H:%M:%S")

    event_start_dt = start_dt + timedelta(seconds=config["event_start_time"])
    event_start_label = f"Event start time ({event_start_dt.strftime('%H:%M:%S')})"

    arrival_dt = start_dt + timedelta(seconds=seismic_arrival_time)
    arrival_label = f"Seismic wave arrival ({arrival_dt.strftime('%H:%M:%S')})"

    plt.axhline(y=threshold, color='black', linestyle='--', label=f"Threshold = {threshold:.5f}", linewidth=2)
    plt.axvline(x=config["event_start_time"], color='gray', linestyle='--', label=event_start_label, linewidth=2)
    plt.axvline(x=seismic_arrival_time, color='blue', linestyle='--', label=arrival_label, linewidth=2)

    if exceeding_time is not None:
        exceeding_dt = start_dt + timedelta(seconds=exceeding_time)
        exceeding_label = f"Threshold exceeded ({exceeding_dt.strftime('%H:%M:%S')})"
        plt.axvline(x=exceeding_time, color='red', linestyle='--', label=exceeding_label, linewidth=2)

    plt.xlabel("Time (seconds)", fontsize=18)
    plt.ylabel("FFT magnitude", fontsize=18)
    plt.title(f"Spectral detection: {station}-{component} (AR Lag: {config['ar_lag']}, Step: {config['step']}s)")
    plt.ylim(0, 0.0016)
    plt.legend(fontsize=14, loc='upper right')
    plt.tick_params(axis='both', labelsize=14)
    plt.grid(True)

    config["save_dir"].mkdir(parents=True, exist_ok=True)
    # Create a unique filename for each plot
    filename = f'fft_{config["motion_type"]}_{station}_{component}_{config["step"]}step.png'
    full_path = config["save_dir"] / filename
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    # plt.show() # Disabled for batch processing
    plt.close()  # Close the figure to save memory
    print(f"    Plot saved to {full_path}")


# --- 3. ANALYSIS FUNCTION (The old 'main') ---

def run_analysis_for_pair(station, component, ar_lag):
    """Runs the entire analysis workflow for a single station-component pair."""

    print(f"\n--- Starting analysis for {station}-{component} (AR Lag: {ar_lag}) ---")

    # --- Set temporary config values for this run ---
    # We modify the global CONFIG directly for this single-threaded script
    CONFIG["ar_lag"] = ar_lag
    CONFIG["station"] = station
    CONFIG["component"] = component

    # --- Calculate percentile values from CONFIG ---
    confidence_p = CONFIG["threshold_percentile"]
    alpha = (100.0 - confidence_p) / 100.0  # e.g., (100 - 99.9) / 100 = 0.001

    # Corrected two-sided CI for the plot
    lower_percentile = (alpha / 2.0) * 100.0  # e.g., 0.05
    upper_percentile = (1.0 - (alpha / 2.0)) * 100.0  # e.g., 99.95
    # One-sided threshold for detection
    threshold_percentile = (1.0 - alpha) * 100.0  # e.g., 99.9

    # --- Automatically set the arrival time ---
    station_code = CONFIG["station"]
    if station_code in STATION_ARRIVAL_TIMES:
        CONFIG["seismic_arrival_time"] = STATION_ARRIVAL_TIMES[station_code]
    else:
        print(f"    Warning: Arrival time for station '{station_code}' not defined. Defaulting to 0.")
        CONFIG["seismic_arrival_time"] = 0

    # --- Define data files dynamically ---
    data_files = {
        "Flank collapse": f"{CONFIG['motion_type']}{CONFIG['station']}{CONFIG['component']}.txt",
        "Volcanic 1": f"{CONFIG['motion_type']}{CONFIG['station']}{CONFIG['component']}1.txt",
        "Volcanic 2": f"{CONFIG['motion_type']}{CONFIG['station']}{CONFIG['component']}2.txt",
        "Volcanic 3": f"{CONFIG['motion_type']}{CONFIG['station']}{CONFIG['component']}3.txt"
    }

    all_results = {}
    max_fft_value = -np.inf

    for event_name, filename in data_files.items():
        print(f"  Processing {event_name} from {filename}...")
        file_path = CONFIG['data_dir'] / filename
        seismic_data = load_seismic_data(file_path)
        if seismic_data is None:
            continue

        predicted_data, residuals = perform_rolling_ar_forecast(seismic_data, CONFIG)
        time, mag, low, high, all_boot_mags = calculate_bootstrap_spectra(seismic_data, predicted_data, residuals,
                                                                          CONFIG, lower_percentile, upper_percentile)

        all_results[event_name] = {"time": time, "magnitude": mag, "lower": low, "upper": high}

        # Calculate threshold *only* from non-flank-collapse events
        if event_name != "Flank collapse":
            for window_boot_mags in all_boot_mags:
                if window_boot_mags:
                    percentile_val = np.percentile(window_boot_mags, threshold_percentile)
                    if np.isfinite(percentile_val):
                        max_fft_value = max(max_fft_value, percentile_val)

    if not np.isfinite(max_fft_value) or max_fft_value < 0:
        max_fft_value = 0.0

    print(f"  -> Analysis complete. Final threshold = {max_fft_value:.5f}")

    # --- Find detection time ---
    exceeding_time = None
    if "Flank collapse" in all_results:
        fc_results = all_results["Flank collapse"]
        if fc_results['magnitude'].size > 0:
            exceeding_indices = np.where(fc_results['magnitude'] > max_fft_value)[0]
            if len(exceeding_indices) > 0:
                exceeding_time = float(fc_results['time'][exceeding_indices[0]])
                print(f"  -> Threshold first exceeded at: {exceeding_time:.2f} seconds")

    # --- Plot results for this pair ---
    plot_results(all_results, max_fft_value, exceeding_time, CONFIG, station, component)


# --- 4. MAIN WORKFLOW (New master loop) ---

def main():
    """Main function to loop over all pairs and run the analysis."""
    stations_list = CONFIG["stations_list"]
    components_list = CONFIG["components_list"]

    print("===== Starting Batch Analysis for All Station-Component Pairs =====")

    for station in stations_list:
        for component in components_list:
            try:
                # Look up the specific AR lag for this pair
                ar_lag = CONFIG["AR_LAG_TABLE"][station][component]
                # Run the entire analysis and plotting for this pair
                run_analysis_for_pair(station, component, ar_lag)

            except KeyError:
                print(f"\n--- SKIPPING {station}-{component}: No AR Lag specified in AR_LAG_TABLE. ---")
            except Exception as e:
                print(f"\n--- FAILED: {station}-{component}. Error: {e} ---")
                # Continue to the next pair even if one fails
                pass

    print("\n===== Batch analysis complete. =====")


if __name__ == "__main__":
    main()