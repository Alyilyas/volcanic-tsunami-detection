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
    "fs": 20,  # Sampling rate in Hz
    "ar_lag": 11,  # Lag for the AR model
    "train_points": 200,  # Points for AR model training window
    "pred_points": 200,  # Points for AR model prediction window
    "analysis_window": 200,  # Points for the final analysis window (bootstrap/FFT)
    "step": 20,  # Step size (200 for non-overlapping, <200 for overlapping, 20 for overlapping with 1s step)
    "n_bootstrap": 100,  # Number of bootstrap iterations, pls change to 10000
    "target_frequency": 0.1,  # Target frequency in Hz for FFT analysis

    # File and data parameters
    "station": 'SBJI',
    "component": 'BHE',
    "motion_type": 'd',  # 'd' for displacement, 'v' for velocity, 'a' for acceleration
    #"data_dir": Path("."),  # Assumes data files are in the same directory as the script
    "data_dir": Path("data"),  # Assumes data files are in the other folder we need to specify the path
    "save_dir": Path("output/figures"),

    # Plotting parameters
    "data_start_time_utc": "13:54:00",  # The UTC start time of the data files
    "event_start_time": 108,  # In seconds from data start above, pre-defined 17 April 2024 Eruption happened around 13:55:48 UTC
    "seismic_arrival_time": None,  # This will be set automatically below
    "plot_colors": {
        "17 April 2024 Eruption": "blue",
        "Volcanic 1": "red",
        "Volcanic 2": "green",
        "Volcanic 3": "purple"
    }
}

# --- Arrival time mapping pre-defined based on previous research---
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
        print(f"Error: Data file not found at {file_path}")
        return None


def get_fft_magnitude(data, freq, sampling_rate):
    """Calculates the FFT magnitude at a specific target frequency using interpolation."""
    data = np.asarray(data, dtype=float)
    n = len(data)
    if n == 0:
        return 0.0
    data = data - np.mean(data)  # Reduce DC component, optional but helpful for low frequency
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
            # Leave NaNs for failed windows, they will be skipped
            continue
    return predicted_data, residuals


def calculate_bootstrap_spectra(original_data, predicted_data, residuals, config):
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
        lower_bounds.append(np.percentile(bootstrap_magnitudes_for_window, 0.05))
        upper_bounds.append(np.percentile(bootstrap_magnitudes_for_window, 99.95))
        time_stamps.append((start_idx + window_size) / config["fs"])

    return (np.array(time_stamps), np.array(magnitudes), np.array(lower_bounds),
            np.array(upper_bounds), all_bootstrap_magnitudes)


def plot_results(all_results, threshold, exceeding_time, config):
    """Generates and saves the final plot with UTC time labels and window markers."""
    plt.figure(figsize=(12, 6))

    max_time = 0
    for event_name, result in all_results.items():
        if result['time'].size > 0:
            # Find the maximum time value across all events to set the plot range
            max_time = max(max_time, result['time'][-1])
            color = config["plot_colors"].get(event_name, 'gray')
            plt.plot(result['time'], result['magnitude'], label=event_name, color=color, linewidth=2)
            plt.fill_between(result['time'], result['lower'], result['upper'], color=color, alpha=0.3)

    # Calculate durations in seconds from the config
    window_duration_seconds = config["analysis_window"] / config["fs"]
    step_duration_seconds = config["step"] / config["fs"]

    # Add vertical dashed lines at every analysis window (main divisions)
    if window_duration_seconds > 0:
        for i in np.arange(0, max_time + 1, window_duration_seconds):
            plt.axvline(x=i, color='gray', linestyle='--', linewidth=0.2)

    # Add thinner, lighter vertical lines at every step (subdivisions)
    if step_duration_seconds > 0:
        for i in np.arange(0, max_time + 1, step_duration_seconds):
            plt.axvline(x=i, color='lightgray', linestyle='--', linewidth=0.1)

    # Convert base time string to a datetime object for labeling
    start_dt = datetime.strptime(config["data_start_time_utc"], "%H:%M:%S")

    # Calculate and format the timestamps for the legend
    #event_start_dt = start_dt + timedelta(seconds=config["event_start_time"])
    #event_start_label = f"Event Start Time ({event_start_dt.strftime('%H:%M:%S')})"

    #arrival_dt = start_dt + timedelta(seconds=config["seismic_arrival_time"])
    #arrival_label = f"Seismic Wave Arrival ({arrival_dt.strftime('%H:%M:%S')})"

    # Draw threshold and event markers with new UTC labels
    plt.axhline(y=threshold, color='black', linestyle='--', label=f"Threshold = {threshold:.5f}", linewidth=2)
    #plt.axvline(x=config["event_start_time"], color='gray', linestyle='--', label=event_start_label, linewidth=2)
    #plt.axvline(x=config["seismic_arrival_time"], color='blue', linestyle='--', label=arrival_label, linewidth=2)

    if exceeding_time is not None:
        exceeding_dt = start_dt + timedelta(seconds=exceeding_time)
        exceeding_label = f"Threshold exceeded ({exceeding_dt.strftime('%H:%M:%S')})"
        plt.axvline(x=exceeding_time, color='red', linestyle='--', label=exceeding_label, linewidth=2)

    plt.xlabel("Time (seconds)", fontsize=18)
    plt.ylabel("FFT magnitude", fontsize=18)
    plt.ylim(0, 0.0016)
    plt.legend(fontsize=14, loc='upper right')
    plt.tick_params(axis='both', labelsize=14)
    plt.grid(True)

    config["save_dir"].mkdir(parents=True, exist_ok=True)
    filename = f'fft_{config["motion_type"]}_{config["station"]}_{config["component"]}_{config["step"]}step.png'
    full_path = config["save_dir"] / filename
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.show()


# --- 3. MAIN WORKFLOW ---

def main():
    """Main function to run the entire analysis workflow."""

    # --- Automatically set the arrival time, the time based on previous study ---
    '''
    station_code = CONFIG["station"]
    if station_code in STATION_ARRIVAL_TIMES:
        CONFIG["seismic_arrival_time"] = STATION_ARRIVAL_TIMES[station_code]
    else:
        print(f"Warning: Arrival time for station '{station_code}' not defined. Defaulting to None.")
        CONFIG["seismic_arrival_time"] = 0
    '''
    data_files = {
        "Volcanic 1": f"{CONFIG['motion_type']}{CONFIG['station']}{CONFIG['component']}1.txt",
        "Volcanic 2": f"{CONFIG['motion_type']}{CONFIG['station']}{CONFIG['component']}2.txt",
        "Volcanic 3": f"{CONFIG['motion_type']}{CONFIG['station']}{CONFIG['component']}3.txt"
    }

    all_results = {}
    max_fft_value = -np.inf

    print("--- Starting analysis ---")
    for event_name, filename in data_files.items():
        print(f"Processing {event_name} from {filename}...")
        file_path = CONFIG['data_dir'] / filename
        seismic_data = load_seismic_data(file_path)
        if seismic_data is None:
            continue

        predicted_data, residuals = perform_rolling_ar_forecast(seismic_data, CONFIG)
        time, mag, low, high, all_boot_mags = calculate_bootstrap_spectra(seismic_data, predicted_data, residuals,
                                                                          CONFIG)

        all_results[event_name] = {"time": time, "magnitude": mag, "lower": low, "upper": high}

        if event_name != "17 April 2024 Eruption":
            for window_boot_mags in all_boot_mags:
                if window_boot_mags:
                    percentile_val = np.percentile(window_boot_mags, 99.9)
                    if np.isfinite(percentile_val):
                        max_fft_value = max(max_fft_value, percentile_val)

    if not np.isfinite(max_fft_value) or max_fft_value < 0:
        max_fft_value = 0.0

    print(f"--- Analysis complete. Final threshold = {max_fft_value:.5f} ---")

    exceeding_time = None

    plot_results(all_results, max_fft_value, exceeding_time, CONFIG)


if __name__ == "__main__":
    main()