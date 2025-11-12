import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from scipy.fft import fft
from scipy.interpolate import interp1d
import os
from pathlib import Path

# --- 1. Configuration ---
CONFIG = {
    # Analysis parameters
    "fs": 20,  # Sampling rate in Hz
    # "ar_lag" is REMOVED from here. It will be pulled from the table below.
    "train_points": 200,  # Points for AR model training window
    "pred_points": 200,  # Points for AR model prediction window
    "analysis_window": 200,  # Points for the final analysis window (bootstrap/FFT)
    "step": 20,  # Step size (20 for overlapping with 1s step)
    "n_bootstrap": 100,  # Number of bootstrap iterations
    "target_frequency": 0.1,  # Target frequency in Hz for FFT analysis

    # --- Data and run parameters ---
    "stations_list": ['CGJI', 'LWLI', 'MDSI', 'SBJI'],
    "components_list": ['BHE', 'BHN', 'BHZ'],
    "motion_type": 'd',  # 'd' for displacement
    "data_dir": Path("data"),  # Assumes data files are in the data/ folder
    "save_dir": Path("output"),  # Directory to save the final table
}

# --- AR Lag Lookup Table ---
# This table provides the specific, optimal AR lag for each station-component pair.
# Based on Manuscript Table 1 (Mean column, rounded to nearest integer).
AR_LAG_TABLE = {
    'CGJI': {
        'BHE': 10,  # Mean 10.25
        'BHN': 10,  # Mean 9.50
        'BHZ': 11  # Mean 11.25
    },
    'SBJI': {
        'BHE': 10,  # Mean 9.50
        'BHN': 11,  # Mean 10.50
        'BHZ': 10  # Mean 10.00
    },
    'LWLI': {
        'BHE': 9,  # Mean 8.50
        'BHN': 8,  # Mean 7.75
        'BHZ': 9  # Mean 8.50
    },
    'MDSI': {
        'BHE': 12,  # Mean 11.50
        'BHN': 11,  # Mean 11.00
        'BHZ': 12  # Mean 11.50
    }
}


# --- 2. Helper functions (Unchanged) ---

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
    # This function now relies on CONFIG["ar_lag"] being set correctly *before* it's called
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


def calculate_bootstrap_spectra(original_data, predicted_data, residuals, config):
    """Performs sliding window bootstrap analysis to get spectral confidence intervals."""
    window_size, step = config["analysis_window"], config["step"]
    all_bootstrap_magnitudes = []

    if len(original_data) < window_size:
        return []

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

        if bootstrap_magnitudes_for_window:
            all_bootstrap_magnitudes.append(bootstrap_magnitudes_for_window)

    return all_bootstrap_magnitudes


# --- 3. Main workflow (Modified to use AR_LAG_TABLE) ---

def main():
    """Main function to run the threshold generation for all station-components."""

    stations_list = CONFIG["stations_list"]
    components_list = CONFIG["components_list"]

    all_thresholds = []
    print("Starting threshold generation for all station-components...")

    for station in stations_list:
        for component in components_list:

            # --- NEW: Get the specific AR lag for this pair ---
            try:
                current_ar_lag = AR_LAG_TABLE[station][component]
                print(f"\nProcessing {station}-{component} (using AR Lag: {current_ar_lag})...")
                # Set the lag in the config for this loop iteration
                CONFIG["ar_lag"] = current_ar_lag
            except KeyError:
                print(f"\nWarning: No AR lag found for {station}-{component} in AR_LAG_TABLE. Skipping.")
                continue
            # --- End of new block ---

            # 1. Update CONFIG for file names
            CONFIG["station"] = station
            CONFIG["component"] = component

            # 2. Define data files
            data_files = {
                "Volcanic 1": f"{CONFIG['motion_type']}{CONFIG['station']}{CONFIG['component']}1.txt",
                "Volcanic 2": f"{CONFIG['motion_type']}{CONFIG['station']}{CONFIG['component']}2.txt",
                "Volcanic 3": f"{CONFIG['motion_type']}{CONFIG['station']}{CONFIG['component']}3.txt"
            }

            max_fft_value = -np.inf

            # 4. Run the analysis on baseline events
            for event_name, filename in data_files.items():
                print(f"  Processing baseline file: {filename}")
                file_path = CONFIG['data_dir'] / filename
                seismic_data = load_seismic_data(file_path)

                if seismic_data is None:
                    continue

                    # Pass the config, which now has the *correct* ar_lag
                predicted_data, residuals = perform_rolling_ar_forecast(seismic_data, CONFIG)
                all_boot_mags = calculate_bootstrap_spectra(seismic_data, predicted_data, residuals, CONFIG)

                for window_boot_mags in all_boot_mags:
                    if window_boot_mags:
                        percentile_val = np.percentile(window_boot_mags, 99.9)
                        if np.isfinite(percentile_val):
                            max_fft_value = max(max_fft_value, percentile_val)

            if not np.isfinite(max_fft_value) or max_fft_value < 0:
                max_fft_value = 0.0

            print(f"  -> Final threshold for {station}-{component}: {max_fft_value:.5f}")

            all_thresholds.append({
                "Station": station,
                "Component": component,
                "Threshold": max_fft_value
            })

    # --- Loops finished, create and display the table ---
    print("\nOverall analysis complete.")

    threshold_table = pd.DataFrame(all_thresholds)

    try:
        threshold_table_pivot = threshold_table.pivot(
            index="Station",
            columns="Component",
            values="Threshold"
        )

        threshold_table_pivot = threshold_table_pivot.reindex(columns=components_list)

        print("\nThreshold table (pivoted format):")
        print(threshold_table_pivot.to_string(float_format="%.5f"))

        CONFIG["save_dir"].mkdir(parents=True, exist_ok=True)
        table_filename = CONFIG["save_dir"] / "detection_threshold_table.csv"
        threshold_table_pivot.to_csv(table_filename, float_format="%.5f")
        print(f"\nPivoted table saved to {table_filename}")

    except Exception as e:
        print(f"Could not pivot table. Error: {e}")
        print("\nThreshold table (flat format):")
        print(threshold_table.to_string(float_format="%.5f"))

        CONFIG["save_dir"].mkdir(parents=True, exist_ok=True)
        table_filename_flat = CONFIG["save_dir"] / "detection_threshold_list.csv"
        threshold_table.to_csv(table_filename_flat, index=False, float_format="%.5f")
        print(f"Flat list saved to {table_filename_flat}")


if __name__ == "__main__":
    main()