# volcanic-tsunami-detection

A statistical framework for the near real-time detection of tsunami-generating volcanic flank collapses.



This repository contains the data and code for the manuscript (Ilyas et al, 2026). The analysis framework is designed to detect tsunami-generating volcanic flank collapses from single-station seismic data in near real-time, using the 2018 Anak Krakatau event as a case study.



The core of the methodology involves:

1\.  Fitting a rolling autoregressive (AR) model to seismic data to generate predictions and residuals.

2\.  Using bootstrap resampling on the residuals to construct a 99.9% prediction interval for the signal's spectral magnitude.

3\.  Establishing a conservative detection threshold based on the most extreme spectral values from powerful, non-tsunamigenic baseline eruptions.

4\.  Applying this threshold to the flank collapse event to test for rapid detection.



---



## 📂 Repository structure



\-   `/code`: Contains the main Python script `generate\_spectral\_detection\_plot.py`for producing Fig 11 & 12 (also figures in Appendix B) to determine the system threshold.

\-   `/data`: Contains the sample dataset used to run the analysis.

\-   `/output/figures`: The default directory where generated figures are saved.

\-   `requirements.txt`: The file listing all required Python packages for reproducibility.



---

## 📊 Data availability

The full seismic dataset used in this study is restricted and was provided by the Indonesian Meteorological, Climatological, and Geophysical Agency (BMKG). Access for research purposes can be requested directly from BMKG.

To ensure the methods are transparent and reproducible, a sample dataset is included in the `/data` directory. This sample contains short time windows of the 2018 flank collapse event and the three baseline volcanic eruptions, formatted as single-column `.txt` files. The analysis script is configured to run using this sample data.

---

## ⚙️ Setup instructions

This guide provides step-by-step instructions for setting up the project. Please follow the guide for your operating system:

### macOS guide 

On macOS, the best way to manage developer tools is with **Homebrew**, a package manager for command-line tools.

#### 1. Prerequisites

* **Homebrew:** If you don't have it, open the **Terminal** app and run this command:
    ```bash
    /bin/bash -c "$(curl -fsSL [https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh](https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh))"
    ```

* **Git & Python 3.12:** Once Homebrew is ready, install Git and Python 3.12:
    ```bash
    brew install git python@3.12
    ```

#### 2. Installation steps

1.  **Open the Terminal App**
    Find it in `Applications/Utilities` or use Spotlight (`⌘ + Space`).

2.  **Clone the repository**
    ```bash
    git clone [https://github.com/Alyilyas/volcanic-tsunami-detection.git](https://github.com/Alyilyas/volcanic-tsunami-detection.git)
    ```
    ```bash
    cd volcanic-tsunami-detection
    ```

3.  **Create and activate a virtual environment**
    ```bash
    python3 -m venv venv
    ```
    ```bash
    source venv/bin/activate
    ```

4.  **Install required packages**
    ```bash
    pip3 install -r requirements.txt
    ```

---
### Windows guide ❖

#### 1. Prerequisites

Before you begin, you need to install Git and a specific version of Python.

* **Python 3.12:** This project requires **Python 3.12** to ensure compatibility with all scientific packages. Newer versions (like 3.13) may cause installation errors.
    1.  Download the **Windows installer (64-bit)** from the official Python website: [https://www.python.org/downloads/release/python-3124/](https://www.python.org/downloads/release/python-3124/)
    2.  Run the installer. **Crucially, check the box that says "Add python.exe to PATH"** on the first screen. 

* **Git:** This is the tool used to download the repository from GitHub.
    1.  Download and install Git for Windows: [https://git-scm.com/download/win](https://git-scm.com/download/win)
    2.  During installation, accept the default settings. Ensure you are on the page "Adjusting your PATH environment" and that the recommended option, **"Git from the command line and also from 3rd-party software"**, is selected.

### 2. Installation steps

1.  **Open command prompt**
    Press the Windows key, type `cmd`, and press Enter.

2.  **Clone the repository**
    This command downloads the project files to your computer. Navigate to a directory where you want to store the project (e.g., your Desktop or Documents folder) and run:
    ```bash
    git clone https://github.com/Alyilyas/volcanic-tsunami-detection.git
    ```
    ```bash
    cd volcanic-tsunami-detection
    ```

3.  **Create and activate a virtual environment**
    A virtual environment is an isolated space for the project's dependencies.
    ```bash
    python -m venv venv
    ```
    ```bash
    venv\Scripts\activate
    ```
    Your command prompt should now start with `(venv)`.

4.  **Install required packages**
    This command installs all necessary Python libraries from the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 How to run the analysis

To run the default analysis (for station `SBJI`, component `BHE`, with non-overlapping windows):
```bash
python code/generate_spectral_detection_plot.py
```


The script will process the four events, calculate the threshold, determine the detection time for the flank collapse, and display the final plot. The plot will also be saved to the `/output/figures` directory.



\### Customizing the analysis



To reproduce all the figures in the manuscript, simply modify the parameters in the `CONFIG` dictionary within the script.



\### Key parameters to change:



\-   `"station"`: Change to `'CGJI'`, `'LWLI'`, or `'MDSI'` to analyze a different station.

\-   `"component"`: Change to `'BHN'` (North-South) or `'BHZ'` (Vertical).

\-   `"step"`:

&nbsp;   -   `200`: For \*\*non-overlapping\*\* windows (as in Fig. 11 of the manuscript).

&nbsp;   -   `20`: For \*\*1-second overlapping\*\* windows (as in Fig. 12 of the manuscript).

\-   `"n\_bootstrap"`:

&nbsp;   -   `100`: For quick tests and debugging.

&nbsp;   -   `10000`: For producing the final, high-quality results for the manuscript. (Note: this will be computationally intensive).

\-   `"ar\_lag"`: Set the AR model lag order. This should be consistent with the values determined in your analysis (e.g., 10 or 11) or refer to table 1 in the manuscript.



\### Example: Reproducing the overlapping window plot for station SBJI (BHE component)



1\.  Open `code/generate\_spectral\_detection\_plot.py`.

2\.  Modify the `CONFIG` dictionary:

```python

CONFIG = {

# ...

"step": 20,              # Set to 20 for 1-second overlap

"station": 'SBJI',

"component": 'BHE',

# ...

}

```

3\.  Run the script from your terminal:

```bash 
python code/generate\_spectral\_detection\_plot.py

```



---



## 📝 Citation



If you use this code or methodology in your research, please cite our manuscript:



> \[Ilyas, A., et al. (2026). A statistical framework for the near real-time seismic detection of tsunami-generating volcanic flank collapses Focused on Anak krakatau.]

